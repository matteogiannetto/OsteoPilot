import os
import argparse
import time
import gc
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.ndimage
import skimage.morphology
from skimage import measure, filters
from skimage.measure import regionprops
from scipy.ndimage import find_objects
from ifb_framework import Quantity
from ifb_framework.registration import compute_initial_guess

# --- GPU Setup with Diagnostics ---
GPU_ERROR_MSG = None
try:
    import cupy as cp
    from cupyx.scipy import ndimage as ndi_gpu
    # Try a dummy allocation to verify CUDA context works
    _ = cp.array([1])
    USE_GPU_GLOBAL = True
    print("[ INFO ] GPU detected and CuPy is functional.")
except Exception as e:
    USE_GPU_GLOBAL = False
    GPU_ERROR_MSG = str(e)

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def left_bound(a):
    return int(np.around((abs(a)+a)/2))

def right_bound_x(a, scan_shape):
    return int(min(np.around(a), scan_shape[0]))

def right_bound_y(a, scan_shape):
    return int(min(np.around(a), scan_shape[1]))

def right_bound_z(a, scan_shape):
    return int(min(np.around(a), scan_shape[2]))

def mesh_volume(faces, verts):
    volume = 0
    for face in faces:
        volume += signed_vol(verts[face[0]], verts[face[1]], verts[face[2]])
    return abs(volume)


def legacy_len_from_vertices(vol_verts, centroid_local):
    """
    Legacy Len:
    len_temp = (max(|dx|)^3 + max(|dy|)^3 + max(|dz|)^3)^(1/3)
    where dx = verts[:,0] - cx, etc.
    """
    dx = np.abs(vol_verts[:, 0] - centroid_local[0])
    dy = np.abs(vol_verts[:, 1] - centroid_local[1])
    dz = np.abs(vol_verts[:, 2] - centroid_local[2])
    return ((np.max(dx) ** 3) + (np.max(dy) ** 3) + (np.max(dz) ** 3)) ** (1.0 / 3.0)


def signed_vol(p1, p2, p3):
    v321 = p3[0]*p2[1]*p1[2]
    v231 = p2[0]*p3[1]*p1[2]
    v312 = p3[0]*p1[1]*p2[2]
    v132 = p1[0]*p3[1]*p2[2]
    v213 = p2[0]*p1[1]*p3[2]
    v123 = p1[0]*p2[1]*p3[2]
    return (1.0/6.0)*(-v321 + v231 + v312 - v132 - v213 + v123)

def clear_border_cpu(labels, buffer_size=0, bgval=0):
    """
    Clears objects touching the border.
    Optimized to avoid creating large lists.
    """
    mask = np.zeros(labels.shape, dtype=bool)
    
    # X-borders
    mask[:buffer_size, :, :] = True
    mask[-buffer_size:, :, :] = True
    # Y-borders
    mask[:, :buffer_size, :] = True
    mask[:, -buffer_size:, :] = True
    # Z-borders
    mask[:, :, :buffer_size] = True
    mask[:, :, -buffer_size:] = True
    
    # Find labels that overlap with the border mask
    border_labels = np.unique(labels[mask])
    
    # Remove 0 (background) if present
    if border_labels.size > 0 and border_labels[0] == 0:
        border_labels = border_labels[1:]
        
    if border_labels.size > 0:
        # Create a boolean mask for all pixels belonging to border labels
        # np.isin can be slow for huge arrays, but safer than iterating
        pixel_mask = np.isin(labels, border_labels)
        labels[pixel_mask] = bgval
        
    return labels

def calculate_overlap_blocks(start_index, end_index, block_size, overlap_percentage):
    if overlap_percentage >= 100:
        raise ValueError("Overlap percentage must be less than 100%")
    
    overlap_size = int(block_size * overlap_percentage / 100)
    step_size = block_size - overlap_size
    
    if step_size <= 0:
        raise ValueError(f"Invalid block configuration: Size {block_size} with overlap {overlap_percentage}% results in step size {step_size}")

    blocks = []
    z_curr = start_index
    
    # Loop until we cover the requested range
    while z_curr < end_index:
        # Determine the end of the current block
        z_block_end = min(z_curr + block_size, end_index)
        
        blocks.append((z_curr, z_block_end))
        
        # If this block reached the global end, stop
        if z_block_end >= end_index:
            break
            
        z_curr += step_size
        
    return blocks

def remove_duplicate_lacunae(df, tolerance=1e-6):
    if len(df) == 0: return df
    print("   -> Starting deduplication...")
    df = df.copy()
    # Dynamic decimal places based on tolerance
    decimal_places = max(0, -int(np.log10(tolerance)))
    
    # Create a temporary key for deduplication
    df['centroid_key'] = list(zip(
        np.round(df['Centroid X'], decimal_places),
        np.round(df['Centroid Y'], decimal_places),
        np.round(df['Centroid Z'], decimal_places)
    ))
    
    df_unique = df.drop_duplicates(subset=['centroid_key'], keep='first')
    df_unique = df_unique.drop(columns=['centroid_key'])
    df_unique = df_unique.reset_index(drop=True)
    df_unique['Lacunar ID'] = np.arange(1, len(df_unique) + 1)
    return df_unique

def load_nifti_block_proxy(dataobj, z_start, z_end, dtype=None):
    block = np.asanyarray(dataobj[:, :, z_start:z_end])
    if dtype is not None:
        block = block.astype(dtype, copy=False)
    return block


def calculate_lacunae_features(lacunae_stats_df, lac_density_vol):
    from scipy.spatial.distance import pdist, squareform
    
    features = {'count_lacunae': len(lacunae_stats_df)}
    features['Volume Fraction of Lacunae in Bone Tissue'] = lac_density_vol

    cols_map = {
        'Volume (um^3)': 'volume',
        'Surface Area (um^2)': 'surface_area',
        'Surface Area to Vol Ratio (um)': 'sa_vol_ratio',
        'Major Axis (um) (radius) (from PCA on volume)': 'major_axis',
        'Minor Axis (um) (radius) (from PCA on volume)': 'minor_axis',
        'Minor to Major Axis Ratio': 'axis_ratio'
    }

    for col, name in cols_map.items():
        if col in lacunae_stats_df.columns:
            data = lacunae_stats_df[col]
            features.update({
                f'mean_{name}': np.mean(data),
                f'std_{name}': np.std(data),
                f'min_{name}': np.min(data),
                f'max_{name}': np.max(data),
                f'median_{name}': np.median(data),
                f'q25_{name}': np.percentile(data, 25),
                f'q75_{name}': np.percentile(data, 75)
            })
    return pd.DataFrame([features])

# -------------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------------

def main(
    nr_slice_per_block: int = 100,
    sample_name: str = None,
    path_original_volume: str = None,
    path_seg: str = None,
    processed: bool = True,
    parent_folder: str = None,
    overlap_percentage: int = 30,
    slice_start: int = 0,      
    slice_end: int = None,   
    save_block_files: bool = False,    
):
    # 1. Initialize Local GPU Flag safely
    use_gpu_flag = USE_GPU_GLOBAL

    STRICT_LEGACY_MARCHING_CUBES = True  # legacy would crash if marching_cubes fails


    if not use_gpu_flag:
        print(f"[ WARN ] GPU is NOT being used. Reason: {GPU_ERROR_MSG}")
        print("[ INFO ] Continuing with CPU (this will be slower)...")

    if sample_name is None or path_original_volume is None or path_seg is None:
        raise ValueError("Missing required paths or sample name.")

    # Paths Setup
    vol_dir = os.path.dirname(os.path.abspath(path_original_volume))
    if parent_folder is None:
        parent_folder = os.path.join(vol_dir, "lacunae_analysis")
    
    data_parent_folder = os.path.abspath(parent_folder)
    sample_out_dir = os.path.join(data_parent_folder, sample_name, "3D_params_files")
    os.makedirs(sample_out_dir, exist_ok=True)

    print(f"[ INFO ] Saving results to: {sample_out_dir}")

    # Temporary file
    temp_csv_path = os.path.join(sample_out_dir, f"temp_{sample_name}_cumulative.csv")
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

    # Parameters
    voxelsize = Quantity(0.0016, "mm")
    voxelsize_um = voxelsize.magnitude * 1000.0  # mm -> um

    boxsize = 1
    edgewidth = 2
    minfilter = 58
    maxfilter = 900
    
    # Constants for columns
    index_centroid_x = "Centroid X"
    index_centroid_y = "Centroid Y"
    index_centroid_z = "Centroid Z"
    index_vol_um = "Volume (um^3)"
    index_sa_vol = "Surface Area to Vol Ratio (raw)"
    index_LcSAV_ratio = "Surface Area to Vol Ratio (um)"
    index_sa_um = "Surface Area (um^2)"
    index_maj = "Major Axis (um) (radius) (from PCA on volume)"
    index_mna = "Minor Axis (um) (radius) (from PCA on volume)"
    index_ax_rat = "Minor to Major Axis Ratio"
    index_tot_lac_num = "Total Number of Lacunae"
    index_stretch_lacunae = "index Lc_St"
    index_lacunae_oblateness = "index Lc_Ob"
    index_Lc_Or_2_x = "index Lc.Or2_x"
    index_Lc_Or_2_y = "index Lc.Or2_y"
    index_Lc_Or_2_z = "index Lc.Or2_z"
    index_Lc_Or_1_x = "index Lc.Or1_x"
    index_Lc_Or_1_y = "index Lc.Or1_y"
    index_Lc_Or_1_z = "index Lc.Or1_z"

    # Volume accumulation
    lacunar_volume_blocks = []
    background_pixels_blocks = []
    total_vol_bone_blocks = []
    lac_per_vol = []
    lac_density_vol_blocks = []

    # Load Volume Info
    img_nii = nib.load(path_original_volume)
    seg_nii = nib.load(path_seg)
    img_proxy = img_nii.dataobj
    seg_proxy = seg_nii.dataobj
    img_shape = img_nii.shape
    seg_shape = seg_nii.shape

    if img_shape != seg_shape:
        raise ValueError(
            "Original volume and segmentation mask must have matching shapes. "
            f"Original {path_original_volume} has shape {img_shape}; "
            f"segmentation {path_seg} has shape {seg_shape}."
        )
    
    max_z = seg_shape[2]
    
    # 1. Handle Start
    if slice_start < 0: slice_start = 0
    if slice_start >= max_z:
        raise ValueError(f"slice_start ({slice_start}) is beyond volume depth ({max_z}).")

    # 2. Handle End
    if slice_end is None or slice_end > max_z:
        slice_end = max_z
    
    if slice_end <= slice_start:
        raise ValueError(f"slice_end ({slice_end}) must be greater than slice_start ({slice_start}).")

    print(f"[ CONFIG ] Analysis Range: Slice {slice_start} to {slice_end}")
    print(f"[ CONFIG ] Block Size: {nr_slice_per_block}, Overlap: {overlap_percentage}%")
    
    # Generate blocks based on specific range
    z_blocks = calculate_overlap_blocks(slice_start, slice_end, nr_slice_per_block, overlap_percentage)

    print(f"Volume Shape: {seg_shape}")
    print(f"Total Blocks to process: {len(z_blocks)}")

    block_nr = 1
    header_written = False


    for z_start, z_end in z_blocks:
        print(f"--- Processing Block {block_nr}: slices {z_start}-{z_end} ---")

        # Load Data
        # float32 cuts img memory by half vs float64
        img_block = load_nifti_block_proxy(img_proxy, z_start, z_end, dtype=np.uint8)

        # seg only needs uint8 / bool
        seg_block = load_nifti_block_proxy(seg_proxy, z_start, z_end, dtype=np.uint8)
        scan_shape = seg_block.shape

        # --- GPU Processing ---
        if use_gpu_flag:
            try:
                seg_gpu = cp.asarray(seg_block)
                img_gpu = cp.asarray(img_block)
                
                # Thresholding
                val = filters.threshold_otsu(img_block)
                #tissue_gpu = (img_gpu > val).astype(cp.uint8) * 255
                
                # Labeling
                s_gpu = ndi_gpu.generate_binary_structure(3, 2)
                labeled_gpu, _ = ndi_gpu.label(seg_gpu.astype(cp.uint8), structure=s_gpu)
                
                # Retrieve
                labeled_array = cp.asnumpy(labeled_gpu)
                #tissue_block = cp.asnumpy(tissue_gpu)

                # Clean VRAM
                del seg_gpu, img_gpu, labeled_gpu
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                print(f"[ WARN ] GPU processing failed for Block {block_nr}. Reason: {e}")
                print("[ INFO ] Falling back to CPU for this and future blocks.")
                use_gpu_flag = False 

        # --- CPU Processing (Fallback) ---
        if not use_gpu_flag:
            val = filters.threshold_otsu(img_block)
            s = scipy.ndimage.generate_binary_structure(3, 2)
            labeled_array, _ = scipy.ndimage.label(seg_block, structure=s)  # seg_block already uint8
            labeled_array = labeled_array.astype(np.int32, copy=False)

        # --- Morphology & Analysis (CPU) ---
        vol_clear = clear_border_cpu(labeled_array, buffer_size=edgewidth, bgval=0)
        declust = np.logical_xor(
            skimage.morphology.remove_small_objects(vol_clear, max_size=minfilter - 1, connectivity=10),
            skimage.morphology.remove_small_objects(vol_clear, max_size=maxfilter - 1, connectivity=10),
        )
        
        s = scipy.ndimage.generate_binary_structure(3, 2)
        labeled_array, num_features = scipy.ndimage.label(declust.astype(int), structure=s)
        labeled_array = labeled_array.astype(np.int32, copy=False)
        
        del vol_clear, declust
        gc.collect()

        print(f"... # Found Lacunae: {num_features}")

       

        # Extract features (legacy-equivalent)
        data_collector = {k: [] for k in ['Area', 'Centroid', 'Volume', 'SurfaceArea', 'Len',
                                  'Distance_Max', 'Distance_Min', 'Lc_St', 'Lc_Ob',
                                  'Lc_Or_1_x', 'Lc_Or_1_y', 'Lc_Or_1_z',
                                  'Lc_Or_2_x', 'Lc_Or_2_y', 'Lc_Or_2_z',
                                  'BBox']}

        # 2. Run regionprops once (The Speedup)
        # Labeled array is already int32 from previous steps
        props = regionprops(labeled_array)
        

        for prop in props:
            # --- CRITICAL STEP 1: PADDING ---
            # Mimics your original 'boxsize=1'. 
            # We add a 1-pixel border of zeros around the tight binary mask.
            padded_vol = np.pad(prop.image, pad_width=1, mode='constant', constant_values=0)

            # 3. Marching Cubes
            try:
                vol_verts, vol_faces, _, _ = measure.marching_cubes(padded_vol, method="lewiner")
            except (ValueError, RuntimeError):
                # Strict legacy behavior: if MC fails, skip this lacuna
                continue

            # 4. Standard features
            est_surface = skimage.measure.mesh_surface_area(vol_verts, vol_faces)
            est_volume = mesh_volume(vol_faces, vol_verts)
            
            data_collector['SurfaceArea'].append(est_surface)
            data_collector['Volume'].append(est_volume)
            data_collector['Area'].append(prop.area)

            bb = prop.bbox
            d_x_um = (bb[3] - bb[0]) * voxelsize_um
            d_y_um = (bb[4] - bb[1]) * voxelsize_um
            d_z_um = (bb[5] - bb[2]) * voxelsize_um
            data_collector['BBox'].append((d_x_um, d_y_um, d_z_um))
            

            # --- CRITICAL STEP 2: GLOBAL CENTROID ---
            # prop.centroid is (z,y,x) relative to the current block (labeled_array)
            # We just need to add the block's Z-start offset.
            c_block = np.array(prop.centroid)
            c_global = np.array([
                c_block[0],             # X (in your code orientation)
                c_block[1],             # Y
                c_block[2] + z_start    # Z (Global)
            ])
            data_collector['Centroid'].append(c_global)

            # --- CRITICAL STEP 3: LEGACY LEN ---
            # vol_verts are coordinates inside 'padded_vol'.
            # prop.centroid_local is the center inside 'prop.image'.
            # Since 'padded_vol' is shifted by +1 pixel, we shift the centroid by +1 to match.
            c_local_padded = np.array(prop.centroid_local) + 1
            
            len_temp = legacy_len_from_vertices(vol_verts, c_local_padded)
            data_collector['Len'].append(len_temp)

            # 6. PCA (Orientation & Shape)
            # Matches legacy: compute_initial_guess used to run on (vol > 0). 
            # prop.image is exactly that boolean mask.
            try:
                # Need uint8 for the PCA function
                vol_bin_uint8 = prop.image.astype(np.uint8)
                evals, evecs = compute_initial_guess.get_principal_axes_grayscale(vol_bin_uint8, 0.5)

                # Sort eigenvalues and eigenvectors together (descending order)
                idx = evals.argsort()[::-1]   
                evals = evals[idx]
                evecs = evecs[:, idx]

                # Exact legacy formulas
                if evals[0] > 0:
                    res_St = (np.sqrt(evals[0]) - np.sqrt(evals[2])) / np.sqrt(evals[0])
                else:
                    res_St = 0.0
                
                denom = (np.sqrt(evals[0]) - np.sqrt(evals[2]))
                if denom != 0:
                    res_Ob = (2 * ((np.sqrt(evals[1]) - np.sqrt(evals[2])) / denom) - 1)
                else:
                    res_Ob = 0.0

                data_collector['Lc_St'].append(res_St)
                data_collector['Lc_Ob'].append(res_Ob)
                
                # Orientation vectors
                data_collector['Lc_Or_1_x'].append(evecs[0, 0])
                data_collector['Lc_Or_1_y'].append(evecs[1, 0])
                data_collector['Lc_Or_1_z'].append(evecs[2, 0])
                data_collector['Lc_Or_2_x'].append(evecs[0, 1])
                data_collector['Lc_Or_2_y'].append(evecs[1, 1])
                data_collector['Lc_Or_2_z'].append(evecs[2, 1])

                data_collector['Distance_Max'].append(evals[0] / 2.0)
                data_collector['Distance_Min'].append(evals[2])

            except Exception:
                # Zeros fallback
                data_collector['Lc_St'].append(0.0)
                data_collector['Lc_Ob'].append(0.0)
                for k in ['Lc_Or_1_x','Lc_Or_1_y','Lc_Or_1_z','Lc_Or_2_x','Lc_Or_2_y','Lc_Or_2_z']:
                    data_collector[k].append(0.0)
                data_collector['Distance_Max'].append(0.0)
                data_collector['Distance_Min'].append(0.0)

        Area_arr = np.array(data_collector['Area'], dtype=float)
        lac_vol_micron = Area_arr  # legacy uses voxel count (region.area) as "volume in voxels"
        total_vol_lac = float(np.sum(lac_vol_micron))

        
        background_pixels = np.count_nonzero(img_block <= val)

        bg_vol_um = background_pixels * ((voxelsize.magnitude * 1000) ** 3)

        total_pixels = seg_block.shape[0] * seg_block.shape[1] * seg_block.shape[2]
        total_vol_bone = (total_pixels * ((voxelsize.magnitude * 1000) ** 3)) - bg_vol_um - total_vol_lac

        lacunar_volume_blocks.append(total_vol_lac)
        background_pixels_blocks.append(bg_vol_um)
        total_vol_bone_blocks.append(total_vol_bone)
        lac_density_vol_blocks.append(total_vol_lac / total_vol_bone if total_vol_bone > 0 else 0.0)
        lac_per_vol.append(len(Area_arr) / total_vol_bone if total_vol_bone > 0 else 0.0)


        if len(data_collector['Centroid']) == 0:
            print("... No valid lacunae parameters extracted in this block. Volume stats recorded, skipping lacuna CSV.")
            del labeled_array, seg_block, img_block
            gc.collect()
            if use_gpu_flag:
                cp.get_default_memory_pool().free_all_blocks()
            print(f"... Block {block_nr} RAM cleared.\n")
            block_nr += 1
            continue
        

        # Create DataFrame
        Distance_Max = np.array(data_collector['Distance_Max']) * (voxelsize.magnitude * 1000)
        Distance_Min = np.array(data_collector['Distance_Min']) * (voxelsize.magnitude * 1000)
        lac_vol_micron = np.array(data_collector['Area'])
        lac_Vol_cubic_um = lac_vol_micron * ((voxelsize.magnitude * 1000) ** 3)
        
        SurfaceArea = np.array(data_collector['SurfaceArea'])
        Volume_raw = np.array(data_collector['Volume'])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            SV_ratio = np.where(Volume_raw > 0, SurfaceArea / Volume_raw, 0)
            SA_um = SurfaceArea * ((voxelsize.magnitude * 1000) ** 2)
            LcSAV_ratio = np.where(lac_Vol_cubic_um > 0, SA_um / lac_Vol_cubic_um, 0)
            MM_Ratio = np.where(Distance_Max > 0, Distance_Min / Distance_Max, 0)

        
        if len(data_collector['Centroid']) > 0:
            Centroid = np.stack(data_collector['Centroid'])
        else:
            Centroid = np.array([]) # Should be caught by previous if, but safety check
        Centroid += 1 

        bbox_arr = np.array(data_collector.get('BBox', []), dtype=float)
        if bbox_arr.size == 0:
            bbox_arr = np.zeros((len(Centroid), 3), dtype=float)


        dict_par = {
            
            "Lacunar ID": np.arange(1, len(Centroid) + 1),

            # --- RENAMED metadata (these are scan/block dimensions, not lacuna dimensions) ---
            "Scan Width (px)":  seg_block.shape[0],  # was Dimension X
            "Scan Height (px)": seg_block.shape[1],  # was Dimension Y
            "Block Depth (px)": seg_block.shape[2],  # was Dimension Z

            # --- NEW real lacuna dimensions ---
            "Lacuna BBox Size X (um)": bbox_arr[:, 0],
            "Lacuna BBox Size Y (um)": bbox_arr[:, 1],
            "Lacuna BBox Size Z (um)": bbox_arr[:, 2],

            index_centroid_x: Centroid[:, 0],
            index_centroid_y: Centroid[:, 1],
            index_centroid_z: Centroid[:, 2],
            index_vol_um: lac_Vol_cubic_um,
            index_stretch_lacunae: data_collector['Lc_St'],
            index_lacunae_oblateness: data_collector['Lc_Ob'],
            index_Lc_Or_1_x: data_collector['Lc_Or_1_x'],
            index_Lc_Or_1_y: data_collector['Lc_Or_1_y'],
            index_Lc_Or_1_z: data_collector['Lc_Or_1_z'],
            index_Lc_Or_2_x: data_collector['Lc_Or_2_x'],
            index_Lc_Or_2_y: data_collector['Lc_Or_2_y'],
            index_Lc_Or_2_z: data_collector['Lc_Or_2_z'],
            index_sa_vol: SV_ratio,
            index_LcSAV_ratio: LcSAV_ratio,
            index_sa_um: SA_um,
            index_maj: Distance_Max,
            index_mna: Distance_Min,
            index_ax_rat: MM_Ratio,
            "Voxel Size (mm)": voxelsize.magnitude,
            index_tot_lac_num: len(lac_Vol_cubic_um),
            
        }

        block_df = pd.DataFrame(dict_par)
        
        # Save Incrementally
        write_header = not header_written
        block_df.to_csv(temp_csv_path, mode='a', header=write_header, index=False)
        header_written = True
        
        
        # 2. NEW: Save Individual Block File (if requested)
        if save_block_files:
            # Padded zeros ensure lexicographical sorting (e.g., block001 comes before block010)
            fmt_block = f"{block_nr:03d}"
            fmt_start = f"{z_start:04d}"
            fmt_end = f"{z_end:04d}"
            
            # Name: {Sample}_block{N}_lacunae_subset_z{Start}-{End}.csv
            block_filename = f"{sample_name}_block{fmt_block}_lacunae_subset_z{fmt_start}-{fmt_end}.csv"
            block_file_path = os.path.join(sample_out_dir, block_filename)
            
            block_df.to_csv(block_file_path, index=False)
            print(f"... [SAVE] Block {block_nr} data saved: {block_filename}")

        print(f"... Block {block_nr} saved to temporary CSV.")


        # Cleanup
        del block_df, dict_par, data_collector, Centroid, labeled_array, seg_block, img_block
        if 'labeled_gpu' in locals(): del labeled_gpu
        gc.collect()
        if use_gpu_flag: cp.get_default_memory_pool().free_all_blocks()

        print(f"... Block {block_nr} RAM cleared.\n")
        block_nr += 1

    # End Processing
    print("[ INFO ] Processing complete. Starting Aggregation and Deduplication...")

    n_blocks = len(lacunar_volume_blocks)
    range_suffix = f"z{slice_start}-{slice_end}"
    
    
    dict_par_block_vol = {
        "Block ID": np.arange(1, n_blocks + 1),
        "Z_Start": [b[0] for b in z_blocks[:n_blocks]], # Add explicit Z coords for the LLM
        "Z_End": [b[1] for b in z_blocks[:n_blocks]],
        "Total Lacunar Volume [um3]": lacunar_volume_blocks,
        "Total Background Volume [um3]": background_pixels_blocks,
        "Total Tissue Volume [um3]": total_vol_bone_blocks,
        "Number of Lacunae/Bone Tissue": lac_per_vol,
        "Volume Fraction of Lacunae in Bone Tissue": lac_density_vol_blocks,
    }
    block_volume_stats_df = pd.DataFrame(dict_par_block_vol)
    
    # Name: {Sample}_bone_density_distribution_per_block_{Range}.csv
    block_dist_filename = f"{sample_name}_bone_density_distribution_per_block_{range_suffix}.csv"
    block_volume_analysis_path = os.path.join(sample_out_dir, block_dist_filename)
    block_volume_stats_df.to_csv(block_volume_analysis_path, index=False)


    # ---------------------------------------------------------
    # 2. Individual Morphometry (Raw Data) & 3. Global Summary
    # ---------------------------------------------------------
    aggregated_lacunae_path = None
    summary_features_path = None
    
    if os.path.exists(temp_csv_path):
        full_df = pd.read_csv(temp_csv_path)
        print(f"Total lacunae (raw): {len(full_df)}")
        
        unique_df = remove_duplicate_lacunae(full_df)
        print(f"Total unique lacunae: {len(unique_df)}")
        
        # Name: {Sample}_lacunae_individual_morphometry_{Range}.csv
        indiv_filename = f"{sample_name}_lacunae_individual_morphometry_{range_suffix}.csv"
        aggregated_lacunae_path = os.path.join(sample_out_dir, indiv_filename)
        unique_df.to_csv(aggregated_lacunae_path, index=False)
        
        if len(unique_df) > 0 and np.sum(total_vol_bone_blocks) > 0:
            global_lac_density = np.sum(lacunar_volume_blocks) / np.sum(total_vol_bone_blocks)
            features_df = calculate_lacunae_features(unique_df, global_lac_density)
            
            # Name: {Sample}_global_morphometry_summary_{Range}.csv
            summary_filename = f"{sample_name}_global_morphometry_summary_{range_suffix}.csv"
            summary_features_path = os.path.join(sample_out_dir, summary_filename)
            features_df.to_csv(summary_features_path, index=False)
        
        os.remove(temp_csv_path)

    # Return clear keys for the LLM Tool
    return {
        "individual_lacunae_csv": aggregated_lacunae_path,
        "density_distribution_csv": block_volume_analysis_path,
        "global_summary_csv": summary_features_path,
    }

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description='Extract 3D lacunar parameters')
        parser.add_argument('--slices', '-s', type=int, default=100)
        parser.add_argument('--sample', '-n', type=str, default="T1_S7_step0_Z0")
        parser.add_argument('--images', '-i', type=str, required=False)
        parser.add_argument('--segmentations', '-g', type=str, required=False)
        parser.add_argument('--processed', '-p', action='store_true')
        return parser.parse_args()

    args = parse_args()

    # Placeholder defaults used when --images/--segmentations are not supplied.
    # Point these at your own volume and matching segmentation mask.
    sample = args.sample
    path_img = '/path/to/data/volumes/<sample>.nii.gz'
    path_seg = '/path/to/data/segmentations/<sample>_segmented.nii.gz'

    if args.images: path_img = args.images
    if args.segmentations: path_seg = args.segmentations

    main(
        nr_slice_per_block=args.slices,
        sample_name=sample,
        path_original_volume=path_img,
        path_seg=path_seg,
        processed=args.processed
    )
