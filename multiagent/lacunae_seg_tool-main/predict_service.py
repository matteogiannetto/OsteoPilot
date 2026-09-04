"""
Parallelized SR-microCT lacunae segmentation for NIfTI volumes.
Processes slices in batches to minimize memory usage.
"""

import os
import argparse
import pyfiglet
import numpy as np
import nibabel as nib
import tempfile
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from skimage import transform, filters

import torch
from torchvision import transforms

from utils import *
from model.unet import *
from patching import *
from preprocessing import *

SEG_TOOL_ROOT = Path(__file__).resolve().parent
if str(SEG_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(SEG_TOOL_ROOT))


def select_torch_device(torch_module=torch, gpu_id="0"):
    """Select CUDA first, then Apple Metal/MPS, then CPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda:0")
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_built()
        and mps_backend.is_available()
    ):
        return torch_module.device("mps")
    return torch_module.device("cpu")


def seed_torch_for_inference(torch_module=torch, seed=19):
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
    mps_module = getattr(torch_module, "mps", None)
    if mps_module is not None and hasattr(mps_module, "manual_seed"):
        mps_module.manual_seed(seed)


def legacy_patch_grid_shape(image_shape, patch_size):
    """Return the full-patch grid shape used by PatchExtractor."""
    height, width = int(image_shape[0]), int(image_shape[1])
    cols = int((width - patch_size) / patch_size + 1)
    rows = int((height - patch_size) / patch_size + 1)
    return cols, rows


def iter_legacy_patch_origins(image_shape, patch_size):
    """Yield row-major paste origins matching PatchExtractor.extract_img_patches()."""
    cols, rows = legacy_patch_grid_shape(image_shape, patch_size)
    for row_idx in range(rows):
        row = row_idx * patch_size
        for col_idx in range(cols):
            col = col_idx * patch_size
            yield row, col


def segment_slice_array(slice_data, model, device, configs):
    """Segment one 2D slice using the original patch extraction/reassembly flow."""
    if slice_data.dtype != np.uint8:
        slice_data = ((slice_data - slice_data.min()) /
                      (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)

    nr_patches = configs.test_params["nr_patches"]
    patch_size = max(int(slice_data.shape[0] / nr_patches),
                     int(slice_data.shape[1] / nr_patches))

    extractor = PatchExtractor(
        img=Image.fromarray(slice_data),
        patch_size=patch_size,
        stride=patch_size,
    )
    patches_img = extractor.extract_img_patches()

    # Tissue segmentation for patch selection
    val = filters.threshold_otsu(slice_data)
    seg_tissue = np.uint8((slice_data > val) * 255)
    extractor_tissue = PatchExtractor(
        img=Image.fromarray(seg_tissue),
        patch_size=patch_size,
        stride=patch_size,
    )
    patches_tiss = extractor_tissue.extract_img_patches()

    seg_slice = np.zeros(slice_data.shape, dtype=np.uint8)
    img_dim_row = configs.test_params["img_dim_row"]
    img_dim_col = configs.test_params["img_dim_col"]
    probab_th = configs.test_params["probab_th"]

    patch_origins = list(iter_legacy_patch_origins(slice_data.shape, patch_size))
    if len(patch_origins) != len(patches_img):
        raise RuntimeError(
            f"Patch grid mismatch: {len(patches_img)} extracted patches but "
            f"{len(patch_origins)} paste origins for shape {slice_data.shape}."
        )

    for patch_idx, (r, c) in enumerate(patch_origins):
        tissue_patch = np.asarray(patches_tiss[patch_idx])
        tissue_ratio = np.count_nonzero(tissue_patch) / tissue_patch.size

        if tissue_ratio > 0:
            img_patch_resized = transform.resize(
                np.asarray(patches_img[patch_idx]),
                (img_dim_row, img_dim_col),
            )
            patch_tensor = transforms.ToTensor()(np.float32(img_patch_resized))
            patch_tensor = torch.stack([patch_tensor, patch_tensor, patch_tensor], 1).to(device)

            if torch.count_nonzero(patch_tensor) != 0:
                with torch.no_grad():
                    masks_pred_prob = model(patch_tensor)
                masks_pred = (masks_pred_prob > probab_th).float()
                patch_pred = np.asarray(masks_pred[0, 0, :, :].cpu()) * 255
                patch_pred = patch_pred.astype(np.uint8)
            else:
                patch_pred = np.zeros((img_dim_row, img_dim_col), dtype=np.uint8)
        else:
            patch_pred = np.zeros((img_dim_row, img_dim_col), dtype=np.uint8)

        if r < slice_data.shape[0] and c < slice_data.shape[1]:
            patch_resized = transform.resize(
                patch_pred,
                (patch_size, patch_size),
                order=0,
            ).astype(np.uint8)
            end_r = min(r + patch_size, slice_data.shape[0])
            end_c = min(c + patch_size, slice_data.shape[1])
            seg_slice[r:end_r, c:end_c] = patch_resized[:end_r-r, :end_c-c]

    seg_slice[seg_slice > 0] = 1
    return seg_slice


def process_slice_batch(slice_batch, slice_indices, model, device, configs, temp_dir, pbar=None):
    """Process a batch of slices and save results to temporary files"""
    results = []
    
    with torch.no_grad():
        for i, (slice_data, slice_idx) in enumerate(zip(slice_batch, slice_indices)):
            seg_slice = segment_slice_array(slice_data, model, device, configs)
            
            # Save temporary result
            temp_file = os.path.join(temp_dir, f"slice_{slice_idx:06d}.npy")
            np.save(temp_file, seg_slice)
            results.append((slice_idx, temp_file))
            
            # Update progress bar if provided
            if pbar:
                pbar.update(1)
    
    return results

def load_slice_batch(nii_img, start_idx, batch_size):
    """Load a batch of slices from NIfTI file without loading entire volume"""
    total_slices = nii_img.shape[2]
    end_idx = min(start_idx + batch_size, total_slices)
    
    # Load only the required slices
    slice_data = nii_img.dataobj[:, :, start_idx:end_idx]
    return np.asarray(slice_data), list(range(start_idx, end_idx))

def assemble_volume_from_temp(temp_dir, output_path, original_nii, total_slices):
    """Assemble final volume from temporary slice files"""
    print("Assembling final volume...")
    
    # Get template slice to determine output shape
    temp_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.npy')])
    first_slice = np.load(os.path.join(temp_dir, temp_files[0]))
    
    output_shape = (*first_slice.shape, total_slices)

    # Keep the full output volume off RAM. The previous implementation allocated
    # output_data for the entire 3D mask, which can exceed tens of GB.
    memmap_path = os.path.join(temp_dir, "segmented_volume_uint8.dat")
    output_data = np.memmap(memmap_path, mode="w+", dtype=np.uint8, shape=output_shape)

    for slice_idx in tqdm(range(total_slices), desc="Assembling volume"):
        temp_file = os.path.join(temp_dir, f"slice_{slice_idx:06d}.npy")
        if os.path.exists(temp_file):
            output_data[:, :, slice_idx] = np.load(temp_file, mmap_mode="r")
        else:
            output_data[:, :, slice_idx] = 0

    output_data.flush()

    header = original_nii.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_data_shape(output_shape)
    header.set_slope_inter(1, 0)
    output_nii = nib.Nifti1Image(output_data, original_nii.affine, header)
    output_nii.set_data_dtype(np.uint8)
    nib.save(output_nii, output_path)
    print(f"Segmented volume saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='SR-microCT NIfTI lacunae segmentation',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-nifti', required=True,
                       help='path to input NIfTI file (.nii.gz)')
    parser.add_argument('--output-dir', required=True,
                       help='output directory for segmented volume')
    parser.add_argument('--model-path', default='/path/to/lacunae_seg_tool-main/model',
                       help='path to CNN model directory')
    parser.add_argument('--model-configs', type=str, default='config_predict.py',
                       help='model configuration file')
    parser.add_argument('--gpu-id', default='0',
                       help='GPU ID to use')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='number of slices to process in parallel')
    parser.add_argument('--max-workers', type=int, default=30,
                       help='maximum number of worker threads')
    
    args = parser.parse_args()
    
    print('\n', pyfiglet.figlet_format('LACUNAE MASKING', font='pebbles'))
    print('Starting NIfTI segmentation ...\n')
    
    # Setup
    configs = load_config(args.model_configs)
    device = select_torch_device(torch, args.gpu_id)
    print(f"Using PyTorch device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Seeding
    SEED = 19
    seed_torch_for_inference(torch, SEED)
    np.random.seed(SEED)
    
    # Load model
    model = UNet(input_channels=configs.model_params["input_ch"], 
                nclasses=configs.model_params["nr_classes"])
    model.load_state_dict(
        torch.load(
            os.path.join(args.model_path, configs.model_params["model_name"] + '.h5'),
            map_location="cpu",
        )
    )
    model = model.to(device)
    model.eval()
    
    # Load NIfTI file header only
    print(f"Loading NIfTI file: {args.input_nifti}")
    nii_img = nib.load(args.input_nifti)
    total_slices = nii_img.shape[2]
    print(f"Volume shape: {nii_img.shape}")
    print(f"Total slices to process: {total_slices}")
    print(f"Data type: {nii_img.dataobj.dtype}")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="lacunae_seg_")
    
    try:
        # Process slices in batches
        batch_size = args.batch_size
        
        with tqdm(total=total_slices, desc="Processing slices") as pbar:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = set()
                
                # Submit batch jobs
                for start_idx in range(0, total_slices, batch_size):
                    slice_batch, slice_indices = load_slice_batch(nii_img, start_idx, batch_size)
                    
                    # Convert to list of individual slices
                    individual_slices = [slice_batch[:, :, i] for i in range(slice_batch.shape[2])]
                    
                    future = executor.submit(process_slice_batch, individual_slices,
                                           slice_indices, model, device, configs, temp_dir, pbar)
                    futures.add(future)

                    if len(futures) >= args.max_workers:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                
                # Wait for all futures to complete
                for future in futures:
                    future.result()
        
        # Assemble final volume
        output_filename = Path(args.input_nifti).stem.replace('.nii', '_segmented.nii.gz')
        output_path = os.path.join(args.output_dir, output_filename)
        
        assemble_volume_from_temp(temp_dir, output_path, nii_img, total_slices)
        
    finally:
        # Cleanup temporary files
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
    print("Segmentation completed successfully!")

if __name__ == '__main__':
    main()
