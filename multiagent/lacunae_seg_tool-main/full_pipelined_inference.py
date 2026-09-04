import torch
import torch.nn as nn
from torchvision import models
from model.segmentation_UNet import U_Net
import argparse
import cv2
import numpy as np
import torch.multiprocessing as mp
from functools import partial


def select_torch_device(torch_module=torch):
    """Select CUDA first, then Apple Metal/MPS, then CPU."""
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_built()
        and mps_backend.is_available()
    ):
        return torch_module.device("mps")
    return torch_module.device("cpu")


def extract_patch(params):
    """Extract a single patch from an image.
    
    Args:
        params: Tuple containing (coord, image, size)
            coord: Tuple (start_i, start_j) - starting coordinates
            image: The full image array
            size: Size of patch to extract
    """
    coord, image, size = params
    start_i, start_j = coord
    patch = image[start_i:start_i+size, start_j:start_j+size]
    if len(patch.shape) == 3:
        patch = np.squeeze(patch, axis=2)
    return {'patch': patch.astype(np.float32), 'coord': (start_i, start_j)}

# Fix the parallel_patch_extraction function to properly pass parameters to extract_patch
def parallel_patch_extraction(image, size=128, overlap=30, num_workers=None):
    """Split patch extraction across multiple CPU cores"""
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    
    # Calculate grid layout
    if overlap >= 70:
        raise Warning("Patches are overlapping by more than 70%.")
    overlap_px = int(size * (overlap / 100))
    
    # Create grid coordinates for parallel processing
    coords = []
    for i in range(0, image.shape[0], size - overlap_px):
        for j in range(0, image.shape[1], size - overlap_px):
            start_i = min(i, image.shape[0] - size)
            start_j = min(j, image.shape[1] - size)
            coords.append((start_i, start_j))
    
    # Create parameter tuples for each patch extraction
    params = [(coord, image, size) for coord in coords]

    # Use process pool to extract patches in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(extract_patch, params)  # Pass the full parameter tuples
    
    # Organize results
    patches = [r['patch'] for r in results]
    patch_coords = [r['coord'] for r in results]
    
    return {'img': patches, 'gt': [], 'coord': patch_coords}

def process_patches_batch(batch_data):
    """Process a batch of patches in a worker process"""
    patches_batch, coords_batch, patch_size, img_shape = batch_data
    
    # Create local arrays for this batch
    local_reconstructed = np.zeros(img_shape, dtype=np.float64)
    local_count_map = np.zeros(img_shape, dtype=np.uint8)
    
    # Process each patch in the batch
    for patch, (x, y) in zip(patches_batch, coords_batch):
        local_reconstructed[x:x + patch_size[0], y:y + patch_size[1]] += patch[0]
        local_count_map[x:x + patch_size[0], y:y + patch_size[1]] += 1
        
    return local_reconstructed, local_count_map

def reconstruct_image_parallel(patches, coords, img_shape, num_workers=None):
    """
    Reconstructs a full image from its patches using the stored coordinates.
    Uses multiprocessing to speed up the reconstruction process.
    """
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    
    if not patches:
        # Handle empty patches case
        reconstructed = np.zeros(img_shape, dtype=np.uint8)
        return reconstructed
    
    patch_size = patches[0][0].shape
    
    # Split patches and coordinates into batches for workers
    total_patches = len(patches)
    batch_size = max(1, total_patches // num_workers)
    batches = []
    
    for i in range(0, total_patches, batch_size):
        end_idx = min(i + batch_size, total_patches)
        patches_batch = patches[i:end_idx]
        coords_batch = coords[i:end_idx]
        batches.append((patches_batch, coords_batch, patch_size, img_shape))
    
    # Process batches in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_patches_batch, batches)
    
    # Combine results from all workers
    reconstructed = np.zeros(img_shape, dtype=np.float64)
    count_map = np.zeros(img_shape, dtype=np.uint8)
    
    for local_reconstructed, local_count_map in results:
        reconstructed += local_reconstructed
        count_map += local_count_map
    
    # Handle zero counts to avoid division by zero
    count_map[count_map == 0] = 1
    
    # Normalize overlapping regions
    reconstructed = reconstructed // count_map
    
    # Scale the image appropriately
    if np.max(reconstructed) > 1:
        reconstructed = np.clip(reconstructed, 0, 255)
    else:  # If values are between 0-1, scale up (for GT visualization)
        reconstructed = reconstructed * 255
        
    return reconstructed.astype(np.uint8)


def build_convnext():
    """
    Builds a ConvNeXt_Base model modified to:
      - Accept a single-channel (e.g. 128x128) input,
      - Output 2 logits (wrapped in softmax for probability outputs).
    """
    # Load pretrained ConvNeXt_Base model
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    # Modify the first convolution to accept 1-channel input
    # In newer PyTorch versions, the first conv is in model.features[0]
    original_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding
    )
    # Average the weights over the 3 RGB channels
    new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
    new_conv.bias.data = original_conv.bias.data
    model.features[0][0] = new_conv

    # Modify the classifier head to output 2 logits.
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, 2)
    
    # Wrap the model with a softmax to output probabilities.
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    
    return model



def filter_dataset_batched(patches_dic,checkpoint_path, batch_size=128,treshold=0.15):
    """
    Process dataset patches in batches for more efficient inference.
    
    Args:
        model: The trained model for inference
        dataset: Dictionary containing partitions with image patches
        device: Computation device (CPU/GPU)
        batch_size: Number of patches to process in a single batch
        
    Returns:
        Filtered dataset with only patches that have class 1 probability >= treshold
    """
    
    # Set device
    device = select_torch_device(torch)
    # Build model and load pretrained weights
    model = build_convnext()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)

    model.eval()  # Set model to evaluation mode
    
    total_patches_in_image = len(patches_dic['img'])
    indices_to_keep = []
    
    

    with torch.no_grad():
        img_patches = patches_dic['img']
        coords = patches_dic['coord']
        
        
        # Process in batches
        for batch_start in range(0, total_patches_in_image, batch_size):
            batch_end = min(batch_start + batch_size, total_patches_in_image)
            batch_indices = list(range(batch_start, batch_end))
            
            # Create batch tensors
            batch_patches = [img_patches[i] for i in batch_indices]
            
            # Convert patches to tensors
            batch_tensors = []
            for patch in batch_patches:
                tensor = torch.tensor(patch, dtype=torch.float32)
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)  # Add channel dimension
                batch_tensors.append(tensor)

            
            # Stack tensors into a batch
            batch_input = torch.stack(batch_tensors).to(device)
            
            # Perform batch inference
            batch_output = model(batch_input)
            
            # Get class 1 probabilities
            batch_probs = batch_output[:, 1].cpu().numpy()
            
            # Determine which patches to keep
            for idx, (i, prob) in enumerate(zip(batch_indices, batch_probs)):
                if prob >= treshold:
                    indices_to_keep.append(i)
                    
    
        # Filter the image details to keep only selected patches
        filtered_img = [img_patches[i] for i in indices_to_keep]
        filtered_coord = [coords[i] for i in indices_to_keep]
        
    return {'img': filtered_img, 'gt':[], 'coord': filtered_coord}

def segmetation(patches_dic,checkpoint_path):
    '''
    takes as input the dataset, load the preselectior weights and calculate filter the dataset using ai
    '''
    # Set device
    device = select_torch_device(torch)
    # Build model and load pretrained weights
    model = U_Net()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    # Function to process a patch: convert to tensor and add channel dimension as expected by the model for training
    def process_patch(patch):
        tensor = torch.tensor(patch, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # shape: [1, 128, 128] theorically
        return tensor
    
    
    with torch.no_grad():
        # Iterate over both dataset partitions (e.g., "train" and "val")

        for i in range(len(patches_dic['img'])):
            
            patch = patches_dic['img'][i]
            inp = process_patch(patch).unsqueeze(0).to(device)  # shape: [1, 1, 128, 128]
            assert inp.shape == (1, 1, 128, 128), f"Expected shape (1, 1, 128, 128), got {inp.shape}"
            output = model(inp)  # output shape: [1, 2, 128, 128]
            output = torch.argmax(output, dim=1).cpu().numpy()  # shape: [1, 128, 128]
            
            patches_dic['gt'].append(output)
           

                       
    return patches_dic


def full_pipelined_cricca_inference(img):
    '''
    takes as input the image and returns the segmented image
    the image cannot be unposecced, must have been preprocessed
    '''

    assert img is not None, f"Image not found"
    original_shape = img.shape
    
    # Extract patches
    patches = parallel_patch_extraction(img)
    
    preselector_weights_path = './model/best_preselector_weights.pth'
    
    # Preselect patches
    preselected_patches = filter_dataset_batched(patches,preselector_weights_path)
    
    segmeter_weights_path = './model/segmenter_weights.pth' 
    # Segment patches
    segmented_patches = segmetation(preselected_patches,segmeter_weights_path)
    
    # Reconstruct the segmented image
    segmented_img = reconstruct_image_parallel(segmented_patches['gt'], segmented_patches['coord'], original_shape)
    
    return segmented_img
    
    
if __name__ =="__main__":
    # Display a warning about the experimental nature of the pipeline
    print("=" * 80)
    print("WARNING: this pipeline is not meant to be executed directly.")
    print("        an assertion will be trigghered to stop the execution.")
    print("        ")
    print("=" * 80)
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run bone segmentation on a microCT image')
    parser.add_argument('--img_path', type=str, help='Path to the .tiff image file') # require prerpoceesed image
    args = parser.parse_args()

    # Validate image path and extension
    img_path = args.img_path
    assert img_path.endswith(('.tif', '.tiff')), "Input file must be a TIFF image (.tif or .tiff)"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    segmentation = full_pipelined_cricca_inference(img)
    assert False, "no output specified"
    print("Segmentation completed successfully.")
    
