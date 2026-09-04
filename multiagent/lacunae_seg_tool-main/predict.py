"""
Main script to run inference on a single e complete SR-microCT image volume.
Run --- python3 predict.py
"""

import os
import re
import random
import argparse
import pyfiglet
import numpy as np

from glob import glob
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from skimage import transform

import torch
from torchvision import transforms

from utils import *
from model.unet import *
from patching import *
from preprocessing import *
from predict_service import seed_torch_for_inference, select_torch_device

c_wd = os.getcwd()

# parse input arguments
parser = argparse.ArgumentParser(description='SR-microCT WSI lacunae segmentation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--imgs-dir-path', default='./data',
                    help='path name of the dir of the images to process')
parser.add_argument('--model-path', default='./model',
                    help='path name of the dir of the CNN model to use')
parser.add_argument("--model_configs", type=str, default='config_predict.py',
                    help="filename of the model configuration file.")
parser.add_argument('--gpu-id', default = '0',
                    help='id of the gpu to use for training')

args = parser.parse_args()

# Let's *art :)
print ('\n', pyfiglet.figlet_format('LACUNAE MASKING', font='pebbles'))
print ('Starting segmentation ... \n')

# Load parameters
configs = load_config(args.model_configs)

# Define the device to run prediction: CUDA, Apple Metal/MPS, or CPU
device = select_torch_device(torch, args.gpu_id)
print(f"Using PyTorch device: {device}")
os.makedirs(os.path.join(args.imgs_dir_path, configs.test_params["patient_id"], 'predicted_seg'), exist_ok=True)

# Seeding for reproducibility
SEED = 19                                   
seed_torch_for_inference(torch, SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Collecting image paths 
imgs_path_list = glob(os.path.join(args.imgs_dir_path, configs.test_params["patient_id"], '*/slices/*.tif'))
imgs_path_list.sort(key=lambda f: int(re.sub('\D', '', f)))
# Defining image patches
nr_patches = configs.test_params["nr_patches"]

# Create a new instance of the model
model_ft = UNet(input_channels = configs.model_params["input_ch"], nclasses = configs.model_params["nr_classes"]) 
# Load the model
model_ft.load_state_dict(
    torch.load(
        os.path.join(args.model_path, configs.model_params["model_name"] + '.h5'),
        map_location="cpu",
    )
)
model = model_ft.to(device)
# Set the model in evaluation mode
model.eval()

# Start iterating through all the images
for img_path in tqdm(imgs_path_list): 

    # Image processing
    img = img_processing(img_path)

    # Image patches extraction
    patch_size = max(int (np.array(img).shape[0] / nr_patches), int (np.array(img).shape[1] / nr_patches))
    extractor = PatchExtractor(img=Image.fromarray(img), patch_size=patch_size, stride=patch_size)
    patches_img = extractor.extract_img_patches()
    # Tissue patches extraction via otsu tissue segmentation
    val = filters.threshold_otsu(np.asarray(img))
    seg_tissue = np.uint8((np.asarray(img) > val)*255)
    extractor = PatchExtractor(img=Image.fromarray(seg_tissue), patch_size=patch_size, stride=patch_size)
    patches_tiss = extractor.extract_img_patches()


    # Place holder to host the WSI predicted segmentation
    seg_wsi = np.zeros ((img.shape[0], img.shape[0]))
    # Initializing row and column indexes to reconstruct WSI
    r, c = 0, 0

    # Iterating through all the extracted image patches
    for i in range(len(patches_img)):
        # Patch selection
        if (((np.count_nonzero(np.asarray(patches_tiss[i]))/(np.asarray(patches_tiss[i]).shape[0]*np.asarray(patches_tiss[i]).shape[1])) > 0)):
            with torch.no_grad():
                # Patch resize to match the pretrained model resolution
                img_patch_resized = transform.resize(np.asarray(patches_img[i]), (configs.test_params["img_dim_row"], configs.test_params["img_dim_col"]))
                patch_tensor = transforms.ToTensor()(np.float32(img_patch_resized))

                # From 1 channel to 3 channels
                patch_tensor = torch.stack([patch_tensor,patch_tensor,patch_tensor], 1).to(device)
                # Patch segmentation prediction
                if torch.count_nonzero(patch_tensor) != 0:
                    masks_pred_prob = model(patch_tensor)
                    masks_pred = (masks_pred_prob > configs.test_params["probab_th"]).float()
                    patch_pred = np.asarray(masks_pred[0, 0, :, :].cpu())*255
                    patch_pred = patch_pred.astype(np.uint8)

        else:
            # Attach a background patch if the image patch is not selected
            patch_pred = np.zeros((configs.test_params["img_dim_row"], configs.test_params["img_dim_col"])).astype(np.uint8)

        # Reconstructing the WSI segmentation
        if ((r < img.shape[0]) & (c < img.shape[1])):
            seg_wsi[r:r+patch_size, c:c+patch_size] = transform.resize(patch_pred, (patch_size, patch_size), order = 0).astype(np.uint8)
            c += patch_size
            if (img.shape[1] - c < patch_size):
                seg_wsi[r:r+patch_size, c:c+patch_size] = np.zeros((patch_size, img.shape[1]-c)).astype(np.uint8)
                c = img.shape[1]
        if ((r < img.shape[0]) & (c == img.shape[1])):
            c = 0
            r += patch_size

    # Save in ".tif" file extension the WSI segmentation
    seg_wsi[seg_wsi > 0] = 1
    seg_save = Image.fromarray(seg_wsi)
    pos_seg_path = Path(img_path)
    seg_save.save(os.path.join(args.imgs_dir_path, configs.test_params["patient_id"], 'predicted_seg', pos_seg_path.stem + '.tif'))
