"""
Utility function for image pre-processing.
"""
import os
import cv2
import bisect
import argparse
import numpy as np

from PIL import Image
from skimage import filters
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.morphology import square, disk
    

def img_processing (img_path): 
    """
    Preprocess the WSI SR-microCT image, according to: 
    Poles, Isabella, et al. "On How to Unravel Bone Microscale Phenomena: 
    A Mask-Guided Attention SR-microCT Image Classification Approach." 
    2023 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI). IEEE, 2023.
    @param img_path: image path 
    @return: preprocessed image
    """
    img = np.asarray(Image.open(img_path))
    img_norm = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img_adj = imadjust(img_norm)
    img_gaus = gaussian_filter(img_adj, sigma = 2)

    if (np.sum(np.array(img) >= 0) > np.sum(np.array(img) <= 0)):
        val = filters.threshold_otsu(img_gaus)

        img_bin = img_gaus > val
        img_bin = np.array(img_bin).astype(bool)
    else: 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3
        im_v = np.reshape(img_gaus, (img_gaus.shape [0]*img_gaus.shape [1],1))

        _, labels, (centers) = cv2.kmeans(im_v, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        labels = (labels).flatten()

        segmented_image = centers[labels.flatten()]

        seg_img0 = segmented_image==centers[0]
        seg_img1 = segmented_image==centers[1]
        seg_img2 = segmented_image==centers[2]
        seg_res = [seg_img0.reshape(img_gaus.shape), seg_img1.reshape(img_gaus.shape), seg_img2.reshape(img_gaus.shape)]
        min_pix = np.argmin ((np.sum (seg_res[0]), np.sum (seg_res[1]), np.sum (seg_res[2])))

        img_bin = seg_res[min_pix]

    img_open = ndimage.binary_opening(img_bin, structure = square(20)).astype(bool)
    mask = ndimage.binary_closing(img_open, structure = disk(25)).astype(bool)

    img_adj[mask==0] = 0
    img_fin = img_adj.astype('uint8')

    return img_fin

def imadjust(src, tol = 1, vin = [0,255], vout = (0,255)):
    """
    Implementation of the maps the intensity values in grayscale image to new values. 
    By default, imadjust saturates the bottom 1% and the top 1% of all pixel values. 
    The function linearly maps pixel values between the saturation limits to values between 0 and 1. 
    This operation increases the contrast of the output image.
    @param src: input image
    @param tol: tolerance parameter
    @param vin: range of video intensities in input
    @param vout: range of video intensities in output
    @return: preprocessed image
    """
    assert len(src.shape) == 2 ,'Input image should be 2-dims'

    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(src,bins=list(range(256)),range=(0,255))[0]

        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, 255): 
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src-vin[0]
    vs[src<vin[0]]=0
    vd = vs*scale+0.5 + vout[0]
    vd[vd>vout[1]] = vout[1]
    dst = vd

    return dst

# Usage example
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Preprocess and save an image using img_processing.")
    parser.add_argument("--img-path", type=str, default='./Raw/slice_610.tif', help="Path to the input image.")
    parser.add_argument("--save-dir", type=str, default='./Cleaned', help="Directory to save the processed image.")
    args = parser.parse_args()

    # Check if input image exists
    if not os.path.exists(args.img_path):
        print(f"Error: The file '{args.img_path}' does not exist.")
        exit(1)

    # Ensure the save directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Process the image
    prep_img = img_processing(args.img_path)

    # Save the processed image
    filename = os.path.basename(args.img_path)
    save_path = os.path.join(args.save_dir, filename)

    img_save = Image.fromarray(prep_img)
    img_save.save(save_path)

    print(f"Processed {filename} image saved at: {save_path}")

if __name__ == "__main__":
    main()