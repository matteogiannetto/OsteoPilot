"""
Worker functions for microscopy preprocessing submitted to ProcessPoolExecutor.

These must live in a standalone module so Python's pickle can resolve them by
their stable qualified name (standard_microscopy_preprocessing_workers.<func>).  If they
were defined inside the tool file, the LangChain tool-scoping mechanism would
re-import that file under a private namespace (_scoped_tool_…), causing pickle
to see a name mismatch and raise PicklingError at runtime.
"""
import bisect
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage  # type: ignore[import-untyped]
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]
from skimage import filters
from skimage.morphology import disk, footprint_rectangle


def imadjust(
    src: np.ndarray,
    tol: int = 1,
    vin: list[int] = [0, 255],
    vout: Sequence[int] = (0, 255),
) -> np.ndarray:
    assert len(src.shape) == 2, "Input image should be 2-dims"

    tol = max(0, min(100, tol))

    if tol > 0:
        hist = np.histogram(src, bins=list(range(256)), range=(0, 255))[0]

        cum = hist.copy()
        for i in range(1, 255):
            cum[i] = cum[i - 1] + hist[i]

        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = src - vin[0]
    vs[src < vin[0]] = 0
    vd = vs * scale + 0.5 + vout[0]
    vd[vd > vout[1]] = vout[1]
    dst = vd

    return dst


def img_processing(img_path: str) -> np.ndarray:
    img = np.asarray(Image.open(img_path))
    # OpenCV's stubs do not express its supported ``dst=None`` API or the
    # concrete tuple returned by kmeans, so keep that library boundary dynamic.
    cv2_normalize: Any = cv2.normalize
    img_norm: Any = cv2_normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_adj = imadjust(img_norm)
    img_gaus = gaussian_filter(img_adj, sigma=2)

    if np.sum(np.array(img) >= 0) > np.sum(np.array(img) <= 0):
        # scikit-image does not expose type information for this runtime API.
        val = filters.threshold_otsu(img_gaus)  # type: ignore[no-untyped-call]
        img_bin = img_gaus > val
        img_bin = np.array(img_bin).astype(bool)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3
        im_v = np.reshape(img_gaus, (img_gaus.shape[0] * img_gaus.shape[1], 1))

        cv2_kmeans: Any = cv2.kmeans
        kmeans_result: Any = cv2_kmeans(
            im_v,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        _, labels, raw_centers = kmeans_result
        centers: Any = np.uint8(raw_centers)
        labels = labels.flatten()

        segmented_image = centers[labels.flatten()]

        seg_img0 = segmented_image == centers[0]
        seg_img1 = segmented_image == centers[1]
        seg_img2 = segmented_image == centers[2]
        seg_res = [
            seg_img0.reshape(img_gaus.shape),
            seg_img1.reshape(img_gaus.shape),
            seg_img2.reshape(img_gaus.shape),
        ]
        min_pix = np.argmin((np.sum(seg_res[0]), np.sum(seg_res[1]), np.sum(seg_res[2])))
        img_bin = seg_res[min_pix]

    # ``footprint_rectangle((20, 20))`` is the non-deprecated equivalent of
    # the historical ``square(20)`` footprint used by this pipeline.
    opening_footprint = footprint_rectangle((20, 20))  # type: ignore[no-untyped-call]
    img_open = ndimage.binary_opening(img_bin, structure=opening_footprint).astype(bool)
    # scikit-image does not expose type information for this runtime API.
    closing_footprint = disk(25)  # type: ignore[no-untyped-call]
    mask = ndimage.binary_closing(img_open, structure=closing_footprint).astype(bool)

    img_adj[mask == 0] = 0
    img_fin = img_adj.astype("uint8")

    return img_fin


def _preprocess_tiff_worker(tiff_path: str, output_path: str) -> tuple[str, str | None]:
    try:
        with Image.open(tiff_path) as im:
            if getattr(im, "n_frames", 1) != 1:
                return output_path, "Multi-page TIFFs are not supported; provide a single-slice 2D image."
            arr = np.asarray(im)
            if arr.ndim != 2:
                return output_path, "Image must be grayscale (2D). Provide a single-channel TIFF."

        proc = img_processing(tiff_path)
        Image.fromarray(proc).save(output_path)
        return output_path, None
    except AssertionError as ae:
        return output_path, f"Algorithm assertion failed: {ae}. Ensure the image is 2D grayscale."
    except Exception as exc:
        return output_path, f"{type(exc).__name__}: {exc}"


def _preprocess_nifti_slice_worker(
    z_index: int,
    slice_array: np.ndarray,
) -> tuple[int, np.ndarray]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"slice_{z_index:04d}.tiff"
        Image.fromarray(np.asarray(slice_array)).save(temp_path)
        return z_index, img_processing(str(temp_path))
