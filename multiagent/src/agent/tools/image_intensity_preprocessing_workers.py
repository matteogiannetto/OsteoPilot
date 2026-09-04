"""
Worker functions for image_intensity_preprocessing_tools submitted to ProcessPoolExecutor.

These must live in a standalone module so Python's pickle can resolve them by
their stable qualified name (image_intensity_preprocessing_workers.<func>).  If they were
defined inside the tool file, the LangChain tool-scoping mechanism would
re-import that file under a private namespace (_scoped_tool_…), causing pickle
to see a name mismatch and raise PicklingError at runtime.

Pure helper functions (_z_axis, _z_slice, _cast_normalized, _cast_filtered,
_stats_summary) are also kept here so the worker functions are fully
self-contained when reimported in worker processes.  The tool file imports
them back from here so there is only one definition.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
from typing import Any, cast

from numpy.typing import NDArray

from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]
from skimage.filters import threshold_otsu


# ---------------------------------------------------------------------------
# Pure helpers (also re-exported to image_intensity_preprocessing_tools)
# ---------------------------------------------------------------------------

def _z_axis(shape: tuple[int, ...]) -> int:
    if not shape:
        return 0
    return len(shape) - 1


def _z_slice(shape: tuple[int, ...], z_index: int) -> tuple[slice | int, ...]:
    selector: list[slice | int] = [slice(None)] * len(shape)
    selector[_z_axis(shape)] = z_index
    return tuple(selector)


def _cast_normalized(array: np.ndarray, dtype_name: str) -> NDArray[np.generic]:
    if dtype_name == "float32":
        return array.astype(np.float32)
    if dtype_name == "uint8":
        return cast(
            NDArray[np.generic],
            np.clip(np.rint(array), 0, np.iinfo(np.uint8).max).astype(np.uint8),
        )
    if dtype_name == "uint16":
        return cast(
            NDArray[np.generic],
            np.clip(np.rint(array), 0, np.iinfo(np.uint16).max).astype(np.uint16),
        )
    raise ValueError(f"Unsupported output dtype: {dtype_name}")


def _cast_filtered(
    array: np.ndarray,
    source_dtype: np.dtype[Any],
    dtype_name: str,
) -> NDArray[np.generic]:
    if dtype_name == "float32":
        return array.astype(np.float32)
    if dtype_name == "source":
        if np.issubdtype(source_dtype, np.integer):
            limits = np.iinfo(source_dtype)
            return cast(
                NDArray[np.generic],
                np.clip(np.rint(array), limits.min, limits.max).astype(source_dtype),
            )
        return array.astype(source_dtype)
    raise ValueError(f"Unsupported output dtype: {dtype_name}")


def _stats_summary(array: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(array)
    if data.size == 0:
        return {"count": 0, "mean": 0.0, "m2": 0.0, "min": 0.0, "max": 0.0}
    data_float = data.astype(np.float64, copy=False)
    mean = float(np.mean(data_float))
    return {
        "count": int(data_float.size),
        "mean": mean,
        "m2": float(np.sum((data_float - mean) ** 2)),
        "min": float(np.min(data_float)),
        "max": float(np.max(data_float)),
    }


# ---------------------------------------------------------------------------
# Slice-level workers
# ---------------------------------------------------------------------------

def _normalize_slice_worker(
    *,
    z_index: int,
    chunk: np.ndarray,
    output_min: float,
    output_max: float,
    output_dtype: str,
) -> tuple[int, np.ndarray, float, float, bool]:
    input_min = float(np.min(chunk))
    input_max = float(np.max(chunk))
    constant_slice = input_max == input_min
    if constant_slice:
        scaled = np.full(chunk.shape, float(output_min), dtype=np.asarray(chunk).dtype)
    else:
        scaled = (chunk - input_min) / (input_max - input_min)
        scaled = scaled * (output_max - output_min) + output_min
    cast = _cast_normalized(scaled, output_dtype)
    return z_index, cast, input_min, input_max, constant_slice


# ---------------------------------------------------------------------------
# ProcessPoolExecutor workers (NIfTI per-slice → memmap)
# ---------------------------------------------------------------------------

def _normalize_nifti_slice_to_memmap_worker(
    *,
    input_path: str,
    output_memmap_path: str,
    shape: tuple[int, ...],
    output_np_dtype_name: str,
    z_index: int,
    output_min: float,
    output_max: float,
    output_dtype: str,
) -> tuple[int, dict[str, float | int], dict[str, float | int], float, float, bool]:
    img: Any = nib.load(input_path)
    chunk = np.asarray(img.dataobj[_z_slice(shape, z_index)], dtype=np.float32)
    input_stats = _stats_summary(chunk)
    _index, cast, input_min, input_max, constant_slice = _normalize_slice_worker(
        z_index=z_index,
        chunk=chunk,
        output_min=output_min,
        output_max=output_max,
        output_dtype=output_dtype,
    )
    output = np.memmap(
        output_memmap_path,
        dtype=np.dtype(output_np_dtype_name),
        mode="r+",
        shape=shape,
    )
    output[_z_slice(shape, z_index)] = cast
    output.flush()
    output_stats = _stats_summary(cast)
    del output, chunk, cast
    return z_index, input_stats, output_stats, input_min, input_max, constant_slice


def _gaussian_nifti_slice_to_memmap_worker(
    *,
    input_path: str,
    output_memmap_path: str,
    shape: tuple[int, ...],
    output_np_dtype_name: str,
    z_index: int,
    sigma: float,
    source_dtype_name: str,
    output_dtype: str,
) -> tuple[int, dict[str, float | int], dict[str, float | int]]:
    img: Any = nib.load(input_path)
    chunk = np.asarray(img.dataobj[_z_slice(shape, z_index)], dtype=np.float32)
    input_stats = _stats_summary(chunk)
    filtered = gaussian_filter(chunk, sigma=float(sigma))
    cast = _cast_filtered(filtered, np.dtype(source_dtype_name), output_dtype)
    output = np.memmap(
        output_memmap_path,
        dtype=np.dtype(output_np_dtype_name),
        mode="r+",
        shape=shape,
    )
    output[_z_slice(shape, z_index)] = cast
    output.flush()
    output_stats = _stats_summary(cast)
    del output, chunk, filtered, cast
    return z_index, input_stats, output_stats


def _otsu_nifti_slice_to_memmap_worker(
    *,
    input_path: str,
    output_memmap_path: str,
    shape: tuple[int, ...],
    output_np_dtype_name: str,
    z_index: int,
    output_mode: str,
    foreground_value: int,
    background_value: int,
) -> tuple[int, dict[str, float | int], float, bool, int]:
    output_np_dtype = np.dtype(output_np_dtype_name)
    img: Any = nib.load(input_path)
    chunk = np.asarray(img.dataobj[_z_slice(shape, z_index)], dtype=np.float32)
    input_min = float(np.min(chunk))
    input_max = float(np.max(chunk))
    constant_slice = input_max == input_min
    if constant_slice:
        threshold = input_min
    else:
        # scikit-image does not expose type information for this runtime API.
        threshold = float(threshold_otsu(chunk.reshape(-1)))  # type: ignore[no-untyped-call]
    threshold_mask = chunk > threshold
    retained_voxels = int(np.count_nonzero(threshold_mask))
    if output_mode == "binary_mask":
        output_slice = np.where(threshold_mask, foreground_value, background_value).astype(np.uint8)
    else:
        output_slice = np.where(threshold_mask, chunk, 0).astype(output_np_dtype, copy=False)
    output = np.memmap(
        output_memmap_path,
        dtype=output_np_dtype,
        mode="r+",
        shape=shape,
    )
    output[_z_slice(shape, z_index)] = output_slice
    output.flush()
    output_stats = _stats_summary(output_slice)
    del output, chunk, output_slice, threshold_mask
    return z_index, output_stats, threshold, constant_slice, retained_voxels
