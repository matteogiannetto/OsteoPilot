from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
import difflib
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator, Literal

import nibabel as nib
import numpy as np
import tifffile
from PIL import Image
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]
from skimage.filters import threshold_otsu

from agent.tools.id_factory import new_id
from agent.tools.path_resolution import missing_input_path_message, resolve_input_path
from agent.tools.image_intensity_preprocessing_workers import (
    _z_axis,
    _cast_normalized,
    _cast_filtered,
    _normalize_slice_worker,
    _normalize_nifti_slice_to_memmap_worker,
    _gaussian_nifti_slice_to_memmap_worker,
    _otsu_nifti_slice_to_memmap_worker,
)

logger = logging.getLogger(__name__)

TOOL_KIND = "image_preprocessing"
SUPPORTED_FORMATS = ".tif, .tiff, .nii, .nii.gz, or a folder of TIFF slices"
NIFTI_STREAM_TARGET_BYTES = 128 * 1024 * 1024
OTSU_HISTOGRAM_BINS = 65536
MAX_MINMAX_NIFTI_WORKERS = 10
MAX_MINMAX_NIFTI_INFLIGHT_SLICES = 10
MAX_GAUSSIAN_NIFTI_WORKERS = 10
MAX_GAUSSIAN_NIFTI_INFLIGHT_SLICES = 10
MAX_OTSU_NIFTI_WORKERS = 10
MAX_OTSU_NIFTI_INFLIGHT_SLICES = 10

_GENERATED_NAME_PREFIXES = (
    "otsu_thresholded",
    "gaussian",
    "gauss",
    "norm",
    "normalized",
    "seg",
    "mask",
    "preprocessed",
    "prep",
    "otsu",
)
_GENERATED_NAME_SUFFIXES = (
    r"_min_max_normalized",
    r"_gaussian_sigma_\d+(?:_\d+)?",
    r"_otsu_binary_mask",
    r"_otsu_threshold",
    r"_segmented",
    r"_preprocessed",
    r"_original",
    r"_mask",
)


class ImagePathResolutionError(FileNotFoundError):
    """Expected, user-correctable image input path resolution failure."""

    def __init__(
        self,
        *,
        image_path: str,
        searched_paths: list[str],
        session_path: str | None = None,
        workspace_root: str | None = None,
    ) -> None:
        self.image_path = image_path
        self.searched_paths = list(dict.fromkeys(str(path) for path in searched_paths))
        self.session_path = session_path
        self.workspace_root = workspace_root
        super().__init__(
            missing_input_path_message("Image path", image_path, self.searched_paths)
        )


def _fail(message: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "ok": False,
        "success": False,
        "tool_kind": TOOL_KIND,
        "error": message,
        "message": message,
        "attachments": [],
    }
    payload.update(extra)
    return payload


def _candidate_existing_paths(
    image_path: str,
    *,
    session_path: str | None,
    workspace_root: str | None,
    limit: int = 20,
) -> list[str]:
    filename = Path(image_path).name
    if not filename:
        return []

    roots: list[Path] = []
    for root_value in (session_path, workspace_root):
        if not root_value:
            continue
        root = Path(root_value).expanduser()
        roots.append(root)

    matches: list[str] = []
    seen: set[str] = set()
    for root in roots:
        try:
            candidate = _closest_existing_path(root / image_path)
            if candidate is not None and candidate.exists():
                resolved = str(candidate.resolve(strict=False))
                if resolved in seen:
                    continue
                seen.add(resolved)
                matches.append(resolved)
                if len(matches) >= limit:
                    return matches
        except OSError:
            continue
    return matches


def _closest_existing_path(path: Path) -> Path | None:
    """
    Suggest a nearby existing path without recursively scanning large data trees.

    This walks the requested path until a component is missing, then compares that
    component against siblings in the existing parent directory. It catches errors
    like ``unpreprocessed`` vs ``unpreprocessd`` while staying cheap for huge
    microscopy datasets.
    """
    parts = path.parts
    if not parts:
        return None

    current = Path(parts[0])
    index = 1
    while index < len(parts) and current.exists():
        next_path = current / parts[index]
        if next_path.exists():
            current = next_path
            index += 1
            continue
        break

    if index >= len(parts) or not current.exists() or not current.is_dir():
        return None

    requested_part = parts[index]
    sibling_names = [child.name for child in current.iterdir()]
    close_names = difflib.get_close_matches(
        requested_part,
        sibling_names,
        n=5,
        cutoff=0.78,
    )
    for close_name in close_names:
        candidate = current / close_name
        for remaining_part in parts[index + 1 :]:
            candidate = candidate / remaining_part
        if candidate.exists():
            return candidate
    return None


def _image_path_resolution_failure(exc: ImagePathResolutionError) -> dict[str, Any]:
    candidate_paths = _candidate_existing_paths(
        exc.image_path,
        session_path=exc.session_path,
        workspace_root=exc.workspace_root,
    )
    message = (
        "The image file or folder could not be found, so image preprocessing was "
        "not executed. Use an existing session attachment path, an existing absolute "
        "path, or an existing path relative to the workspace root."
    )
    return _fail(
        message,
        error_type="input_path_not_found",
        error_code="IMAGE_INPUT_PATH_NOT_FOUND",
        recoverable=True,
        image_path=exc.image_path,
        input_path=exc.image_path,
        searched_paths=exc.searched_paths,
        candidate_paths=candidate_paths,
        session_path=exc.session_path,
        workspace_root=exc.workspace_root,
        supported_formats=SUPPORTED_FORMATS,
        user_action=(
            "Verify the filename and directory, inspect session/data or workspace/data "
            "if needed, and retry with one of the existing candidate paths."
        ),
    )


def _safe_name_token(value: str, *, fallback: str = "sample", max_length: int = 80) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("._-")
    if not token:
        token = fallback
    return token[:max_length].rstrip("._-") or fallback


def _source_stem(path: Path) -> str:
    if path.is_dir():
        return path.name
    name = path.name
    for suffix in (".nii.gz", ".tiff", ".tif", ".nii"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _sample_id_from_path(path: Path) -> str:
    stem = _source_stem(path)
    if "__" in stem:
        stem = stem.split("__", 1)[0]
    stem = re.sub(r"_20\d{6}_\d{6}(?=$|_)", "", stem)

    changed = True
    while changed:
        changed = False
        lowered = stem.lower()
        for prefix in _GENERATED_NAME_PREFIXES:
            marker = f"{prefix}_"
            if lowered.startswith(marker):
                stem = stem[len(marker) :]
                changed = True
                break
        if changed:
            continue
        for suffix in _GENERATED_NAME_SUFFIXES:
            trimmed = re.sub(suffix + r"(?:_20\d{6}_\d{6})?(?=$|_)", "", stem, flags=re.IGNORECASE)
            if trimmed != stem:
                stem = trimmed
                changed = True
                break

    return _safe_name_token(stem, fallback=path.parent.name or "sample")


def _gaussian_operation_code(sigma: float) -> str:
    return "gauss-s" + ("%g" % float(sigma)).replace(".", "p").replace("-", "m")


def _artifact_path(
    *,
    source_path: Path,
    output_dir: Path,
    operation_code: str,
    extension: str,
) -> tuple[Path, dict[str, str]]:
    artifact_id = new_id("img")
    short_id = artifact_id.rsplit("-", 1)[-1][-8:].lower()
    sample_id = _sample_id_from_path(source_path)
    clean_operation = _safe_name_token(operation_code, fallback="artifact", max_length=32)
    ext = extension if extension.startswith(".") else f".{extension}"
    return (
        output_dir / f"{sample_id}__{clean_operation}__{short_id}{ext}",
        {
            "artifact_id": artifact_id,
            "sample_id": sample_id,
            "operation_code": clean_operation,
        },
    )


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, values: np.ndarray) -> None:
        data = np.asarray(values)
        if data.size == 0:
            return
        data_float = data.astype(np.float64, copy=False)
        chunk_count = int(data_float.size)
        chunk_mean = float(np.mean(data_float))
        chunk_m2 = float(np.sum((data_float - chunk_mean) ** 2))
        chunk_min = float(np.min(data_float))
        chunk_max = float(np.max(data_float))

        if self.count == 0:
            self.count = chunk_count
            self.mean = chunk_mean
            self.m2 = chunk_m2
            self.minimum = chunk_min
            self.maximum = chunk_max
            return

        total_count = self.count + chunk_count
        delta = chunk_mean - self.mean
        self.mean += delta * chunk_count / total_count
        self.m2 += chunk_m2 + delta * delta * self.count * chunk_count / total_count
        self.count = total_count
        self.minimum = min(self.minimum, chunk_min)
        self.maximum = max(self.maximum, chunk_max)

    def merge_summary(
        self,
        *,
        count: int,
        mean: float,
        m2: float,
        minimum: float,
        maximum: float,
    ) -> None:
        if count == 0:
            return
        if self.count == 0:
            self.count = int(count)
            self.mean = float(mean)
            self.m2 = float(m2)
            self.minimum = float(minimum)
            self.maximum = float(maximum)
            return

        total_count = self.count + int(count)
        delta = float(mean) - self.mean
        self.mean += delta * int(count) / total_count
        self.m2 += float(m2) + delta * delta * self.count * int(count) / total_count
        self.count = total_count
        self.minimum = min(self.minimum, float(minimum))
        self.maximum = max(self.maximum, float(maximum))

    def as_dict(self) -> dict[str, float]:
        if self.count == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": float(np.sqrt(self.m2 / self.count)),
        }


class LoadedImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Any
    source_kind: Literal["tiff_file", "tiff_folder", "nifti"]
    source_path: Path
    affine: Any = None
    header: Any = None


class ImagePreprocessingBaseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(
        ...,
        description=f"Path to the image or volume to transform. Supported inputs: {SUPPORTED_FORMATS}.",
    )
    session_path: str | None = Field(
        default=None,
        description="INTERNAL: optional active session directory. Injected by the subgraph when available.",
    )
    workspace_root: str | None = Field(
        default=None,
        description="INTERNAL: optional workspace root used to resolve relative paths.",
    )
    output_prefix: str | None = Field(
        default=None,
        description="Deprecated.",
    )
    output_format: Literal["auto", "tiff", "nifti"] = Field(
        default="auto",
        description=(
            "Output format. Use 'auto' to preserve NIfTI inputs as NIfTI and all TIFF inputs "
            "as a TIFF file. Use 'tiff' when the task explicitly asks for a TIFF stack."
        ),
    )


class MinMaxNormalizationArgs(ImagePreprocessingBaseArgs):
    output_min: float = Field(
        default=0.0,
        description="Target value assigned to the minimum input intensity by linear scaling.",
    )
    output_max: float = Field(
        default=1.0,
        description="Target value assigned to the maximum input intensity by linear scaling.",
    )
    output_dtype: Literal["float32", "uint8", "uint16"] = Field(
        default="float32",
        description=(
            "Storage dtype for the normalized output. Use float32 for ranges like [0, 1]; "
            "use uint8 or uint16 only when the requested output range is suitable for integer storage."
        ),
    )

    @model_validator(mode="after")
    def _validate_range(self) -> "MinMaxNormalizationArgs":
        if self.output_min >= self.output_max:
            raise ValueError("output_min must be smaller than output_max.")
        return self


class GaussianFilteringArgs(ImagePreprocessingBaseArgs):
    sigma: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Gaussian kernel sigma in pixels/voxels. Larger values produce stronger smoothing.",
    )
    output_dtype: Literal["float32", "source"] = Field(
        default="float32",
        description="Storage dtype for the filtered output. 'source' casts back to the input dtype after clipping.",
    )


class OtsuThresholdingArgs(ImagePreprocessingBaseArgs):
    output_mode: Literal["masked_image", "binary_mask"] = Field(
        default="masked_image",
        description=(
            "Controls the saved artifact. 'masked_image' preserves the current behavior: "
            "input values above the Otsu threshold are kept and all other values become 0. "
            "'binary_mask' saves only the threshold mask using foreground_value/background_value."
        ),
    )
    foreground_value: int = Field(
        default=1,
        ge=1,
        le=255,
        description=(
            "Value written for pixels/voxels above the Otsu threshold when output_mode='binary_mask'. "
            "Ignored when output_mode='masked_image'."
        ),
    )
    background_value: int = Field(
        default=0,
        ge=0,
        le=254,
        description=(
            "Value written for pixels/voxels at or below the Otsu threshold when output_mode='binary_mask'. "
            "Ignored when output_mode='masked_image'."
        ),
    )


class ImageIO:
    @staticmethod
    def resolve(image_path: str, session_path: str | None, workspace_root: str | None) -> Path:
        resolved = resolve_input_path(
            image_path,
            session_path=session_path,
            workspace_root=workspace_root,
        )
        if resolved.path is not None:
            return resolved.path
        raise ImagePathResolutionError(
            image_path=image_path,
            searched_paths=resolved.searched_paths,
            session_path=session_path,
            workspace_root=workspace_root,
        )

    @staticmethod
    def load(path: Path) -> LoadedImage:
        if path.is_dir():
            tiff_files = sorted(
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
            )
            if not tiff_files:
                raise ValueError(f"No TIFF slices found in folder: {path}")
            slices: list[np.ndarray] = []
            for tiff_file in tiff_files:
                with Image.open(str(tiff_file)) as img:
                    if getattr(img, "n_frames", 1) != 1:
                        raise ValueError(
                            f"TIFF folders must contain single-frame slices; multipage file found: {tiff_file}"
                        )
                    slices.append(np.asarray(img))
            shape0 = slices[0].shape
            if any(slice_.shape != shape0 for slice_ in slices):
                raise ValueError("All TIFF slices in a folder must have the same shape.")
            return LoadedImage(data=np.stack(slices, axis=0), source_kind="tiff_folder", source_path=path)

        if not path.is_file():
            raise ValueError(f"Path is neither a file nor a folder: {path}")

        suffixes = "".join(path.suffixes).lower()
        if ".nii" in suffixes:
            nifti_image: Any = nib.load(str(path))
            return LoadedImage(
                data=nifti_image.dataobj,
                source_kind="nifti",
                source_path=path,
                affine=nifti_image.affine,
                header=nifti_image.header.copy(),
            )

        if ".tif" in suffixes or ".tiff" in suffixes:
            with Image.open(str(path)) as img:
                n_frames = getattr(img, "n_frames", 1)
                if n_frames == 1:
                    data = np.asarray(img)
                else:
                    frames = []
                    for frame_index in range(n_frames):
                        img.seek(frame_index)
                        frames.append(np.asarray(img))
                    data = np.stack(frames, axis=0)
            return LoadedImage(data=data, source_kind="tiff_file", source_path=path)

        raise ValueError(f"Unsupported image format: {path}. Supported inputs are {SUPPORTED_FORMATS}.")

    @staticmethod
    def output_dir(session_path: str | None) -> Path:
        if session_path:
            path = Path(session_path) / "image_preprocessing"
        else:
            path = Path("statics") / "output_microscopy" / "image_preprocessing"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def base_name(path: Path) -> str:
        name = path.name
        for suffix in [".nii.gz", ".tiff", ".tif", ".nii"]:
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return path.stem

    @staticmethod
    def save(
        *,
        array: np.ndarray,
        loaded: LoadedImage,
        output_dir: Path,
        operation_code: str,
        output_prefix: str | None,
        output_format: Literal["auto", "tiff", "nifti"],
    ) -> tuple[Path, str, dict[str, str]]:
        selected_format = output_format
        if selected_format == "auto":
            selected_format = "nifti" if loaded.source_kind == "nifti" else "tiff"

        if selected_format == "nifti":
            output_path, artifact = _artifact_path(
                source_path=loaded.source_path,
                output_dir=output_dir,
                operation_code=operation_code,
                extension=".nii.gz",
            )
            affine = loaded.affine if loaded.affine is not None else np.eye(4)
            header = loaded.header.copy() if loaded.header is not None else None
            # NiBabel's installed stubs omit this stable image-construction API.
            img: Any = nib.Nifti1Image(  # type: ignore[no-untyped-call]
                array,
                affine=affine,
                header=header,
            )
            img.header.set_data_dtype(array.dtype)
            nib.save(img, str(output_path))
            return output_path, "nifti", artifact

        output_path, artifact = _artifact_path(
            source_path=loaded.source_path,
            output_dir=output_dir,
            operation_code=operation_code,
            extension=".tif",
        )
        tiff_array = array
        if loaded.source_kind == "nifti" and np.asarray(array).ndim == 3:
            tiff_array = np.moveaxis(array, -1, 0)
        ImageIO.save_tiff(tiff_array, output_path)
        return output_path, "tiff", artifact

    @staticmethod
    def save_tiff(array: np.ndarray, output_path: Path) -> None:
        data = np.asarray(array)
        if data.ndim == 2:
            Image.fromarray(data).save(str(output_path))
            return
        if data.ndim != 3:
            raise ValueError(f"TIFF output supports 2D images or 3D stacks, got shape {data.shape}.")
        frames = [Image.fromarray(data[index]) for index in range(data.shape[0])]
        frames[0].save(str(output_path), save_all=True, append_images=frames[1:])

    @staticmethod
    def tiff_output_path(
        *,
        source_path: Path,
        output_dir: Path,
        operation_code: str,
        output_prefix: str | None,
    ) -> tuple[Path, dict[str, str]]:
        return _artifact_path(
            source_path=source_path,
            output_dir=output_dir,
            operation_code=operation_code,
            extension=".tif",
        )

    @staticmethod
    def save_tiff_stack_from_array(
        *,
        array: np.ndarray,
        source_path: Path,
        output_dir: Path,
        operation_code: str,
        output_prefix: str | None,
        stack_axis: int = -1,
    ) -> tuple[Path, dict[str, str]]:
        output_path, artifact = ImageIO.tiff_output_path(
            source_path=source_path,
            output_dir=output_dir,
            operation_code=operation_code,
            output_prefix=output_prefix,
        )
        data = np.asarray(array)
        if data.ndim == 2:
            tifffile.imwrite(str(output_path), data)
            return output_path, artifact
        if data.ndim != 3:
            raise ValueError(f"TIFF output supports 2D images or 3D stacks, got shape {data.shape}.")
        axis = stack_axis if stack_axis >= 0 else data.ndim + stack_axis
        with tifffile.TiffWriter(str(output_path), bigtiff=True) as writer:
            for index in range(data.shape[axis]):
                writer.write(np.asarray(np.take(data, index, axis=axis)))
        return output_path, artifact

    @staticmethod
    def nifti_output_path(
        *,
        source_path: Path,
        output_dir: Path,
        operation_code: str,
        output_prefix: str | None,
    ) -> tuple[Path, dict[str, str]]:
        return _artifact_path(
            source_path=source_path,
            output_dir=output_dir,
            operation_code=operation_code,
            extension=".nii.gz",
        )

    @staticmethod
    def save_nifti_from_array(
        *,
        array: np.ndarray,
        source_path: Path,
        affine: Any,
        header: Any,
        output_dir: Path,
        operation_code: str,
        output_prefix: str | None,
    ) -> tuple[Path, dict[str, str]]:
        output_path, artifact = ImageIO.nifti_output_path(
            source_path=source_path,
            output_dir=output_dir,
            operation_code=operation_code,
            output_prefix=output_prefix,
        )
        output_header = header.copy() if header is not None else None
        # NiBabel's installed stubs omit this stable image-construction API.
        img: Any = nib.Nifti1Image(  # type: ignore[no-untyped-call]
            array,
            affine=affine if affine is not None else np.eye(4),
            header=output_header,
        )
        img.header.set_data_dtype(array.dtype)
        nib.save(img, str(output_path))
        return output_path, artifact


def _safe_stats(array: np.ndarray) -> dict[str, float]:
    data = np.asarray(array)
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }


def _slice_axis_for_array(array: np.ndarray, loaded: LoadedImage) -> int:
    if array.ndim < 3:
        return 0
    return -1 if loaded.source_kind == "nifti" else 0


def _array_slice(array: np.ndarray, axis: int, index: int) -> tuple[slice | int, ...]:
    axis = axis if axis >= 0 else array.ndim + axis
    selector: list[slice | int] = [slice(None)] * array.ndim
    selector[axis] = index
    return tuple(selector)


def _normalize_array_per_slice(
    data: np.ndarray,
    *,
    output_min: float,
    output_max: float,
    output_dtype: str,
    slice_axis: int,
) -> tuple[np.ndarray, list[dict[str, float | int]], list[int]]:
    output = np.empty(data.shape, dtype=_dtype_for_normalized(output_dtype))
    per_slice_ranges: list[dict[str, float | int]] = []
    constant_slices: list[int] = []
    axis = slice_axis if slice_axis >= 0 else data.ndim + slice_axis
    max_workers = _bounded_worker_count(
        data.shape[axis],
        max_default=MAX_MINMAX_NIFTI_WORKERS,
    )
    max_inflight = max(
        max_workers,
        min(data.shape[axis], MAX_MINMAX_NIFTI_INFLIGHT_SLICES),
    )
    pending: set[Future[tuple[int, np.ndarray, float, float, bool]]] = set()

    def consume_completed(
        done_futures: set[Future[tuple[int, np.ndarray, float, float, bool]]],
    ) -> None:
        for future in done_futures:
            index, cast, input_min, input_max, constant_slice = future.result()
            output[_array_slice(data, axis, index)] = cast
            per_slice_ranges.append({"slice": index, "input_min": input_min, "input_max": input_max})
            if constant_slice:
                constant_slices.append(index)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index in range(data.shape[axis]):
            selector = _array_slice(data, axis, index)
            slice_data = data[selector].astype(np.float64, copy=False)
            pending.add(
                executor.submit(
                    _normalize_slice_worker,
                    z_index=index,
                    chunk=slice_data,
                    output_min=float(output_min),
                    output_max=float(output_max),
                    output_dtype=output_dtype,
                )
            )
            if len(pending) >= max_inflight:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                consume_completed(done)

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            consume_completed(done)

    per_slice_ranges.sort(key=lambda item: int(item["slice"]))
    constant_slices.sort()
    return output, per_slice_ranges, constant_slices


def _gaussian_array_per_slice(
    data: np.ndarray,
    *,
    sigma: float,
    source_dtype: np.dtype,
    output_dtype: str,
    slice_axis: int,
) -> np.ndarray:
    output_np_dtype = np.dtype(np.float32) if output_dtype == "float32" else source_dtype
    output = np.empty(data.shape, dtype=output_np_dtype)
    axis = slice_axis if slice_axis >= 0 else data.ndim + slice_axis
    for index in range(data.shape[axis]):
        selector = _array_slice(data, axis, index)
        filtered = gaussian_filter(data[selector].astype(np.float32, copy=False), sigma=float(sigma))
        output[selector] = _cast_filtered(filtered, source_dtype, output_dtype)
    return output


def _otsu_array_per_slice(
    data: np.ndarray,
    *,
    output_mode: str,
    foreground_value: int,
    background_value: int,
    slice_axis: int,
) -> tuple[np.ndarray, list[dict[str, float | int]], list[int], int]:
    output_dtype = np.dtype(np.uint8) if output_mode == "binary_mask" else data.dtype
    output = np.empty(data.shape, dtype=output_dtype)
    thresholds: list[dict[str, float | int]] = []
    constant_slices: list[int] = []
    retained_voxels = 0
    axis = slice_axis if slice_axis >= 0 else data.ndim + slice_axis
    for index in range(data.shape[axis]):
        selector = _array_slice(data, axis, index)
        slice_data = data[selector]
        input_min = float(np.min(slice_data))
        input_max = float(np.max(slice_data))
        if input_max == input_min:
            threshold = input_min
            constant_slices.append(index)
        else:
            # scikit-image does not expose type information for this runtime API.
            threshold = float(threshold_otsu(slice_data.reshape(-1)))  # type: ignore[no-untyped-call]
        thresholds.append({"slice": index, "threshold": threshold})
        threshold_mask = slice_data > threshold
        retained_voxels += int(np.count_nonzero(threshold_mask))
        if output_mode == "binary_mask":
            output[selector] = np.where(threshold_mask, foreground_value, background_value).astype(np.uint8)
        else:
            output[selector] = np.where(threshold_mask, slice_data, 0).astype(data.dtype, copy=False)
    return output, thresholds, constant_slices, retained_voxels


def _bounded_worker_count(total_items: int, *, max_default: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(total_items), int(cpu_count), int(max_default)))


def _gaussian_filter_slice_worker(
    *,
    z_index: int,
    chunk: np.ndarray,
    sigma: float,
    source_dtype: np.dtype,
    output_dtype: str,
) -> tuple[int, np.ndarray]:
    filtered = gaussian_filter(chunk, sigma=float(sigma))
    cast = _cast_filtered(filtered, source_dtype, output_dtype)
    return z_index, cast


def _dtype_for_normalized(dtype_name: str) -> np.dtype:
    if dtype_name == "float32":
        return np.dtype(np.float32)
    if dtype_name == "uint8":
        return np.dtype(np.uint8)
    if dtype_name == "uint16":
        return np.dtype(np.uint16)
    raise ValueError(f"Unsupported output dtype: {dtype_name}")


def _is_nifti_path(path: Path) -> bool:
    return ".nii" in "".join(path.suffixes).lower()


def _nifti_chunk_axis(shape: tuple[int, ...]) -> int:
    if not shape:
        return 0
    return int(np.argmin(shape))


def _nifti_chunk_depth(shape: tuple[int, ...], axis: int, dtype: np.dtype, *, multiplier: int = 2) -> int:
    if not shape:
        return 1
    plane_values = int(np.prod([dim for index, dim in enumerate(shape) if index != axis], dtype=np.int64))
    bytes_per_slab = max(1, plane_values * np.dtype(dtype).itemsize * multiplier)
    return max(1, int(NIFTI_STREAM_TARGET_BYTES // bytes_per_slab))


def _chunk_slice(ndim: int, axis: int, start: int, stop: int) -> tuple[slice, ...]:
    slices = [slice(None)] * ndim
    slices[axis] = slice(start, stop)
    return tuple(slices)


def _iter_nifti_chunks(
    dataobj: Any,
    *,
    dtype: Any = np.float32,
) -> Iterator[tuple[tuple[slice, ...], np.ndarray]]:
    shape = tuple(int(dim) for dim in dataobj.shape)
    axis = _nifti_chunk_axis(shape)
    depth = _nifti_chunk_depth(shape, axis, np.dtype(dtype))
    for start in range(0, shape[axis], depth):
        stop = min(shape[axis], start + depth)
        selector = _chunk_slice(len(shape), axis, start, stop)
        yield selector, np.asarray(dataobj[selector], dtype=dtype)


def _input_stats_nifti(dataobj: Any) -> RunningStats:
    stats = RunningStats()
    for _, chunk in _iter_nifti_chunks(dataobj, dtype=np.float32):
        stats.update(chunk)
    return stats


def _safe_stats_nifti(dataobj: Any) -> dict[str, float]:
    return _input_stats_nifti(dataobj).as_dict()


def _memmap_for_nifti(shape: tuple[int, ...], dtype: np.dtype) -> tuple[np.memmap, Path]:
    temp = tempfile.NamedTemporaryFile(prefix="image_preprocessing_", suffix=".dat", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    array = np.memmap(temp_path, dtype=dtype, mode="w+", shape=shape)
    return array, temp_path


def _save_streamed_nifti_result(
    *,
    array: np.ndarray,
    source_path: Path,
    affine: Any,
    header: Any,
    output_dir: Path,
    operation: str,
    output_prefix: str | None,
    output_format: Literal["auto", "tiff", "nifti"],
) -> tuple[Path, str, dict[str, str]]:
    if output_format == "tiff":
        output_path, artifact = ImageIO.save_tiff_stack_from_array(
            array=array,
            source_path=source_path,
            output_dir=output_dir,
            operation_code=operation,
            output_prefix=output_prefix,
            stack_axis=-1,
        )
        return output_path, "tiff", artifact
    output_path, artifact = ImageIO.save_nifti_from_array(
        array=array,
        source_path=source_path,
        affine=affine,
        header=header,
        output_dir=output_dir,
        operation_code=operation,
        output_prefix=output_prefix,
    )
    return output_path, "nifti", artifact


def _normalization_nifti(args: MinMaxNormalizationArgs, path: Path) -> dict[str, Any]:
    img: Any = nib.load(str(path))
    shape = tuple(int(dim) for dim in img.shape)
    output_dtype = _dtype_for_normalized(args.output_dtype)
    output, temp_path = _memmap_for_nifti(shape, output_dtype)
    output_stats = RunningStats()
    input_stats = RunningStats()
    per_slice_stats: list[dict[str, float]] = []
    constant_slices: list[int] = []
    max_workers = _bounded_worker_count(
        shape[_z_axis(shape)],
        max_default=MAX_MINMAX_NIFTI_WORKERS,
    )
    max_inflight = max(
        max_workers,
        min(shape[_z_axis(shape)], MAX_MINMAX_NIFTI_INFLIGHT_SLICES),
    )
    try:
        pending: set[Future[tuple[int, dict[str, float | int], dict[str, float | int], float, float, bool]]] = set()

        def consume_completed(
            done_futures: set[Future[tuple[int, dict[str, float | int], dict[str, float | int], float, float, bool]]],
        ) -> None:
            for future in done_futures:
                z_index, input_summary, output_summary, input_min, input_max, constant_slice = future.result()
                input_stats.merge_summary(
                    count=int(input_summary["count"]),
                    mean=float(input_summary["mean"]),
                    m2=float(input_summary["m2"]),
                    minimum=float(input_summary["min"]),
                    maximum=float(input_summary["max"]),
                )
                output_stats.merge_summary(
                    count=int(output_summary["count"]),
                    mean=float(output_summary["mean"]),
                    m2=float(output_summary["m2"]),
                    minimum=float(output_summary["min"]),
                    maximum=float(output_summary["max"]),
                )
                per_slice_stats.append({"z": z_index, "input_min": input_min, "input_max": input_max})
                if constant_slice:
                    constant_slices.append(z_index)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for z_index in range(shape[_z_axis(shape)]):
                pending.add(
                    executor.submit(
                        _normalize_nifti_slice_to_memmap_worker,
                        input_path=str(path),
                        output_memmap_path=str(temp_path),
                        shape=shape,
                        output_np_dtype_name=np.dtype(output_dtype).name,
                        z_index=z_index,
                        output_min=float(args.output_min),
                        output_max=float(args.output_max),
                        output_dtype=args.output_dtype,
                    )
                )
                if len(pending) >= max_inflight:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    consume_completed(done)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                consume_completed(done)

        per_slice_stats.sort(key=lambda item: int(item["z"]))
        constant_slices.sort()
        output.flush()

        output_path, saved_format, artifact = _save_streamed_nifti_result(
            array=output,
            source_path=path,
            affine=img.affine,
            header=img.header,
            output_dir=ImageIO.output_dir(args.session_path),
            operation="norm",
            output_prefix=args.output_prefix,
            output_format=args.output_format,
        )
    finally:
        try:
            del output
        finally:
            temp_path.unlink(missing_ok=True)

    return _success_payload(
        operation="min_max_normalization",
        input_path=path,
        output_path=output_path,
        output_format=saved_format,
        artifact=artifact,
        parameters={
            "input_min": input_stats.minimum,
            "input_max": input_stats.maximum,
            "normalization_scope": "per_z_slice",
            "z_axis": _z_axis(shape),
            "per_slice_input_ranges": per_slice_stats,
            "constant_slices": constant_slices,
            "output_min": float(args.output_min),
            "output_max": float(args.output_max),
            "output_dtype": args.output_dtype,
            "formula": "(x - slice_min) / (slice_max - slice_min) * (output_max - output_min) + output_min",
            "not_for_ground_truth_masks": True,
            "memory_mode": "bounded_parallel_streaming_nifti_per_slice",
        },
        measurements=output_stats.as_dict(),
        message="Min-max intensity normalization completed.",
    )


def _histogram_threshold_otsu(counts: np.ndarray, centers: np.ndarray) -> float:
    # scikit-image does not expose type information for this runtime API.
    return float(threshold_otsu(hist=(counts, centers)))  # type: ignore[no-untyped-call]


def _otsu_threshold_nifti(args: OtsuThresholdingArgs, path: Path) -> dict[str, Any]:
    img: Any = nib.load(str(path))
    dataobj = img.dataobj
    shape = tuple(int(dim) for dim in dataobj.shape)
    source_dtype = np.asanyarray(dataobj[tuple(slice(0, 1) for _ in shape)]).dtype
    output_dtype = np.dtype(np.uint8) if args.output_mode == "binary_mask" else source_dtype
    output, temp_path = _memmap_for_nifti(shape, output_dtype)
    output_stats = RunningStats()
    retained_voxels = 0
    total_voxels = int(np.prod(shape, dtype=np.int64))
    thresholds: list[dict[str, float | int]] = []
    constant_slices: list[int] = []
    max_workers = _bounded_worker_count(
        shape[_z_axis(shape)],
        max_default=MAX_OTSU_NIFTI_WORKERS,
    )
    max_inflight = max(max_workers, min(shape[_z_axis(shape)], MAX_OTSU_NIFTI_INFLIGHT_SLICES))
    try:
        pending: set[Future[tuple[int, dict[str, float | int], float, bool, int]]] = set()

        def consume_completed(
            done_futures: set[Future[tuple[int, dict[str, float | int], float, bool, int]]],
        ) -> None:
            nonlocal retained_voxels
            for future in done_futures:
                z_index, output_summary, threshold, constant_slice, retained = future.result()
                output_stats.merge_summary(
                    count=int(output_summary["count"]),
                    mean=float(output_summary["mean"]),
                    m2=float(output_summary["m2"]),
                    minimum=float(output_summary["min"]),
                    maximum=float(output_summary["max"]),
                )
                thresholds.append({"z": z_index, "threshold": threshold})
                if constant_slice:
                    constant_slices.append(z_index)
                retained_voxels += retained

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for z_index in range(shape[_z_axis(shape)]):
                pending.add(
                    executor.submit(
                        _otsu_nifti_slice_to_memmap_worker,
                        input_path=str(path),
                        output_memmap_path=str(temp_path),
                        shape=shape,
                        output_np_dtype_name=np.dtype(output_dtype).name,
                        z_index=z_index,
                        output_mode=args.output_mode,
                        foreground_value=int(args.foreground_value),
                        background_value=int(args.background_value),
                    )
                )
                if len(pending) >= max_inflight:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    consume_completed(done)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                consume_completed(done)

        thresholds.sort(key=lambda item: int(item["z"]))
        constant_slices.sort()
        threshold = float(thresholds[-1]["threshold"]) if thresholds else 0.0
        output.flush()

        output_path, saved_format, artifact = _save_streamed_nifti_result(
            array=output,
            source_path=path,
            affine=img.affine,
            header=img.header,
            output_dir=ImageIO.output_dir(args.session_path),
            operation="otsu-mask" if args.output_mode == "binary_mask" else "otsu",
            output_prefix=args.output_prefix,
            output_format=args.output_format,
        )
    finally:
        try:
            del output
        finally:
            temp_path.unlink(missing_ok=True)

    output_formula = (
        "output = foreground_value if input_value > threshold else background_value"
        if args.output_mode == "binary_mask"
        else "output = input_value if input_value > threshold else 0"
    )
    completion_message = (
        "Otsu thresholding completed. Output is a binary mask."
        if args.output_mode == "binary_mask"
        else "Otsu thresholding completed. Output is an intensity-preserving thresholded image, not a binary mask."
    )
    return _success_payload(
        operation="otsu_thresholding",
        input_path=path,
        output_path=output_path,
        output_format=saved_format,
        artifact=artifact,
        parameters={
            "threshold": threshold,
            "output_mode": args.output_mode,
            "retained_rule": "input_value > threshold",
            "suppressed_rule": "input_value <= threshold",
            "output_formula": output_formula,
            "binary_mask": args.output_mode == "binary_mask",
            "foreground_value": int(args.foreground_value),
            "background_value": int(args.background_value),
            "value_parameters_used": args.output_mode == "binary_mask",
            "threshold_scope": "per_z_slice",
            "z_axis": _z_axis(shape),
            "per_slice_thresholds": thresholds,
            "constant_slices": constant_slices,
            "memory_mode": "bounded_parallel_streaming_nifti_per_slice_otsu",
        },
        measurements={
            "otsu_threshold": threshold,
            "retained_voxels": retained_voxels,
            "foreground_voxels": retained_voxels,
            "total_voxels": total_voxels,
            "retained_fraction": retained_voxels / total_voxels if total_voxels else 0.0,
            "foreground_fraction": retained_voxels / total_voxels if total_voxels else 0.0,
            "bv_tv": retained_voxels / total_voxels if total_voxels else 0.0,
            "output": output_stats.as_dict(),
        },
        message=completion_message,
    )


def _gaussian_filter_nifti(args: GaussianFilteringArgs, path: Path) -> dict[str, Any]:
    img: Any = nib.load(str(path))
    dataobj = img.dataobj
    shape = tuple(int(dim) for dim in dataobj.shape)
    if len(shape) != 3:
        return _fail(
            "Streaming Gaussian filtering currently supports 3D NIfTI volumes.",
            error_type="unsupported_streaming_shape",
            recoverable=True,
            shape=list(shape),
        )

    source_dtype = np.asanyarray(dataobj[tuple(slice(0, 1) for _ in shape)]).dtype
    output_dtype = np.dtype(np.float32) if args.output_dtype == "float32" else source_dtype
    output, output_temp_path = _memmap_for_nifti(shape, output_dtype)
    input_stats = RunningStats()
    output_stats = RunningStats()
    max_workers = _bounded_worker_count(
        shape[_z_axis(shape)],
        max_default=MAX_GAUSSIAN_NIFTI_WORKERS,
    )
    max_inflight = max(
        max_workers,
        min(shape[_z_axis(shape)], MAX_GAUSSIAN_NIFTI_INFLIGHT_SLICES),
    )

    try:
        pending: set[Future[tuple[int, dict[str, float | int], dict[str, float | int]]]] = set()

        def consume_completed(
            done_futures: set[Future[tuple[int, dict[str, float | int], dict[str, float | int]]]],
        ) -> None:
            for future in done_futures:
                _z_index, input_summary, output_summary = future.result()
                input_stats.merge_summary(
                    count=int(input_summary["count"]),
                    mean=float(input_summary["mean"]),
                    m2=float(input_summary["m2"]),
                    minimum=float(input_summary["min"]),
                    maximum=float(input_summary["max"]),
                )
                output_stats.merge_summary(
                    count=int(output_summary["count"]),
                    mean=float(output_summary["mean"]),
                    m2=float(output_summary["m2"]),
                    minimum=float(output_summary["min"]),
                    maximum=float(output_summary["max"]),
                )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for z_index in range(shape[_z_axis(shape)]):
                pending.add(
                    executor.submit(
                        _gaussian_nifti_slice_to_memmap_worker,
                        input_path=str(path),
                        output_memmap_path=str(output_temp_path),
                        shape=shape,
                        output_np_dtype_name=np.dtype(output_dtype).name,
                        z_index=z_index,
                        sigma=float(args.sigma),
                        source_dtype_name=np.dtype(source_dtype).name,
                        output_dtype=args.output_dtype,
                    )
                )
                if len(pending) >= max_inflight:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    consume_completed(done)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                consume_completed(done)

        output.flush()

        output_path, saved_format, artifact = _save_streamed_nifti_result(
            array=output,
            source_path=path,
            affine=img.affine,
            header=img.header,
            output_dir=ImageIO.output_dir(args.session_path),
            operation=_gaussian_operation_code(float(args.sigma)),
            output_prefix=args.output_prefix,
            output_format=args.output_format,
        )
    finally:
        try:
            del output
        finally:
            output_temp_path.unlink(missing_ok=True)

    return _success_payload(
        operation="gaussian_filtering",
        input_path=path,
        output_path=output_path,
        output_format=saved_format,
        artifact=artifact,
        parameters={
            "sigma": float(args.sigma),
            "output_dtype": args.output_dtype,
            "filter_scope": "per_z_slice",
            "z_axis": _z_axis(shape),
            "memory_mode": "bounded_parallel_streaming_nifti_per_slice_2d_filter",
        },
        measurements={
            "input": input_stats.as_dict(),
            "output": output_stats.as_dict(),
        },
        message="Gaussian filtering completed.",
    )


def _success_payload(
    *,
    operation: str,
    input_path: Path,
    output_path: Path,
    output_format: str,
    artifact: dict[str, str],
    parameters: dict[str, Any],
    measurements: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    return {
        "success": True,
        "tool_kind": TOOL_KIND,
        "error": None,
        "attachments": [
            {
                "path": str(output_path),
                "kind": operation,
                "artifact_id": artifact["artifact_id"],
                "sample_id": artifact["sample_id"],
                "operation_code": artifact["operation_code"],
                "description": f"Image produced by {operation.replace('_', ' ')}.",
                "parent_arg": "image_path",
            }
        ],
        "operation": operation,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "artifact_id": artifact["artifact_id"],
        "sample_id": artifact["sample_id"],
        "operation_code": artifact["operation_code"],
        "output_format": output_format,
        "parameters": parameters,
        "measurements": measurements,
        "message": message,
    }


class MinMaxIntensityNormalizationTool(BaseTool):
    name: str = "min_max_intensity_normalization"
    description: str = (
        "Preprocesses a grayscale image or volume by min-max intensity normalization. "
        "This is linear scaling: the input minimum is remapped to output_min, the input "
        "maximum is remapped to output_max, and every other value is scaled linearly "
        "between those endpoints. This operation is NOT intended for ground truth masks, "
        "segmentation masks, or categorical label images because it would alter label "
        "values. Supports TIFF files, TIFF slice folders, and NIfTI volumes, and saves "
        "the normalized image as an artifact."
    )
    args_schema: type[BaseModel] = MinMaxNormalizationArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        try:
            args = MinMaxNormalizationArgs(**kwargs)
            path = ImageIO.resolve(args.image_path, args.session_path, args.workspace_root)
            if _is_nifti_path(path):
                return _normalization_nifti(args, path)
            loaded = ImageIO.load(path)
            data = np.asarray(loaded.data)
            input_min = float(np.min(data))
            input_max = float(np.max(data))
            if data.ndim >= 3:
                output, per_slice_ranges, constant_slices = _normalize_array_per_slice(
                    data,
                    output_min=float(args.output_min),
                    output_max=float(args.output_max),
                    output_dtype=args.output_dtype,
                    slice_axis=_slice_axis_for_array(data, loaded),
                )
                normalization_scope = "per_z_slice"
            else:
                if input_max == input_min:
                    return _fail(
                        "Cannot perform min-max normalization on a constant image because input min equals input max."
                    )
                data_float = data.astype(np.float64, copy=False)
                scaled = (data_float - input_min) / (input_max - input_min)
                scaled = scaled * (args.output_max - args.output_min) + args.output_min
                output = _cast_normalized(scaled, args.output_dtype)
                per_slice_ranges = []
                constant_slices = []
                normalization_scope = "single_image"

            output_path, saved_format, artifact = ImageIO.save(
                array=output,
                loaded=loaded,
                output_dir=ImageIO.output_dir(args.session_path),
                operation_code="norm",
                output_prefix=args.output_prefix,
                output_format=args.output_format,
            )
            return _success_payload(
                operation="min_max_normalization",
                input_path=path,
                output_path=output_path,
                output_format=saved_format,
                artifact=artifact,
                parameters={
                    "input_min": input_min,
                    "input_max": input_max,
                    "normalization_scope": normalization_scope,
                    "per_slice_input_ranges": per_slice_ranges,
                    "constant_slices": constant_slices,
                    "output_min": float(args.output_min),
                    "output_max": float(args.output_max),
                    "output_dtype": args.output_dtype,
                    "formula": "(x - input_min) / (input_max - input_min) * (output_max - output_min) + output_min",
                    "not_for_ground_truth_masks": True,
                },
                measurements=_safe_stats(output),
                message="Min-max intensity normalization completed.",
            )
        except ImagePathResolutionError as exc:
            return _image_path_resolution_failure(exc)
        except Exception as exc:
            logger.exception("Min-max normalization failed")
            return _fail(f"{type(exc).__name__}: {exc}")

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


class GaussianFilteringTool(BaseTool):
    name: str = "gaussian_filter_image"
    description: str = (
        "Preprocesses a grayscale TIFF file, TIFF slice folder, or NIfTI volume with "
        "Gaussian smoothing. The sigma parameter is adjustable and is passed directly "
        "to scipy.ndimage.gaussian_filter. Use this for intensity-image smoothing or "
        "denoising, not for categorical masks unless the user explicitly asks for "
        "non-binary soft filtering."
    )
    args_schema: type[BaseModel] = GaussianFilteringArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        try:
            args = GaussianFilteringArgs(**kwargs)
            path = ImageIO.resolve(args.image_path, args.session_path, args.workspace_root)
            if _is_nifti_path(path):
                return _gaussian_filter_nifti(args, path)
            loaded = ImageIO.load(path)
            data = np.asarray(loaded.data)
            if data.ndim >= 3:
                output = _gaussian_array_per_slice(
                    data,
                    sigma=float(args.sigma),
                    source_dtype=data.dtype,
                    output_dtype=args.output_dtype,
                    slice_axis=_slice_axis_for_array(data, loaded),
                )
                filter_scope = "per_z_slice"
            else:
                filtered = gaussian_filter(data.astype(np.float32, copy=False), sigma=float(args.sigma))
                output = _cast_filtered(filtered, data.dtype, args.output_dtype)
                filter_scope = "single_image"

            output_path, saved_format, artifact = ImageIO.save(
                array=output,
                loaded=loaded,
                output_dir=ImageIO.output_dir(args.session_path),
                operation_code=_gaussian_operation_code(float(args.sigma)),
                output_prefix=args.output_prefix,
                output_format=args.output_format,
            )
            return _success_payload(
                operation="gaussian_filtering",
                input_path=path,
                output_path=output_path,
                output_format=saved_format,
                artifact=artifact,
                parameters={
                    "sigma": float(args.sigma),
                    "output_dtype": args.output_dtype,
                    "filter_scope": filter_scope,
                },
                measurements={
                    "input": _safe_stats(data),
                    "output": _safe_stats(output),
                },
                message="Gaussian filtering completed.",
            )
        except ImagePathResolutionError as exc:
            return _image_path_resolution_failure(exc)
        except Exception as exc:
            logger.exception("Gaussian filtering failed")
            return _fail(f"{type(exc).__name__}: {exc}")

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


class OtsuThresholdingTool(BaseTool):
    name: str = "otsu_threshold_image"
    description: str = (
        "Preprocesses a grayscale TIFF file, TIFF slice folder, or NIfTI volume by "
        "computing a global Otsu threshold. By default output_mode='masked_image' "
        "saves an intensity-preserving thresholded image: pixels/voxels with "
        "original intensity at or below the threshold are set to 0, while values "
        "above the threshold keep their original input intensity. Use "
        "output_mode='binary_mask' to save only the binary Otsu mask with "
        "foreground_value/background_value. Use this for grayscale intensity "
        "images, not for evaluating or modifying ground truth masks."
    )
    args_schema: type[BaseModel] = OtsuThresholdingArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        try:
            args = OtsuThresholdingArgs(**kwargs)
            path = ImageIO.resolve(args.image_path, args.session_path, args.workspace_root)
            if _is_nifti_path(path):
                return _otsu_threshold_nifti(args, path)
            loaded = ImageIO.load(path)
            data = np.asarray(loaded.data)
            if data.ndim >= 3:
                output, thresholds, constant_slices, retained_voxels = _otsu_array_per_slice(
                    data,
                    output_mode=args.output_mode,
                    foreground_value=int(args.foreground_value),
                    background_value=int(args.background_value),
                    slice_axis=_slice_axis_for_array(data, loaded),
                )
                threshold = None
                threshold_scope = "per_z_slice"
            else:
                # scikit-image does not expose type information for this runtime API.
                threshold = float(threshold_otsu(data.reshape(-1)))  # type: ignore[no-untyped-call]
                threshold_mask = data > threshold
                retained_voxels = int(np.count_nonzero(threshold_mask))
                thresholds = [{"slice": 0, "threshold": threshold}]
                constant_slices = []
                threshold_scope = "single_image"
                if args.output_mode == "binary_mask":
                    output = np.where(
                        threshold_mask,
                        int(args.foreground_value),
                        int(args.background_value),
                    ).astype(np.uint8)
                else:
                    output = np.where(threshold_mask, data, 0).astype(data.dtype, copy=False)

            output_formula = (
                "output = foreground_value if input_value > threshold else background_value"
                if args.output_mode == "binary_mask"
                else "output = input_value if input_value > threshold else 0"
            )
            completion_message = (
                "Otsu thresholding completed. Output is a binary mask."
                if args.output_mode == "binary_mask"
                else "Otsu thresholding completed. Output is an intensity-preserving thresholded image, not a binary mask."
            )
            total_voxels = int(output.size)

            output_path, saved_format, artifact = ImageIO.save(
                array=output,
                loaded=loaded,
                output_dir=ImageIO.output_dir(args.session_path),
                operation_code="otsu-mask" if args.output_mode == "binary_mask" else "otsu",
                output_prefix=args.output_prefix,
                output_format=args.output_format,
            )
            return _success_payload(
                operation="otsu_thresholding",
                input_path=path,
                output_path=output_path,
                output_format=saved_format,
                artifact=artifact,
                parameters={
                    "threshold": threshold,
                    "output_mode": args.output_mode,
                    "threshold_scope": threshold_scope,
                    "per_slice_thresholds": thresholds,
                    "constant_slices": constant_slices,
                    "retained_rule": "input_value > threshold",
                    "suppressed_rule": "input_value <= threshold",
                    "output_formula": output_formula,
                    "binary_mask": args.output_mode == "binary_mask",
                    "foreground_value": int(args.foreground_value),
                    "background_value": int(args.background_value),
                    "value_parameters_used": args.output_mode == "binary_mask",
                },
                measurements={
                    "otsu_threshold": threshold,
                    "retained_voxels": retained_voxels,
                    "foreground_voxels": retained_voxels,
                    "total_voxels": total_voxels,
                    "retained_fraction": retained_voxels / total_voxels if total_voxels else 0.0,
                    "foreground_fraction": retained_voxels / total_voxels if total_voxels else 0.0,
                    "bv_tv": retained_voxels / total_voxels if total_voxels else 0.0,
                    "output": _safe_stats(output),
                },
                message=completion_message,
            )
        except ImagePathResolutionError as exc:
            return _image_path_resolution_failure(exc)
        except Exception as exc:
            logger.exception("Otsu thresholding failed")
            return _fail(f"{type(exc).__name__}: {exc}")

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


MinMaxNormalizationArgs.model_rebuild(
    _types_namespace={
        "Literal": Literal,
        "Field": Field,
    }
)
GaussianFilteringArgs.model_rebuild(
    _types_namespace={
        "Literal": Literal,
        "Field": Field,
    }
)
OtsuThresholdingArgs.model_rebuild(
    _types_namespace={
        "Literal": Literal,
        "Field": Field,
    }
)
MinMaxIntensityNormalizationTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "MinMaxNormalizationArgs": MinMaxNormalizationArgs,
    }
)
GaussianFilteringTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "GaussianFilteringArgs": GaussianFilteringArgs,
    }
)
OtsuThresholdingTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "OtsuThresholdingArgs": OtsuThresholdingArgs,
    }
)
LoadedImage.model_rebuild(
    _types_namespace={
        "Any": Any,
        "Literal": Literal,
        "Path": Path,
    }
)


EXPORTED_TOOLS: dict[str, BaseTool] = {
    "min_max_intensity_normalization": MinMaxIntensityNormalizationTool(),
    "gaussian_filter_image": GaussianFilteringTool(),
    "otsu_threshold_image": OtsuThresholdingTool(),
}
