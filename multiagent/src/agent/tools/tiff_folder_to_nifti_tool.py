from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from langchain_core.tools import BaseTool
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from agent.tools.path_resolution import input_path_candidates

logger = logging.getLogger(__name__)

TOOL_NAME = "tiff_folder_to_nifti"
TOOL_KIND = "nifti_volume"


class TiffFolderToNiftiArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str = Field(
        ...,
        description=(
            "Folder containing TIFF slice files. May be absolute or relative to "
            "workspace_root/session_path."
        ),
    )
    session_path: str = Field(
        ...,
        description="INTERNAL: absolute path to the active session folder. Injected by the orchestrator.",
    )
    workspace_root: str | None = Field(
        default=None,
        description="INTERNAL: optional read-only base directory for generic relative paths.",
    )
    output_name: str | None = Field(
        default=None,
        description=(
            "Optional output filename. If omitted, the input folder name is used. "
            "The extension .nii.gz is added when missing."
        ),
    )
    recursive: bool = Field(
        default=False,
        description="Whether to collect TIFF files recursively below folder_path.",
    )
    sort_mode: str = Field(
        default="auto",
        description=(
            "Slice ordering mode: 'auto' uses numeric ordering when filenames contain "
            "numbers and lexicographic ordering otherwise; 'numeric' requires numbers; "
            "'lexicographic' sorts by path name."
        ),
    )


class TiffFolderToNiftiResult(BaseModel):
    success: bool
    error: str | None = None
    message: str | None = None
    output_path: str | None = None
    output_dir: str | None = None
    n_slices: int = 0
    height_px: int | None = None
    width_px: int | None = None
    dtype: str | None = None
    slice_indices: list[int] = Field(default_factory=list)
    source_files: list[str] = Field(default_factory=list)


def _error_payload(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "tool_kind": TOOL_KIND,
        "error": message,
        "attachments": [],
    }


def _resolve_folder(
    folder_path: str,
    session_path: str,
    workspace_root: str | None,
) -> Path:
    candidates = input_path_candidates(
        folder_path,
        session_path=session_path,
        workspace_root=workspace_root,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _numeric_key(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Filename does not contain a numeric slice index: {path.name}")
    return int(numbers[-1]), path.name.lower()


def _sort_tiff_files(files: list[Path], sort_mode: str) -> tuple[list[Path], list[int]]:
    normalized = sort_mode.strip().lower()
    if normalized not in {"auto", "numeric", "lexicographic"}:
        raise ValueError("sort_mode must be one of: auto, numeric, lexicographic")

    if normalized == "lexicographic":
        return sorted(files, key=lambda p: str(p).lower()), []

    numeric_pairs: list[tuple[int, Path]] = []
    missing_numbers: list[str] = []
    for path in files:
        numbers = re.findall(r"\d+", path.stem)
        if numbers:
            numeric_pairs.append((int(numbers[-1]), path))
        else:
            missing_numbers.append(path.name)

    if normalized == "numeric" and missing_numbers:
        raise ValueError(
            "numeric sort requested, but these files do not contain numeric indices: "
            + ", ".join(missing_numbers[:10])
        )

    if normalized == "auto" and missing_numbers:
        return sorted(files, key=lambda p: str(p).lower()), []

    seen_indices: dict[int, Path] = {}
    for index, path in numeric_pairs:
        if index in seen_indices:
            raise ValueError(
                f"Duplicate numeric slice index {index} in files "
                f"{seen_indices[index].name} and {path.name}"
            )
        seen_indices[index] = path

    numeric_pairs.sort(key=lambda item: (item[0], item[1].name.lower()))
    return [path for _, path in numeric_pairs], [idx for idx, _ in numeric_pairs]


def _read_tiff_slice(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(
                f"Multi-page TIFF files are not supported by this folder converter: {path}"
            )
        array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D TIFF slice, got shape {array.shape}: {path}")
    return array


class TiffFolderToNiftiCore:
    def execute(self, args: TiffFolderToNiftiArgs) -> TiffFolderToNiftiResult:
        try:
            folder = _resolve_folder(args.folder_path, args.session_path, args.workspace_root)
            if not folder.exists():
                return TiffFolderToNiftiResult(success=False, error=f"Folder not found: {folder}")
            if not folder.is_dir():
                return TiffFolderToNiftiResult(success=False, error=f"Path is not a folder: {folder}")

            pattern = "**/*" if args.recursive else "*"
            files = [
                path
                for path in folder.glob(pattern)
                if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
            ]
            if not files:
                return TiffFolderToNiftiResult(success=False, error=f"No TIFF files found in: {folder}")

            ordered_files, slice_indices = _sort_tiff_files(files, args.sort_mode)

            arrays: list[np.ndarray] = []
            first_shape: tuple[int, int] | None = None
            for path in ordered_files:
                array = _read_tiff_slice(path)
                if first_shape is None:
                    first_shape = tuple(array.shape)
                elif tuple(array.shape) != first_shape:
                    return TiffFolderToNiftiResult(
                        success=False,
                        error=(
                            f"TIFF slice shape mismatch. Expected {first_shape}, "
                            f"got {tuple(array.shape)} for {path}"
                        ),
                    )
                arrays.append(array)

            # PIL/numpy slices are Y,X. NIfTI shape is conventionally X,Y,Z, so transpose
            # each existing slice and stack only the files that are actually present.
            volume = np.stack([array.T for array in arrays], axis=-1)

            output_dir = Path(args.session_path).expanduser().resolve() / "artifacts" / "nifti_volumes"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_name = args.output_name or f"{folder.name}.nii.gz"
            if not output_name.endswith(".nii") and not output_name.endswith(".nii.gz"):
                output_name = f"{output_name}.nii.gz"
            output_path = output_dir / output_name

            # NiBabel's installed stubs omit this stable image-construction API.
            image: Any = nib.Nifti1Image(  # type: ignore[no-untyped-call]
                volume,
                affine=np.eye(4),
            )
            image.header.set_data_dtype(volume.dtype)
            nib.save(image, str(output_path))

            height_px, width_px = first_shape if first_shape else (None, None)
            return TiffFolderToNiftiResult(
                success=True,
                message=(
                    f"Converted {len(ordered_files)} existing TIFF slice file(s) to NIfTI. "
                    "No missing numeric slice positions were filled."
                ),
                output_path=str(output_path),
                output_dir=str(output_dir),
                n_slices=len(ordered_files),
                height_px=int(height_px) if height_px is not None else None,
                width_px=int(width_px) if width_px is not None else None,
                dtype=str(volume.dtype),
                slice_indices=slice_indices,
                source_files=[str(path) for path in ordered_files],
            )
        except Exception as exc:
            logger.exception("TIFF folder to NIfTI conversion failed")
            return TiffFolderToNiftiResult(success=False, error=f"{type(exc).__name__}: {exc}")


class TiffFolderToNiftiTool(BaseTool):
    name: str = TOOL_NAME
    description: str = (
        "Converts a folder of single-slice TIFF files into one NIfTI volume. "
        "Use this when a folder contains TIFF slices and downstream tools require a "
        ".nii or .nii.gz volume. If filenames contain numeric indices, those numbers "
        "are used to order the slices; missing numeric positions are not filled with "
        "synthetic black slices. The tool writes the NIfTI under the active session "
        "artifacts directory and returns an attachment spec. It does not preprocess, "
        "denoise, segment, resample, infer voxel size, or perform quantitative analysis."
    )
    args_schema: type[BaseModel] = TiffFolderToNiftiArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        args = TiffFolderToNiftiArgs(**kwargs)
        result = TiffFolderToNiftiCore().execute(args)

        if not result.success:
            return {
                "success": False,
                "tool_kind": TOOL_KIND,
                "error": result.error,
                "attachments": [],
            }

        assert result.output_path is not None
        return {
            "success": True,
            "tool_kind": TOOL_KIND,
            "error": None,
            "message": result.message,
            "output_dir": result.output_dir,
            "outputs": {
                "nifti_path": result.output_path,
                "n_slices": result.n_slices,
                "height_px": result.height_px,
                "width_px": result.width_px,
                "dtype": result.dtype,
                "slice_indices": result.slice_indices,
                "source_files": result.source_files,
            },
            "attachments": [
                {
                    "path": result.output_path,
                    "kind": TOOL_KIND,
                    "description": (
                        "NIfTI volume generated from existing TIFF files in the input folder. "
                        "No missing slice indices were filled."
                    ),
                    "parent_arg": "folder_path",
                }
            ],
        }

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


TiffFolderToNiftiArgs.model_rebuild(
    _types_namespace={
        "Field": Field,
    }
)
TiffFolderToNiftiResult.model_rebuild(
    _types_namespace={
        "Field": Field,
    }
)
TiffFolderToNiftiTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "TiffFolderToNiftiArgs": TiffFolderToNiftiArgs,
    }
)


def tiff_folder_to_nifti_mcp_call(
    folder_path: str,
    session_path: str,
    workspace_root: str | None = None,
    output_name: str | None = None,
    recursive: bool = False,
    sort_mode: str = "auto",
) -> dict[str, Any]:
    args = TiffFolderToNiftiArgs(
        folder_path=folder_path,
        session_path=session_path,
        workspace_root=workspace_root,
        output_name=output_name,
        recursive=recursive,
        sort_mode=sort_mode,
    )
    result = TiffFolderToNiftiCore().execute(args)
    return result.model_dump()


EXPORTED_TOOLS: dict[str, BaseTool] = {
    TOOL_NAME: TiffFolderToNiftiTool(),
}
