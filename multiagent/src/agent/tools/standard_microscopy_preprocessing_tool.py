from typing import Any
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import os
import logging
import re
import tempfile
from pathlib import Path

# === Third‑party scientific stack ===
# NOTE: These are required at runtime.
import nibabel as nib
import numpy as np
from PIL import Image

from agent.tools.id_factory import new_id
from agent.tools.path_resolution import missing_input_path_message, resolve_input_path
from agent.tools.standard_microscopy_preprocessing_workers import (
    img_processing,
    _preprocess_tiff_worker,
    _preprocess_nifti_slice_worker,
)

logger = logging.getLogger(__name__)

NIFTI_EXTENSIONS = (".nii", ".nii.gz")
TIFF_EXTENSIONS = (".tif", ".tiff")
MAX_PREPROCESS_WORKERS = 10
MAX_PREPROCESS_INFLIGHT_SLICES = 10


def _safe_name_token(value: str, fallback: str = "sample") -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("._-")
    return token[:80].rstrip("._-") or fallback


def _preprocessed_output_path(tiff_path: str, output_dir: str) -> tuple[str, dict[str, str]]:
    source = Path(tiff_path)
    stem = source.stem
    if "__" in stem:
        stem = stem.split("__", 1)[0]
    for suffix in ("_original", "_preprocessed", "_mask"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    sample_id = _safe_name_token(stem, source.parent.name or "sample")
    artifact_id = new_id("img")
    short_id = artifact_id.rsplit("-", 1)[-1][-8:].lower()
    return str(Path(output_dir) / f"{sample_id}__prep__{short_id}.tiff"), {
        "artifact_id": artifact_id,
        "sample_id": sample_id,
        "operation_code": "prep",
    }


def _preprocessed_nifti_output_path(input_path: str, output_dir: str) -> tuple[str, dict[str, str]]:
    source = Path(input_path)
    name = source.name
    for suffix in NIFTI_EXTENSIONS:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    sample_id = _safe_name_token(name, source.parent.name or "sample")
    artifact_id = new_id("img")
    short_id = artifact_id.rsplit("-", 1)[-1][-8:].lower()
    return str(Path(output_dir) / f"{sample_id}__prep__{short_id}.nii.gz"), {
        "artifact_id": artifact_id,
        "sample_id": sample_id,
        "operation_code": "prep",
    }


def _preprocessed_folder_output_path(input_path: str, output_dir: str) -> tuple[str, dict[str, str]]:
    source = Path(input_path)
    sample_id = _safe_name_token(source.name, source.parent.name or "sample")
    artifact_id = new_id("img")
    short_id = artifact_id.rsplit("-", 1)[-1][-8:].lower()
    return str(Path(output_dir) / f"{sample_id}__prep__{short_id}"), {
        "artifact_id": artifact_id,
        "sample_id": sample_id,
        "operation_code": "prep",
    }


def _is_nifti_path(path: str | Path) -> bool:
    lowered = str(path).lower()
    return lowered.endswith(NIFTI_EXTENSIONS)


def _is_tiff_path(path: str | Path) -> bool:
    lowered = str(path).lower()
    return lowered.endswith(TIFF_EXTENSIONS)


def _bounded_worker_count(total_items: int, *, max_default: int = MAX_PREPROCESS_WORKERS) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(int(total_items), int(cpu_count), int(max_default)))


def _algorithm_parameters() -> dict[str, Any]:
    return {
        "normalization": "min-max to [0,255] (float32)",
        "imadjust": {"tol": 1, "vin": [0, 255], "vout": [0, 255]},
        "gaussian_sigma": 2,
        "branching": "Otsu if sum(img>=0) > sum(img<=0) else k-means (k=3)",
        "morphology": {"opening": "square(20)", "closing": "disk(25)"},
        "dtype_output": "uint8",
    }


def _fail(msg: str) -> dict[str, Any]:
    return {
        "success": False,
        "tool_kind": "preprocess",
        "error": msg,
        "attachments": [],
    }


# ARGUMENT SCHEMA (Pydantic v2)
class MicroscopyPreprocessingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tiff_path: str = Field(
        description=(
            "Path to a single-frame 2D grayscale TIFF, a folder of TIFF slices, or a NIfTI volume to preprocess."
        )
    )
    session_path: str | None = Field(
        default=None,
        description=(
            "Optional session directory. If provided, the output image is saved under "
            "'<session_path>/preprocessed/'."
        )
    )
    output_prefix: str | None = Field(
        default=None,
        description="Deprecated."
    )

# --------------------------------
# WRAPPER TOOL (agent-callable)
# --------------------------------
class MicroscopyPreprocessingTool(BaseTool):
    # --- Pydantic v2 compliant fields ---
    name: str = "preprocess_microscopy_tiff"
    description: str = (
        "Run the fixed SR-microCT microscopy preprocessing algorithm. Accepts a single "
        "2D grayscale TIFF, a folder of single-frame TIFF slices, or a NIfTI volume. "
        "Single TIFF inputs produce one TIFF, folder inputs produce a folder of TIFFs, "
        "and NIfTI inputs are processed one slice at a time and saved as a NIfTI volume."
    )
    args_schema: type[BaseModel] = MicroscopyPreprocessingArgs
  

    def __init__(self) -> None:
        super().__init__()
        logger.info("MicroscopyPreprocessingTool initialized (Algorithm Wrapper Mode)")

    def _output_dir(self, session_path: str | None) -> str:
        output_dir = os.path.join(session_path, "preprocessed") if session_path else "statics/output_microscopy/preprocessed"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _validate_single_tiff(self, tiff_path: str) -> dict[str, Any] | None:
        if not _is_tiff_path(tiff_path):
            return _fail(f"File is not a TIFF: {tiff_path}")

        try:
            with Image.open(tiff_path) as im:
                if getattr(im, "n_frames", 1) != 1:
                    return _fail("Multi-page TIFFs are not supported; provide a single-slice 2D image.")
                arr = np.asarray(im)
                if arr.ndim != 2:
                    return _fail("Image must be grayscale (2D). Provide a single-channel TIFF.")
        except Exception as e:
            return _fail(f"Failed to open TIFF: {e}")
        return None

    def _preprocess_single_tiff(self, tiff_path: str, output_dir: str) -> dict[str, Any]:
        validation_error = self._validate_single_tiff(tiff_path)
        if validation_error is not None:
            return validation_error

        output_path, artifact = _preprocessed_output_path(tiff_path, output_dir)

        try:
            proc = img_processing(tiff_path)
        except AssertionError as ae:
            return _fail(f"Algorithm assertion failed: {ae}. Ensure the image is 2D grayscale.")
        except Exception as e:
            logger.exception("Preprocessing failed")
            return _fail(f"Preprocessing error: {e}")

        try:
            Image.fromarray(proc).save(output_path)
        except Exception as e:
            return _fail(f"Failed to save processed image: {e}")

        attachments = [
            {
                "path": output_path,
                "kind": "preprocess",
                **artifact,
                "description": "Preprocessed SR-microCT slice produced by the preprocessing step.",
                "parent_arg": "tiff_path",
            }
        ]

        return {
            "success": True,
            "tool_kind": "preprocess",
            "input_kind": "single_tiff",
            "attachments": attachments,
            "original_path": tiff_path,
            "preprocessed_path": output_path,
            **artifact,
            "message": "Processing completed using the fixed algorithm; output saved.",
            "algorithm_parameters": _algorithm_parameters(),
        }

    def _preprocess_tiff_folder(self, folder_path: str, output_dir: str) -> dict[str, Any]:
        tiff_files = sorted(
            p for p in Path(folder_path).iterdir()
            if p.is_file() and _is_tiff_path(p)
        )
        if not tiff_files:
            return _fail(f"No TIFF files found in folder: {folder_path}")

        folder_output_path, folder_artifact = _preprocessed_folder_output_path(folder_path, output_dir)
        Path(folder_output_path).mkdir(parents=True, exist_ok=True)

        preprocessed_paths: list[str] = []
        attachments: list[dict[str, Any]] = [
            {
                "path": folder_output_path,
                "kind": "preprocess",
                **folder_artifact,
                "description": "Folder of preprocessed SR-microCT slices.",
                "parent_arg": "tiff_path",
            }
        ]

        planned_outputs: list[tuple[Path, str, dict[str, str]]] = []
        for tiff_file in tiff_files:
            output_path, artifact = _preprocessed_output_path(str(tiff_file), folder_output_path)
            planned_outputs.append((tiff_file, output_path, artifact))

        max_workers = _bounded_worker_count(len(planned_outputs))
        max_inflight = max(
            max_workers,
            min(len(planned_outputs), MAX_PREPROCESS_INFLIGHT_SLICES),
        )
        completed_by_index: dict[int, str] = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending: set[Future[tuple[str, str | None]]] = set()
            future_indices: dict[Future[tuple[str, str | None]], int] = {}

            def consume_completed(done_futures: set[Future[tuple[str, str | None]]]) -> dict[str, Any] | None:
                for future in done_futures:
                    index = future_indices.pop(future)
                    output_path, error = future.result()
                    if error is not None:
                        return _fail(f"Failed to preprocess {planned_outputs[index][0]}: {error}")
                    completed_by_index[index] = output_path
                return None

            for index, (tiff_file, output_path, _artifact) in enumerate(planned_outputs):
                future = executor.submit(_preprocess_tiff_worker, str(tiff_file), output_path)
                pending.add(future)
                future_indices[future] = index
                if len(pending) >= max_inflight:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    failure = consume_completed(done)
                    if failure is not None:
                        for future in pending:
                            future.cancel()
                        return failure

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                failure = consume_completed(done)
                if failure is not None:
                    for future in pending:
                        future.cancel()
                    return failure

        for index, (_tiff_file, output_path, artifact) in enumerate(planned_outputs):
            preprocessed_path = completed_by_index[index]
            preprocessed_paths.append(preprocessed_path)
            attachments.append(
                {
                    "path": preprocessed_path,
                    "kind": "preprocess",
                    **artifact,
                    "description": "Preprocessed SR-microCT slice produced by the preprocessing step.",
                    "parent_arg": "tiff_path",
                }
            )

        return {
            "success": True,
            "tool_kind": "preprocess",
            "input_kind": "tiff_folder",
            "memory_mode": "bounded_parallel_tiff_folder_per_slice",
            "attachments": attachments,
            "original_path": folder_path,
            "preprocessed_folder": folder_output_path,
            "preprocessed_paths": preprocessed_paths,
            "processed_slice_count": len(preprocessed_paths),
            **folder_artifact,
            "message": "Folder preprocessing completed using the fixed algorithm; outputs saved.",
            "algorithm_parameters": _algorithm_parameters(),
        }

    def _preprocess_array_via_temp_tiff(self, array: np.ndarray, temp_dir: str, name: str) -> np.ndarray:
        if array.ndim != 2:
            raise ValueError("Each NIfTI slice must be 2D.")
        temp_path = Path(temp_dir) / f"{_safe_name_token(name)}.tiff"
        Image.fromarray(np.asarray(array)).save(temp_path)
        return img_processing(str(temp_path))

    def _preprocess_nifti(self, nifti_path: str, output_dir: str) -> dict[str, Any]:
        try:
            # nibabel's type stub exposes the generic FileBasedImage base class,
            # which omits the NIfTI attributes used below.
            img: Any = nib.load(nifti_path)
        except Exception as e:
            return _fail(f"Failed to open NIfTI: {e}")

        if len(img.shape) not in (2, 3):
            return _fail(f"NIfTI volume must be 2D or 3D, got shape {img.shape}.")

        output_path, artifact = _preprocessed_nifti_output_path(nifti_path, output_dir)

        try:
            header = img.header.copy()
            header.set_data_dtype(np.uint8)
            with tempfile.TemporaryDirectory() as temp_dir:
                if len(img.shape) == 2:
                    slice_array = np.asarray(img.dataobj)
                    output_data = self._preprocess_array_via_temp_tiff(slice_array, temp_dir, "slice_0000")
                else:
                    memmap_path = Path(temp_dir) / "preprocessed_volume.dat"
                    output_data = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=img.shape)
                    max_workers = _bounded_worker_count(img.shape[-1])
                    max_inflight = max(
                        max_workers,
                        min(img.shape[-1], MAX_PREPROCESS_INFLIGHT_SLICES),
                    )
                    pending: set[Future[tuple[int, np.ndarray]]] = set()

                    def consume_completed(done_futures: set[Future[tuple[int, np.ndarray]]]) -> None:
                        for future in done_futures:
                            z_index, processed = future.result()
                            output_data[..., z_index] = processed
                            del processed

                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for z_index in range(img.shape[-1]):
                            slice_array = np.asarray(img.dataobj[..., z_index])
                            pending.add(
                                executor.submit(
                                    _preprocess_nifti_slice_worker,
                                    z_index,
                                    slice_array,
                                )
                            )
                            del slice_array
                            if len(pending) >= max_inflight:
                                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                                consume_completed(done)

                        while pending:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            consume_completed(done)

                    output_data.flush()

                # NiBabel's installed stubs omit this stable image-construction API.
                output_img: Any = nib.Nifti1Image(  # type: ignore[no-untyped-call]
                    output_data,
                    affine=img.affine,
                    header=header,
                )
                output_img.header.set_data_dtype(np.uint8)
                nib.save(output_img, output_path)
        except AssertionError as ae:
            return _fail(f"Algorithm assertion failed: {ae}. Ensure every NIfTI slice is 2D grayscale.")
        except Exception as e:
            logger.exception("NIfTI preprocessing failed")
            return _fail(f"NIfTI preprocessing error: {e}")

        attachments = [
            {
                "path": output_path,
                "kind": "preprocess",
                **artifact,
                "description": "Preprocessed SR-microCT NIfTI volume produced slice by slice.",
                "parent_arg": "tiff_path",
            }
        ]

        return {
            "success": True,
            "tool_kind": "preprocess",
            "input_kind": "nifti",
            "memory_mode": "streaming_nifti_per_slice",
            "attachments": attachments,
            "original_path": nifti_path,
            "preprocessed_path": output_path,
            "processed_slice_count": img.shape[-1] if len(img.shape) == 3 else 1,
            **artifact,
            "message": "NIfTI preprocessing completed using the fixed algorithm; output saved.",
            "algorithm_parameters": _algorithm_parameters(),
        }

    def _run(
        self,
        tiff_path: str,
        session_path: str | None = None,
        output_prefix: str | None = None,
    ) -> dict[str, Any]:
        """Run the unmodified preprocessing algorithm and save the result."""
        logger.info(f"Preparing to preprocess microscopy TIFF: {tiff_path}")

        # ---- 1) Input validation -------------------------------------------------
        resolved_tiff = resolve_input_path(tiff_path, session_path=session_path)
        if resolved_tiff.path is None:
            return _fail(
                missing_input_path_message(
                    "TIFF file",
                    tiff_path,
                    resolved_tiff.searched_paths,
                )
            )
        tiff_path = str(resolved_tiff.path)

        # ---- 2) Output path prep -------------------------------------------------
        output_dir = self._output_dir(session_path)

        if Path(tiff_path).is_dir():
            return self._preprocess_tiff_folder(tiff_path, output_dir)
        if _is_nifti_path(tiff_path):
            return self._preprocess_nifti(tiff_path, output_dir)
        if _is_tiff_path(tiff_path):
            return self._preprocess_single_tiff(tiff_path, output_dir)
        return _fail(f"Input is not a TIFF file, TIFF folder, or NIfTI volume: {tiff_path}")

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async not supported")


EXPORTED_TOOLS = {
    "preprocess_microscopy_tiff": MicroscopyPreprocessingTool(),
}


# =========================
# MCP integration helpers
# =========================

def preprocess_microscopy_tiff_mcp(
    tiff_path: str,
    session_path: str | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    """
    Preprocess a single 2D grayscale microscopy TIFF for downstream analysis workflows.

    Use this tool when you already have one microscopy slice and need to preprocess
    the image before later segmentation or quantitative image analysis. This tool
    works at single-image level and produces a preprocessed version of the same
    input image saved to disk for subsequent processing steps.
    """
    return MicroscopyPreprocessingTool()._run(
        tiff_path=tiff_path,
        session_path=session_path,
        output_prefix=output_prefix,
    )


def register_mcp_tools(mcp_server: Any) -> Any:
    """
    Register this module's MCP-facing tool functions on an existing MCP server instance.

    Example:
        from mcp.server.fastmcp import FastMCP
        from standard_microscopy_preprocessing_tool import register_mcp_tools

        mcp = FastMCP("my-server")
        register_mcp_tools(mcp)
    """
    mcp_server.tool()(preprocess_microscopy_tiff_mcp)
    return mcp_server
