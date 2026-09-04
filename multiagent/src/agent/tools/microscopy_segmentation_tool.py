from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from agent.tools.id_factory import new_id
from agent.tools.path_resolution import missing_input_path_message, resolve_input_path

logger = logging.getLogger(__name__)

TOOL_NAME = "segment_microscopy"
TOOL_KIND = "segment"

DESCRIPTION = (
    "Segment lacunae in microscopy images or volumes for downstream analysis workflows. "
    "Use this tool when you already have SR-microCT microscopy file and need a segmentation "
    "mask that identifies lacunae in the same spatial domain as the input. "
    "This tool works at image or volume level and produces binary segmentation outputs "
    "aligned with the original data for subsequent measurement, validation, or analysis."
)

BIOMED_AGENT_ROOT = Path(__file__).resolve().parents[3]
SEG_REPO_ROOT = (BIOMED_AGENT_ROOT / "lacunae_seg_tool-main").resolve()
DEFAULT_MODEL_PATH = str(SEG_REPO_ROOT / "model")
DEFAULT_MODEL_CONFIG = "config_predict.py"
if SEG_REPO_ROOT.exists() and str(SEG_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SEG_REPO_ROOT))


try:
    # These runtime modules belong to the excluded segmentation repository and ship no stubs.
    from utils import load_config  # type: ignore[import-not-found]
    from patching import PatchExtractor  # type: ignore[import-not-found]
    from model.unet import UNet  # type: ignore[import-not-found]
except Exception as exc:  # noqa: BLE001
    logger.warning("Unable to import eager segmentation modules: %s", exc)
    load_config = None
    PatchExtractor = None
    UNet = None


class MicroscopySegmentationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    image_path: str = Field(
        ...,
        description="Path to the microscopy image or volume to segment. Supported formats: .tif, .tiff, .nii, and .nii.gz.",
    )
    session_path: str | None = Field(
        default=None,
        description="Optional session directory. If provided, outputs are saved under '<session_path>/segmentation'.",
    )
    output_prefix: str | None = Field(
        default=None,
        description="Deprecated.",
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "Optional exact output path for the segmentation mask. If provided, the tool "
            "saves the mask at this path instead of generating a timestamped session output."
        ),
    )
    gpu_id: str | None = Field(
        default=None,
        description="Optional GPU identifier as string. If CUDA is unavailable, inference falls back to CPU.",
    )
    nifti_batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of slices per batch for NIfTI inference. If omitted, the default batch size is 100.",
    )
    nifti_max_workers: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum worker count for NIfTI slice loading and inference orchestration. If omitted, the tool uses an automatic value based on available CPUs.",
    )
    sample_id: str | None = Field(
        default=None,
        description=(
            "Optional stable sample identifier to return in compact outputs. If omitted, "
            "the tool infers it from the input filename."
        ),
    )
    result_mode: Literal["compact", "standard", "verbose"] = Field(
        default="compact",
        description=(
            "Controls the returned payload size. 'compact' returns only artifact identity "
            "needed by downstream agent steps; 'standard' adds output/session details; "
            "'verbose' returns the full legacy debug payload."
        ),
    )
    include_parameters: bool = Field(
        default=False,
        description="Include model/runtime parameters in non-verbose outputs. Defaults to false to reduce agent context.",
    )
    include_attachments: bool = Field(
        default=True,
        description=(
            "Include attachment metadata for produced mask registration. Keep true for LangGraph artifact tracking; "
            "set false only when an ultra-compact payload is required."
        ),
    )


def _strip_none_and_empty(obj: Any) -> Any:
    """Recursively remove None, empty lists, and empty dicts from JSON-like payloads."""
    if isinstance(obj, dict):
        clean: dict[str, Any] = {}
        for key, value in obj.items():
            if value is None or value == [] or value == {}:
                continue
            cleaned_value = _strip_none_and_empty(value)
            if cleaned_value is None or cleaned_value == [] or cleaned_value == {}:
                continue
            clean[key] = cleaned_value
        return clean
    if isinstance(obj, list):
        return [cleaned for item in obj if (cleaned := _strip_none_and_empty(item)) not in (None, [], {})]
    return obj


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON payload after removing optional empty values."""
    cleaned = _strip_none_and_empty(payload)
    if not isinstance(cleaned, dict):
        raise TypeError("A cleaned tool payload must remain a dictionary.")
    return cast(dict[str, Any], cleaned)


def _basename_without_suffixes(path: str) -> str:
    p = Path(path)
    if p.name.endswith(".nii.gz"):
        return p.name[:-7]
    return p.stem


def _infer_sample_id(image_path: str, output_prefix: str | None = None) -> str:
    path = Path(image_path)
    stem = _basename_without_suffixes(image_path)
    if "__" in stem:
        stem = stem.split("__", 1)[0]

    for suffix in ("_original", "_segmented", "_preprocessed", "_mask"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]

    for marker in ("_original_", "_segmented_", "_preprocessed_", "_gaussian_sigma_", "_otsu_threshold"):
        if marker in stem:
            return stem.split(marker, 1)[0]

    return stem or path.parent.name


def _safe_name_token(value: str, fallback: str = "sample") -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("._-")
    return token[:80].rstrip("._-") or fallback


def _generated_mask_path(input_path: str, output_dir: str, extension: str) -> tuple[str, dict[str, str]]:
    artifact_id = new_id("seg")
    short_id = artifact_id.rsplit("-", 1)[-1][-8:].lower()
    sample_id = _safe_name_token(_infer_sample_id(input_path), Path(input_path).parent.name or "sample")
    output_path = Path(output_dir) / f"{sample_id}__seg__{short_id}{extension}"
    return str(output_path), {
        "artifact_id": artifact_id,
        "sample_id": sample_id,
        "operation_code": "seg",
    }


def _mask_path_from_raw(raw: dict[str, Any]) -> str | None:
    outputs = raw.get("segmentation_outputs")
    if isinstance(outputs, dict):
        for key in ("mask_path", "nifti_mask_path"):
            value = outputs.get(key)
            if isinstance(value, str) and value.strip():
                return value

    attachments = raw.get("attachments")
    if isinstance(attachments, list):
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            path = attachment.get("path")
            if isinstance(path, str) and path.strip():
                return path

    return None


def _output_kind_from_path(mask_path: str | None) -> str | None:
    if not mask_path:
        return None
    lower = mask_path.lower()
    if lower.endswith((".tif", ".tiff")):
        return "tiff"
    if lower.endswith((".nii", ".nii.gz")):
        return "nifti"
    return None


def _error_payload(
    message: str,
    *,
    image_path: str | None = None,
    output_prefix: str | None = None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    result = {
        "success": False,
        "tool": TOOL_NAME,
        "sample_id": sample_id or (_infer_sample_id(image_path, output_prefix) if image_path else None),
        "input_path": image_path,
        "error": message,
    }
    return _clean_payload(result)


def _success_payload(
    *,
    input_kind: str,
    image_path: str,
    attachments: list[dict[str, Any]],
    segmentation_outputs: dict[str, str],
    session_path: str | None,
    output_dir: str,
    parameters: dict[str, Any],
    message: str,
    output_prefix: str | None,
    sample_id: str | None,
    result_mode: Literal["compact", "standard", "verbose"],
    include_parameters: bool,
    include_attachments: bool,
    artifact: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw = {
        "success": True,
        "tool_kind": TOOL_KIND,
        "error": None,
        "attachments": attachments,
        "input_kind": input_kind,
        "original_path": image_path,
        "session_path": session_path,
        "output_dir": output_dir,
        "segmentation_outputs": segmentation_outputs,
        "parameters": parameters,
        "message": message,
    }

    if result_mode == "verbose":
        return _clean_payload(raw)

    mask_path = _mask_path_from_raw(raw)
    compact = {
        "success": True,
        "tool": TOOL_NAME,
        "sample_id": sample_id or _infer_sample_id(image_path, output_prefix),
        "artifact_id": artifact.get("artifact_id") if artifact else None,
        "operation_code": artifact.get("operation_code") if artifact else None,
        "source_image_path": image_path,
        "mask_path": mask_path,
        "input_kind": input_kind,
        "output_kind": _output_kind_from_path(mask_path),
        "label_space": "binary",
        "message": message,
    }

    if include_attachments:
        compact["attachments"] = attachments
    if include_parameters:
        compact["parameters"] = parameters

    if result_mode == "compact":
        return _clean_payload(compact)

    standard = {
        **compact,
        "session_path": session_path,
        "output_dir": output_dir,
        "segmentation_outputs": segmentation_outputs,
    }
    return _clean_payload(standard)


class MicroscopySegmentationTool(BaseTool):
    name: str = TOOL_NAME
    description: str = DESCRIPTION
    args_schema: type[BaseModel] = MicroscopySegmentationArgs

    def _run(
        self,
        image_path: str,
        session_path: str | None = None,
        output_prefix: str | None = None,
        output_path: str | None = None,
        gpu_id: str | None = None,
        nifti_batch_size: int | None = None,
        nifti_max_workers: int | None = None,
        sample_id: str | None = None,
        result_mode: Literal["compact", "standard", "verbose"] = "compact",
        include_parameters: bool = False,
        include_attachments: bool = True,
    ) -> dict[str, Any]:
        resolved_image_path, searched_paths = self._resolve_image_path(
            image_path=image_path,
            session_path=session_path,
        )
        if resolved_image_path is None:
            return _error_payload(
                missing_input_path_message(
                    "Image file",
                    image_path,
                    searched_paths=searched_paths,
                )
            )
        image_path = resolved_image_path

        if Path(image_path).is_dir():
            directory_error = self._validate_nifti_input(image_path) or f"Input path is a directory: {image_path}."
            return _error_payload(directory_error)

        detected_type = self._detect_image_type(image_path=image_path)
        if detected_type is None:
            return _error_payload(
                f"Unsupported image format for segmentation: {image_path}. Supported formats are TIFF and NIfTI."
            )

        if detected_type == "tiff":
            tiff_check = self._validate_tiff_input(image_path)
            if tiff_check is not None:
                return _error_payload(tiff_check)
        elif detected_type == "nifti":
            nifti_check = self._validate_nifti_input(image_path)
            if nifti_check is not None:
                return _error_payload(nifti_check)

        if load_config is None or PatchExtractor is None or UNet is None:
            return _error_payload(
                f"Unable to import core modules from '{SEG_REPO_ROOT}'. Ensure the repository exists at the multiagent root and that its dependencies are installed."
            )

        resolved_output_path = self._resolve_output_path(
            output_path=output_path,
            input_path=image_path,
            session_path=session_path,
        )
        output_dir = (
            os.path.dirname(resolved_output_path)
            if resolved_output_path
            else self._resolve_output_dir(session_path)
        )
        os.makedirs(output_dir, exist_ok=True)

        generated_artifact: dict[str, str] | None = None
        if resolved_output_path is None:
            extension = ".tif" if detected_type == "tiff" else ".nii.gz"
            resolved_output_path, generated_artifact = _generated_mask_path(image_path, output_dir, extension)

        resolved_model_path = DEFAULT_MODEL_PATH
        resolved_model_config = DEFAULT_MODEL_CONFIG
        resolved_gpu_id = gpu_id or "0"

        try:
            if detected_type == "tiff":
                mask_path = self._segment_single_tiff(
                    image_path=image_path,
                    output_dir=output_dir,
                    output_basename=Path(resolved_output_path).stem,
                    output_path=resolved_output_path,
                    model_path=resolved_model_path,
                    model_config=resolved_model_config,
                    gpu_id=resolved_gpu_id,
                )
                attachments = [
                    {
                        "path": mask_path,
                        "kind": TOOL_KIND,
                        "role": "predicted_mask",
                        "semantic_target": "lacunae",
                        "file_format": "tiff",
                        "label_space": "binary",
                        "source_image_path": image_path,
                        **(generated_artifact or {}),
                        "description": "Binary lacunae segmentation mask in TIFF format aligned with the input slice.",
                        "parent_arg": "image_path",
                    }
                ]
                return _success_payload(
                    input_kind="tiff",
                    image_path=image_path,
                    attachments=attachments,
                    segmentation_outputs={"mask_path": mask_path},
                    session_path=session_path,
                    output_dir=output_dir,
                    parameters={
                        "model_path": resolved_model_path,
                        "model_config": resolved_model_config,
                        "gpu_id": resolved_gpu_id,
                        "output_path": resolved_output_path,
                    },
                    message="TIFF segmentation completed successfully.",
                    output_prefix=output_prefix,
                    sample_id=sample_id,
                    result_mode=result_mode,
                    include_parameters=include_parameters,
                    include_attachments=include_attachments,
                    artifact=generated_artifact,
                )

            batch_size = nifti_batch_size or 100
            max_workers = nifti_max_workers or max((os.cpu_count() or 2) - 1, 1)
            mask_path = self._segment_nifti_volume(
                input_nifti=image_path,
                output_dir=output_dir,
                prefix="seg_",
                timestamp="",
                model_path=resolved_model_path,
                model_config=resolved_model_config,
                gpu_id=resolved_gpu_id,
                batch_size=batch_size,
                max_workers=max_workers,
                output_path=resolved_output_path,
            )
            attachments = [
                {
                    "path": mask_path,
                    "kind": TOOL_KIND,
                    "role": "predicted_mask",
                    "semantic_target": "lacunae",
                    "file_format": "nifti",
                    "label_space": "binary",
                    "source_image_path": image_path,
                    **(generated_artifact or {}),
                    "description": "Binary lacunae segmentation mask volume in NIfTI format aligned with the input volume.",
                    "parent_arg": "image_path",
                }
            ]
            return _success_payload(
                input_kind="nifti",
                image_path=image_path,
                attachments=attachments,
                segmentation_outputs={"nifti_mask_path": mask_path},
                session_path=session_path,
                output_dir=output_dir,
                parameters={
                    "model_path": resolved_model_path,
                    "model_config": resolved_model_config,
                    "gpu_id": resolved_gpu_id,
                    "nifti_batch_size": batch_size,
                    "nifti_max_workers": max_workers,
                    "output_path": resolved_output_path,
                },
                message="NIfTI segmentation completed successfully.",
                output_prefix=output_prefix,
                sample_id=sample_id,
                result_mode=result_mode,
                include_parameters=include_parameters,
                include_attachments=include_attachments,
                artifact=generated_artifact,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Segmentation failed")
            return _error_payload(f"Segmentation error: {exc}")

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")

    @staticmethod
    def _resolve_output_dir(session_path: str | None) -> str:
        if session_path:
            return os.path.join(session_path, "segmentation")
        return os.path.join("statics", "output_microscopy", "segmentation")

    @staticmethod
    def _resolve_output_path(
        *,
        output_path: str | None,
        input_path: str,
        session_path: str | None,
    ) -> str | None:
        if not output_path:
            return None
        raw = Path(output_path).expanduser()
        if raw.is_absolute():
            return str(raw.resolve(strict=False))
        if session_path and output_path.startswith("data/"):
            return str((Path(session_path).expanduser() / raw).resolve(strict=False))
        return str((Path(input_path).parent / raw).resolve(strict=False))

    @staticmethod
    def _resolve_image_path(
        *,
        image_path: str,
        session_path: str | None,
    ) -> tuple[str | None, list[str]]:
        resolved = resolve_input_path(image_path, session_path=session_path)
        return (str(resolved.path) if resolved.path is not None else None, resolved.searched_paths)

    @staticmethod
    def _basename_without_suffixes(path: str) -> str:
        p = Path(path)
        if p.name.endswith(".nii.gz"):
            return p.name[:-7]
        return p.stem

    @staticmethod
    def _detect_image_type(image_path: str) -> str | None:
        lower = image_path.lower()
        if lower.endswith((".tif", ".tiff")):
            return "tiff"
        if lower.endswith(".nii") or lower.endswith(".nii.gz"):
            return "nifti"
        return None

    @staticmethod
    def _validate_tiff_input(image_path: str) -> str | None:
        try:
            with Image.open(image_path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    return "Multi-page TIFF inputs are not supported; provide a single 2D grayscale TIFF."
                array = np.asarray(image)
                if array.ndim != 2:
                    return "TIFF input must be a single-channel 2D image."
        except Exception as exc:  # noqa: BLE001
            return f"Failed to read TIFF input: {exc}"
        return None

    @staticmethod
    def _validate_nifti_input(image_path: str) -> str | None:
        path = Path(image_path)
        if path.is_dir():
            return (
                f"NIfTI input path is a directory, not a file: {image_path}. "
                "Provide the actual .nii or .nii.gz volume inside the sample directory, "
                "for example '<sample_id>_original.nii.gz'."
            )
        if not path.is_file():
            return f"NIfTI input path is not a file: {image_path}."
        if not (image_path.lower().endswith(".nii") or image_path.lower().endswith(".nii.gz")):
            return (
                f"NIfTI input path does not end with .nii or .nii.gz: {image_path}. "
                "Do not pass a sample directory as image_path."
            )
        return None

    def _segment_single_tiff(
        self,
        image_path: str,
        output_dir: str,
        output_basename: str,
        model_path: str,
        model_config: str,
        gpu_id: str,
        output_path: str | None,
    ) -> str:
        import random as _random

        import torch  # type: ignore[import-not-found]
        from skimage import filters, transform
        from torchvision import transforms  # type: ignore[import-not-found]
        from predict_service import seed_torch_for_inference, select_torch_device  # type: ignore

        configs = load_config(model_config)
        device = select_torch_device(torch, gpu_id)
        logger.info("Using PyTorch device for TIFF segmentation: %s", device)

        seed = 19
        seed_torch_for_inference(torch, seed)
        np.random.seed(seed)
        _random.seed(seed)
        torch.backends.cudnn.deterministic = True

        model = UNet(
            input_channels=configs.model_params["input_ch"],
            nclasses=configs.model_params["nr_classes"],
        )
        model.load_state_dict(
            torch.load(
                os.path.join(model_path, configs.model_params["model_name"] + ".h5"),
                map_location="cpu",
            )
        )
        model = model.to(device)
        model.eval()

        with Image.open(image_path) as image:
            img = np.asarray(image.convert("L"))

        nr_patches = configs.test_params["nr_patches"]
        height, width = img.shape[:2]
        patch_size = max(int(height / nr_patches), int(width / nr_patches))

        extractor_img = PatchExtractor(
            img=Image.fromarray(img),
            patch_size=patch_size,
            stride=patch_size,
        )
        patches_img = extractor_img.extract_img_patches()

        # scikit-image does not expose a typed signature for this runtime API.
        val = filters.threshold_otsu(img)  # type: ignore[no-untyped-call]
        seg_tissue = np.uint8((img > val) * 255)
        extractor_tiss = PatchExtractor(
            img=Image.fromarray(seg_tissue),
            patch_size=patch_size,
            stride=patch_size,
        )
        patches_tiss = extractor_tiss.extract_img_patches()

        seg_wsi = np.zeros((height, width), dtype=np.uint8)
        r, c = 0, 0
        img_dim_row = configs.test_params["img_dim_row"]
        img_dim_col = configs.test_params["img_dim_col"]
        probab_th = configs.test_params["probab_th"]

        for i, patch in enumerate(patches_img):
            tissue_patch = np.asarray(patches_tiss[i])
            tissue_ratio = np.count_nonzero(tissue_patch) / tissue_patch.size

            if tissue_ratio > 0:
                with torch.no_grad():
                    # scikit-image does not expose a typed signature for this runtime API.
                    img_patch_resized = transform.resize(  # type: ignore[no-untyped-call]
                        np.asarray(patch),
                        (img_dim_row, img_dim_col),
                    )
                    patch_tensor = transforms.ToTensor()(np.float32(img_patch_resized))
                    patch_tensor = torch.stack([patch_tensor, patch_tensor, patch_tensor], 1).to(device)

                    if torch.count_nonzero(patch_tensor) != 0:
                        masks_pred_prob = model(patch_tensor)
                        masks_pred = (masks_pred_prob > probab_th).float()
                        patch_pred = (np.asarray(masks_pred[0, 0, :, :].cpu()) * 255).astype(np.uint8)
                    else:
                        patch_pred = np.zeros((img_dim_row, img_dim_col), dtype=np.uint8)
            else:
                patch_pred = np.zeros((img_dim_row, img_dim_col), dtype=np.uint8)

            if (r < height) and (c < width):
                # scikit-image does not expose a typed signature for this runtime API.
                patch_resized = transform.resize(  # type: ignore[no-untyped-call]
                    patch_pred,
                    (patch_size, patch_size),
                    order=0,
                ).astype(np.uint8)
                end_r = min(r + patch_size, height)
                end_c = min(c + patch_size, width)
                seg_wsi[r:end_r, c:end_c] = patch_resized[: end_r - r, : end_c - c]

                c += patch_size
                if (width - c) < patch_size:
                    seg_wsi[r:end_r, c:width] = 0
                    c = width

            if (r < height) and (c == width):
                c = 0
                r += patch_size

        seg_wsi[seg_wsi > 0] = 1
        mask_path = output_path or os.path.join(output_dir, f"{output_basename}.tif")
        Image.fromarray(seg_wsi).save(mask_path)
        return mask_path

    def _segment_nifti_volume(
        self,
        input_nifti: str,
        output_dir: str,
        prefix: str,
        timestamp: str,
        model_path: str,
        model_config: str,
        gpu_id: str,
        batch_size: int,
        max_workers: int,
        output_path: str | None,
    ) -> str:
        import torch
        import nibabel as nib
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
        from tqdm import tqdm  # type: ignore[import-untyped]
        from predict_service import (
            assemble_volume_from_temp,
            load_slice_batch,
            process_slice_batch,
            seed_torch_for_inference,
            select_torch_device,
        )

        configs = load_config(model_config)
        device = select_torch_device(torch, gpu_id)
        logger.info("Using PyTorch device for NIfTI segmentation: %s", device)

        seed = 19
        seed_torch_for_inference(torch, seed)
        np.random.seed(seed)

        model = UNet(
            input_channels=configs.model_params["input_ch"],
            nclasses=configs.model_params["nr_classes"],
        )
        model.load_state_dict(
            torch.load(
                os.path.join(model_path, configs.model_params["model_name"] + ".h5"),
                map_location="cpu",
            )
        )
        model = model.to(device)
        model.eval()

        nii_img = cast(Any, nib.load(input_nifti))
        total_slices = int(nii_img.shape[2])
        temp_dir = tempfile.mkdtemp(prefix="lacunae_seg_")

        try:
            with tqdm(total=total_slices, desc="Processing slices") as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = set()
                    for start_idx in range(0, total_slices, batch_size):
                        slice_batch, slice_indices = load_slice_batch(nii_img, start_idx, batch_size)
                        individual_slices = [slice_batch[:, :, i] for i in range(slice_batch.shape[2])]
                        futures.add(
                            executor.submit(
                                process_slice_batch,
                                individual_slices,
                                slice_indices,
                                model,
                                device,
                                configs,
                                temp_dir,
                                pbar,
                            )
                        )

                        if len(futures) >= max_workers:
                            done, futures = wait(futures, return_when=FIRST_COMPLETED)
                            for future in done:
                                future.result()

                    for future in futures:
                        future.result()

            final_output_path = output_path or _generated_mask_path(input_nifti, output_dir, ".nii.gz")[0]
            assemble_volume_from_temp(temp_dir, final_output_path, nii_img, total_slices)
            return final_output_path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


MicroscopySegmentationArgs.model_rebuild(
    _types_namespace={
        "Literal": Literal,
        "Field": Field,
    }
)
MicroscopySegmentationTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "Literal": Literal,
        "MicroscopySegmentationArgs": MicroscopySegmentationArgs,
    }
)


_TOOL_INSTANCE = MicroscopySegmentationTool()
EXPORTED_TOOLS: dict[str, BaseTool] = {TOOL_NAME: _TOOL_INSTANCE}


def segment_microscopy_mcp(**kwargs: Any) -> dict[str, Any]:
    """Plain MCP-callable function with the same behavior as the BaseTool."""
    result = _TOOL_INSTANCE.invoke(kwargs)
    if not isinstance(result, dict):
        raise TypeError("Microscopy segmentation must return a dictionary payload.")
    return cast(dict[str, Any], result)


def register_mcp_tools(mcp_server: Any) -> Any:
    """Register the tool on an existing MCP server instance without creating or running a server here."""
    decorator = getattr(mcp_server, "tool", None)
    if decorator is None or not callable(decorator):
        raise TypeError("mcp_server must expose a callable .tool() decorator")
    decorator()(segment_microscopy_mcp)
    return mcp_server
