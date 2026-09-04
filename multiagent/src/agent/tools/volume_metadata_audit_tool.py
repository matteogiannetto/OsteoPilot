from __future__ import annotations

import logging
from typing import Any, Literal

# Domain libraries
import nibabel as nib
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.tools import BaseTool

from agent.tools.path_resolution import input_path_candidates

logger = logging.getLogger(__name__)

# ==========================================
# Input and output models (Pydantic schema)
# ==========================================

class VolumeAuditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume_path: str = Field(
        ...,
        description="Absolute or relative path to a NIfTI volume (.nii, .nii.gz) or a TIFF slice folder."
    )
    session_path: str = Field(
        ...,
        description="INTERNAL: Absolute path to the session directory, injected by the subgraph."
    )
    workspace_root: str | None = Field(
        default=None,
        description=(
            "INTERNAL: optional read-only base directory for resolving generic relative paths "
            "such as data/sample_folder."
        ),
    )
    operation: Literal[
        "geometry",
        "first_slice_intensity_audit",
    ] = Field(
        default="geometry",
        description=(
            "Audit operation to perform. Use 'geometry' for dimensions/slice count, "
            "or 'first_slice_intensity_audit' for dtype, first-slice dynamic range, "
            "and inferred bit depth."
        ),
    )

class VolumeAuditResult(BaseModel):
    success: bool
    error: str | None = None
    n_slices: int | None = None
    height_px: int | None = None
    width_px: int | None = None
    dynamic_range: float | None = None
    inferred_bit_depth: int | None = None
    actual_dtype: str | None = None
    message: str | None = None

# ==========================================
# Core logic (framework-agnostic)
# ==========================================

class VolumeAuditCore:
    """
    Framework-independent logic for auditing geometry metadata and lightweight statistics.
    """

    @staticmethod
    def _infer_bit_depth(minimum: float, maximum: float, dtype: np.dtype) -> int:
        dynamic_range = maximum - minimum
        if np.issubdtype(dtype, np.integer):
            bits = int(dtype.itemsize * 8)
            if bits <= 8:
                return 8
            if bits <= 16 and dynamic_range <= 4095:
                return 12
            return 16
        if dynamic_range <= 255:
            return 8
        if dynamic_range <= 4095:
            return 12
        return 16

    @staticmethod
    def _intensity_bounds(array: np.ndarray) -> tuple[float, float]:
        data = np.asarray(array)
        return float(np.min(data)), float(np.max(data))

    def execute(self, args: VolumeAuditArgs) -> VolumeAuditResult:
        try:
            candidates = input_path_candidates(
                args.volume_path,
                session_path=args.session_path,
                workspace_root=args.workspace_root,
            )
            path = next(
                (candidate for candidate in candidates if candidate.exists()),
                candidates[0],
            ).resolve()
            
            if not path.exists():
                return VolumeAuditResult(success=False, error=f"Path not found: {path}")

            # Case 1: directory containing a TIFF stack
            if path.is_dir():
                tiff_files = sorted([
                    f for f in path.iterdir() 
                    if f.is_file() and f.suffix.lower() in ['.tif', '.tiff']
                ])
                
                if not tiff_files:
                    return VolumeAuditResult(success=False, error=f"No TIFF files found in {path}")

                with Image.open(str(tiff_files[0])) as img:
                    width_px, height_px = img.size
                    first_slice = np.asarray(img)
                    actual_dtype = str(first_slice.dtype)

                if args.operation == "first_slice_intensity_audit":
                    minimum, maximum = self._intensity_bounds(first_slice)
                    dynamic_range = maximum - minimum
                    return VolumeAuditResult(
                        success=True,
                        dynamic_range=float(dynamic_range),
                        inferred_bit_depth=self._infer_bit_depth(
                            minimum,
                            maximum,
                            first_slice.dtype,
                        ),
                        actual_dtype=actual_dtype,
                        message="First-slice TIFF intensity audit completed.",
                    )

                return VolumeAuditResult(
                    success=True,
                    n_slices=len(tiff_files),
                    height_px=height_px,
                    width_px=width_px,
                    message=f"Detected a TIFF stack with {len(tiff_files)} slices."
                )

            # Case 2: single file
            elif path.is_file():
                ext = "".join(path.suffixes).lower() # Handles .nii.gz correctly
                
                # NIfTI file
                if '.nii' in ext:
                    nifti_image: Any = nib.load(str(path))
                    shape = nifti_image.shape
                    actual_dtype = str(nifti_image.get_data_dtype())
                    # NIfTI convention: (X, Y, Z, ...)
                    if len(shape) >= 3:
                        w, h, z = shape[:3]
                    else:
                        w, h = shape[:2]
                        z = 1

                    if args.operation == "first_slice_intensity_audit":
                        dataobj = nifti_image.dataobj
                        first_slice = np.asanyarray(dataobj[:, :, 0] if len(shape) >= 3 else dataobj)
                        minimum, maximum = self._intensity_bounds(first_slice)
                        dynamic_range = maximum - minimum
                        return VolumeAuditResult(
                            success=True,
                            dynamic_range=float(dynamic_range),
                            inferred_bit_depth=self._infer_bit_depth(
                                minimum,
                                maximum,
                                np.dtype(nifti_image.get_data_dtype()),
                            ),
                            actual_dtype=actual_dtype,
                            message="First-slice NIfTI intensity audit completed with nibabel.",
                        )

                    return VolumeAuditResult(
                        success=True,
                        n_slices=int(z),
                        height_px=int(h),
                        width_px=int(w),
                        message=f"Detected NIfTI volume: {shape}"
                    )
                
                # Single TIFF file, possibly multipage
                elif '.tif' in ext or '.tiff' in ext:
                    with Image.open(str(path)) as img:
                        w, h = img.size
                        z = getattr(img, "n_frames", 1)
                        actual_dtype = None
                        if args.operation == "first_slice_intensity_audit":
                            first_slice = np.asarray(img)
                            actual_dtype = str(first_slice.dtype)

                    if args.operation == "first_slice_intensity_audit":
                        minimum, maximum = self._intensity_bounds(first_slice)
                        dynamic_range = maximum - minimum
                        return VolumeAuditResult(
                            success=True,
                            dynamic_range=float(dynamic_range),
                            inferred_bit_depth=self._infer_bit_depth(
                                minimum,
                                maximum,
                                first_slice.dtype,
                            ),
                            actual_dtype=actual_dtype,
                            message="First-slice TIFF intensity audit completed.",
                        )

                    return VolumeAuditResult(
                        success=True,
                        n_slices=int(z),
                        height_px=int(h),
                        width_px=int(w),
                        message="Detected a single TIFF file, possibly multipage."
                    )

            return VolumeAuditResult(success=False, error="Unsupported file format.")

        except Exception as e:
            logger.error(f"Audit error: {e}")
            return VolumeAuditResult(success=False, error=f"{type(e).__name__}: {str(e)}")

# ==========================================
# LangChain adapter (subgraph interface)
# ==========================================

class VolumeMetadataAuditTool(BaseTool):
    name: str = "volume_metadata_audit"
    description: str = (
        "Audits TIFF folders, TIFF files, or NIfTI volumes. Supports operation='geometry' "
        "for structural shape (X, Y, Z dimensions), and "
        "operation='first_slice_intensity_audit' for first-slice dtype, dynamic range, "
        "and inferred 8/12/16-bit depth. Uses nibabel for NIfTI files and PIL/numpy for TIFF data."
    )
    args_schema: type[BaseModel] = VolumeAuditArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        # Argument loading and automatic injection are handled by the subgraph
        args = VolumeAuditArgs(**kwargs)
        result = VolumeAuditCore().execute(args)

        if not result.success:
            return {
                "success": False,
                "error": result.error
            }

        # This tool creates no files, so attachments remain empty or omitted.
        return {
            "success": True,
            "n_slices": result.n_slices,
            "height_px": result.height_px,
            "width_px": result.width_px,
            "dynamic_range": result.dynamic_range,
            "inferred_bit_depth": result.inferred_bit_depth,
            "actual_dtype": result.actual_dtype,
            "message": result.message
        }

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


VolumeAuditArgs.model_rebuild(
    _types_namespace={
        "Literal": Literal,
        "Field": Field,
    }
)
VolumeAuditResult.model_rebuild(
    _types_namespace={
        "Field": Field,
    }
)
VolumeMetadataAuditTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "VolumeAuditArgs": VolumeAuditArgs,
    }
)

# ==========================================
# Export Contract
# ==========================================

EXPORTED_TOOLS: dict[str, BaseTool] = {
    "volume_metadata_audit": VolumeMetadataAuditTool(),
}
