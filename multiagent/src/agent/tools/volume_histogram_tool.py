from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.tools.path_resolution import input_path_candidates

logger = logging.getLogger(__name__)


class VolumeHistogramArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume_path: str = Field(
        ...,
        description="Absolute path or path relative to workspace_root/session_path for a NIfTI volume, TIFF file, or TIFF slice folder.",
    )
    session_path: str = Field(
        ...,
        description="INTERNAL: absolute path to the active session folder. Injected by the subgraph.",
    )
    workspace_root: str | None = Field(
        default=None,
        description="INTERNAL: optional read-only base directory for resolving generic relative paths.",
    )
    n_bins: int = Field(
        default=10,
        ge=1,
        le=4096,
        description="Number of equally spaced histogram bins.",
    )
    range_min: float | None = Field(
        default=None,
        description="Optional lower bound for histogram binning. Leave null to use the data minimum.",
    )
    range_max: float | None = Field(
        default=None,
        description="Optional upper bound for histogram binning. Leave null to use the data maximum.",
    )
    normalize: bool = Field(
        default=True,
        description="If true, return relative_frequencies normalized so they sum to 1 over counted voxels.",
    )

    @model_validator(mode="after")
    def _validate_range(self) -> "VolumeHistogramArgs":
        if (self.range_min is None) != (self.range_max is None):
            raise ValueError("range_min and range_max must be provided together.")
        if self.range_min is not None and self.range_max is not None and self.range_min >= self.range_max:
            raise ValueError("range_min must be smaller than range_max.")
        return self


class VolumeHistogramCore:
    @staticmethod
    def _resolve_path(
        volume_path: str,
        session_path: str,
        workspace_root: str | None,
    ) -> Path:
        candidates = input_path_candidates(
            volume_path,
            session_path=session_path,
            workspace_root=workspace_root,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    @staticmethod
    def _load_array(path: Path) -> np.ndarray:
        if path.is_dir():
            tiff_files = sorted(
                f for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in {".tif", ".tiff"}
            )
            if not tiff_files:
                raise ValueError(f"No TIFF files found in {path}")
            slices = []
            for tiff_file in tiff_files:
                with Image.open(str(tiff_file)) as img:
                    slices.append(np.asarray(img))
            return np.stack(slices, axis=0)

        if not path.is_file():
            raise ValueError(f"Path is neither a file nor a folder: {path}")

        ext = "".join(path.suffixes).lower()
        if ".nii" in ext:
            nifti_image: Any = nib.load(str(path))
            return np.asanyarray(nifti_image.dataobj)

        if ".tif" in ext or ".tiff" in ext:
            with Image.open(str(path)) as img:
                n_frames = getattr(img, "n_frames", 1)
                if n_frames == 1:
                    return np.asarray(img)
                slices = []
                for frame_index in range(n_frames):
                    img.seek(frame_index)
                    slices.append(np.asarray(img))
                return np.stack(slices, axis=0)

        raise ValueError(f"Unsupported volume format: {path}")

    def execute(self, args: VolumeHistogramArgs) -> dict[str, Any]:
        try:
            path = self._resolve_path(args.volume_path, args.session_path, args.workspace_root)
            if not path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            data = self._load_array(path)
            histogram_range = None
            if args.range_min is not None and args.range_max is not None:
                histogram_range = (float(args.range_min), float(args.range_max))

            counts, bin_edges = np.histogram(data.reshape(-1), bins=args.n_bins, range=histogram_range)
            total = int(counts.sum())
            if args.normalize and total > 0:
                relative_frequencies = (counts.astype(np.float64) / float(total)).tolist()
            else:
                relative_frequencies = [float(value) for value in counts]

            return {
                "success": True,
                "error": None,
                "bin_edges": [float(value) for value in bin_edges],
                "counts": [int(value) for value in counts],
                "relative_frequencies": relative_frequencies,
                "n_bins": int(args.n_bins),
                "n_voxels_counted": total,
                "normalized": bool(args.normalize),
                "range": list(histogram_range) if histogram_range is not None else None,
                "actual_dtype": str(data.dtype),
                "message": f"Computed intensity histogram over {total} voxels.",
            }
        except Exception as exc:
            logger.exception("Volume histogram computation failed")
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


class VolumeIntensityHistogramTool(BaseTool):
    name: str = "volume_intensity_histogram"
    description: str = (
        "Computes an adjustable intensity histogram for a NIfTI volume, TIFF file, or TIFF slice folder. "
        "Parameters include n_bins, optional range_min/range_max, and normalize. Returns bin_edges, "
        "raw counts, relative_frequencies, dtype, and voxel count. Use this for data QA tasks that ask "
        "for equally spaced intensity bins or normalized histogram frequencies."
    )
    args_schema: type[BaseModel] = VolumeHistogramArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        args = VolumeHistogramArgs(**kwargs)
        return VolumeHistogramCore().execute(args)

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


VolumeHistogramArgs.model_rebuild(
    _types_namespace={
        "Field": Field,
    }
)
VolumeIntensityHistogramTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "VolumeHistogramArgs": VolumeHistogramArgs,
    }
)


EXPORTED_TOOLS: dict[str, BaseTool] = {
    "volume_intensity_histogram": VolumeIntensityHistogramTool(),
}
