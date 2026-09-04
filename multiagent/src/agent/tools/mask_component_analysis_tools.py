from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Literal, cast, overload

import nibabel as nib
import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import ndimage  # type: ignore[import-untyped]
from skimage.filters import threshold_otsu
import tifffile

from agent.tools.path_resolution import missing_input_path_message, resolve_input_path

logger = logging.getLogger(__name__)

TOOL_KIND = "quantitative_imaging_analysis"
SUPPORTED_FORMATS = ".nii, .nii.gz, .tif, .tiff, or a folder of TIFF slices"
MAX_INLINE_COMPONENT_LIST_ITEMS = 1000


class InputPathResolutionError(FileNotFoundError):
    """Expected, user-correctable input path resolution failure."""

    def __init__(
        self,
        *,
        input_path: str,
        searched_paths: list[str],
        session_path: str | None = None,
        workspace_root: str | None = None,
    ) -> None:
        self.input_path = input_path
        self.searched_paths = list(dict.fromkeys(str(path) for path in searched_paths))
        self.session_path = session_path
        self.workspace_root = workspace_root
        super().__init__(
            missing_input_path_message("Input path", input_path, self.searched_paths)
        )


def _fail(message: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "success": False,
        "tool_kind": TOOL_KIND,
        "error": message,
        "message": message,
        "attachments": [],
    }
    result.update(extra)
    return result


class LoadedArray(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Any
    path: Path
    source_kind: Literal["nifti", "tiff_file", "tiff_folder"]
    voxel_size_um: list[float] | None = None
    voxel_size_source: str | None = None


def _is_nifti(path: Path) -> bool:
    return path.name.lower().endswith((".nii", ".nii.gz"))


def _is_tiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _resolve_existing_path(
    input_path: str,
    *,
    session_path: str | None = None,
    workspace_root: str | None = None,
) -> Path:
    resolved = resolve_input_path(
        input_path,
        session_path=session_path,
        workspace_root=workspace_root,
    )
    if resolved.path is None:
        raise InputPathResolutionError(
            input_path=input_path,
            searched_paths=[str(path) for path in resolved.searched_paths],
            session_path=session_path,
            workspace_root=workspace_root,
        )
    return resolved.path


def _input_path_resolution_failure(exc: InputPathResolutionError) -> dict[str, Any]:
    searched_paths = list(dict.fromkeys(str(path) for path in exc.searched_paths))
    message = (
        "The input file or folder could not be found, so the quantitative imaging "
        "analysis was not executed. Use an existing session attachment path, an "
        "existing absolute path, or an existing path relative to the workspace root."
    )
    return _fail(
        message,
        error_type="input_path_not_found",
        input_path=exc.input_path,
        searched_paths=searched_paths,
        session_path=exc.session_path,
        workspace_root=exc.workspace_root,
        supported_formats=SUPPORTED_FORMATS,
        user_action=(
            "Verify the filename and directory, or call/list the session files first and "
            "retry with one of the registered attachment paths."
        ),
    )


def _normalize_voxel_size(
    voxel_size_um: list[float] | None,
    ndim: int,
) -> list[float] | None:
    if voxel_size_um is None:
        return None
    values = [float(v) for v in voxel_size_um]
    if len(values) != ndim:
        raise ValueError(
            f"voxel_size_um must contain exactly {ndim} values for a {ndim}D input."
        )
    if any(v <= 0 for v in values):
        raise ValueError("voxel_size_um values must be positive.")
    return values


def _nifti_zooms_to_um(img: nib.Nifti1Image, ndim: int) -> tuple[list[float] | None, str | None]:
    # NiBabel's installed stubs omit these stable header accessors.
    zooms = img.header.get_zooms()[:ndim]  # type: ignore[no-untyped-call]
    if len(zooms) != ndim or any(float(z) <= 0 for z in zooms):
        return None, None

    spatial_unit = img.header.get_xyzt_units()[0]  # type: ignore[no-untyped-call]
    if spatial_unit == "meter":
        scale = 1_000_000.0
    elif spatial_unit == "mm":
        scale = 1_000.0
    elif spatial_unit == "micron":
        scale = 1.0
    else:
        scale = 1.0
        spatial_unit = spatial_unit or "unknown"

    return [float(z) * scale for z in zooms], f"nifti_header_{spatial_unit}"


def _load_tiff_folder(path: Path) -> np.ndarray:
    tiff_files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if not tiff_files:
        raise ValueError(f"No TIFF slices found in folder: {path}")

    slices = [np.asarray(tifffile.imread(str(tiff_file))) for tiff_file in tiff_files]
    first_shape = slices[0].shape
    if any(slice_.shape != first_shape for slice_ in slices):
        raise ValueError("All TIFF slices in a folder must have the same shape.")
    return np.stack(slices, axis=0)


def _load_array(
    input_path: str,
    *,
    session_path: str | None = None,
    workspace_root: str | None = None,
    voxel_size_um: list[float] | None = None,
) -> LoadedArray:
    path = _resolve_existing_path(
        input_path,
        session_path=session_path,
        workspace_root=workspace_root,
    )

    if path.is_dir():
        data = _load_tiff_folder(path)
        size = _normalize_voxel_size(voxel_size_um, data.ndim)
        return LoadedArray(
            data=data,
            path=path,
            source_kind="tiff_folder",
            voxel_size_um=size,
            voxel_size_source="user_override" if size is not None else None,
        )

    if not path.is_file():
        raise ValueError(f"Path is neither a file nor a folder: {path}")

    if _is_nifti(path):
        img = cast(nib.Nifti1Image, nib.load(str(path)))
        data = np.asanyarray(img.dataobj)
        size = _normalize_voxel_size(voxel_size_um, data.ndim)
        source = "user_override" if size is not None else None
        if size is None:
            size, source = _nifti_zooms_to_um(img, data.ndim)
        return LoadedArray(data=data, path=path, source_kind="nifti", voxel_size_um=size, voxel_size_source=source)

    if _is_tiff(path):
        data = np.asarray(tifffile.imread(str(path)))
        size = _normalize_voxel_size(voxel_size_um, data.ndim)
        return LoadedArray(
            data=data,
            path=path,
            source_kind="tiff_file",
            voxel_size_um=size,
            voxel_size_source="user_override" if size is not None else None,
        )

    raise ValueError(f"Unsupported image format: {path}. Supported inputs are {SUPPORTED_FORMATS}.")


def _connectivity_structure(ndim: int, connectivity: int | None) -> tuple[np.ndarray, int]:
    if ndim == 2:
        value = 8 if connectivity is None else int(connectivity)
        if value == 4:
            return ndimage.generate_binary_structure(2, 1), value
        if value == 8:
            return ndimage.generate_binary_structure(2, 2), value
        raise ValueError("2D connectivity must be 4 or 8.")

    if ndim == 3:
        value = 26 if connectivity is None else int(connectivity)
        if value == 6:
            return ndimage.generate_binary_structure(3, 1), value
        if value == 18:
            return ndimage.generate_binary_structure(3, 2), value
        if value == 26:
            return ndimage.generate_binary_structure(3, 3), value
        raise ValueError("3D connectivity must be 6, 18, or 26.")

    raise ValueError(f"Connected-component analysis supports only 2D or 3D arrays; got {ndim}D.")


def _component_sizes(labels: np.ndarray, n_components: int) -> list[int]:
    if n_components <= 0:
        return []
    counts = np.bincount(labels.ravel())
    return [int(size) for size in counts[1 : n_components + 1]]


def _filter_sizes(
    sizes: list[int],
    *,
    min_size_voxels: int | None,
    max_size_voxels: int | None,
) -> list[int]:
    filtered = sizes
    if min_size_voxels is not None:
        filtered = [size for size in filtered if size >= int(min_size_voxels)]
    if max_size_voxels is not None:
        filtered = [size for size in filtered if size <= int(max_size_voxels)]
    return filtered


def _voxel_measure(voxel_size_um: list[float] | None) -> float | None:
    if voxel_size_um is None:
        return None
    return float(np.prod(np.asarray(voxel_size_um, dtype=float)))


def _stem_without_compound_suffix(path: Path) -> str:
    if path.is_dir():
        return path.name
    name = path.name
    for suffix in (".nii.gz", ".nii", ".tiff", ".tif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


@overload
def _strip_none_and_empty(obj: dict[str, Any]) -> dict[str, Any]: ...


@overload
def _strip_none_and_empty(obj: list[Any]) -> list[Any]: ...


@overload
def _strip_none_and_empty(obj: object) -> object: ...


def _strip_none_and_empty(obj: object) -> object:
    """Recursively remove null/empty fields from tool payloads before returning to the LLM."""
    if isinstance(obj, dict):
        clean: dict[str, Any] = {}
        for key, value in obj.items():
            if value is None or value == [] or value == {}:
                continue
            clean[key] = _strip_none_and_empty(value)
        return clean
    if isinstance(obj, list):
        return [_strip_none_and_empty(item) for item in obj]
    return obj


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove empty values while preserving the tool payload mapping contract."""
    cleaned = _strip_none_and_empty(payload)
    if not isinstance(cleaned, dict):
        raise TypeError("A cleaned tool payload must remain a dictionary.")
    return cast(dict[str, Any], cleaned)


def _infer_sample_id_from_path(input_path: str) -> str:
    """Best-effort generic sample id inference for microscopy artifacts."""
    path = Path(input_path)
    stem = _stem_without_compound_suffix(path)

    # Common input/artifact suffixes. These are generic artifact-name cleanups,
    # not assumptions about one biological workflow.
    for suffix in ("_original", "_mask", "_components"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    for marker in ("_segmented_", "_lacunae_components", "_crack_components", "_component_components"):
        if marker in stem:
            stem = stem.split(marker, 1)[0]

    return stem or path.parent.name


def _project_component_result(
    raw: dict[str, Any],
    *,
    component_domain: str,
    result_mode: Literal["compact", "standard", "verbose"],
    sample_id: str | None,
) -> dict[str, Any]:
    """Project a verbose connected-component result into an agent-sized payload."""
    if result_mode == "verbose" or not raw.get("ok"):
        return _clean_payload(raw)

    inferred_sample_id = sample_id or _infer_sample_id_from_path(str(raw.get("input_path") or ""))
    n_key = {
        "lacunae": "n_lacunae",
        "crack": "n_cracks",
        "component": "n_components",
    }.get(component_domain, "n_components")
    total_voxel_key = {
        "lacunae": "total_lacunae_volume_voxels",
        "crack": "total_crack_volume_voxels",
        "component": "total_component_volume_voxels",
    }.get(component_domain, "total_component_volume_voxels")
    total_um3_key = {
        "lacunae": "total_lacunae_volume_um3",
        "crack": "total_crack_volume_um3",
        "component": "total_component_volume_um3",
    }.get(component_domain, "total_component_volume_um3")
    total_um2_key = {
        "lacunae": "total_lacunae_area_um2",
        "crack": "total_crack_area_um2",
        "component": "total_component_area_um2",
    }.get(component_domain, "total_component_area_um2")

    n_value = raw.get(n_key, raw.get("n_components"))
    total_voxels = raw.get(total_voxel_key, raw.get("total_component_volume_voxels"))
    total_um3 = raw.get(total_um3_key, raw.get("total_component_volume_um3"))
    total_um2 = raw.get(total_um2_key, raw.get("total_component_area_um2"))

    compact: dict[str, Any] = {
        "ok": bool(raw.get("ok", raw.get("success", False))),
        "success": bool(raw.get("success", raw.get("ok", False))),
        "tool": f"count_{component_domain}" if component_domain != "component" else "count_connected_components",
        "sample_id": inferred_sample_id,
        n_key: n_value,
        "component_csv_path": raw.get("component_csv_path"),
        "voxel_size_um": raw.get("voxel_size_um"),
        "voxel_size_source": raw.get("voxel_size_source"),
        "warnings": raw.get("warnings"),
        # Keep attachments small so SubgraphAgent can still register CSV artifacts.
        "attachments": raw.get("attachments"),
    }

    if raw.get("dimensionality") == 3:
        compact[total_voxel_key] = total_voxels
        compact[total_um3_key] = total_um3
    elif raw.get("dimensionality") == 2:
        compact[f"total_{component_domain}_area_pixels" if component_domain != "component" else "total_component_area_pixels"] = total_voxels
        compact[total_um2_key] = total_um2

    if result_mode == "compact":
        return _clean_payload(compact)

    standard = dict(compact)
    standard.update(
        {
            "input_path": raw.get("input_path"),
            "dimensionality": raw.get("dimensionality"),
            "shape": raw.get("shape"),
            "class_value": raw.get("class_value", raw.get("foreground_value")),
            "connectivity": raw.get("connectivity"),
            "n_components_before_filtering": raw.get("n_components_before_filtering"),
            "n_removed_by_filtering": raw.get("n_removed_by_filtering"),
            "min_size_voxels": raw.get("min_size_voxels"),
            "max_size_voxels": raw.get("max_size_voxels"),
        }
    )
    return _clean_payload(standard)


def _is_inside_path(candidate: Path, root: Path) -> bool:
    candidate_abs = Path(candidate).expanduser().resolve()
    root_abs = Path(root).expanduser().resolve()
    try:
        return Path(candidate_abs).is_relative_to(root_abs)
    except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
        try:
            return str(candidate_abs).startswith(str(root_abs))
        except Exception:
            return False


def _resolve_component_csv_path(
    *,
    loaded_path: Path,
    component_domain: str,
    save_component_csv: bool,
    output_csv_path: str | None,
    session_path: str | None,
) -> Path | None:
    if not save_component_csv:
        return None

    session_root = Path(session_path).expanduser().resolve() if session_path else None
    output_root = session_root / "output" if session_root else loaded_path.parent

    if output_csv_path and output_csv_path.strip():
        raw = Path(output_csv_path.strip()).expanduser()
        if raw.suffix.lower() != ".csv":
            raise ValueError("output_csv_path must end with .csv.")
        resolved = raw if raw.is_absolute() else output_root / raw
    else:
        stem = _stem_without_compound_suffix(loaded_path)
        resolved = output_root / f"{stem}_{component_domain}_components.csv"

    if session_root is not None and _is_inside_path(resolved, session_root / "data"):
        raise ValueError("output_csv_path must not write inside the session data folder.")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.resolve()


def _write_component_csv(
    *,
    csv_path: Path,
    component_ids: list[int],
    sizes: list[int],
    volumes_um3: list[float] | None,
    areas_um2: list[float] | None,
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "component_id",
                "voxel_count",
                "volume_um3",
                "area_um2",
                "included_after_filtering",
            ],
        )
        writer.writeheader()
        for idx, component_id in enumerate(component_ids):
            writer.writerow(
                {
                    "component_id": component_id,
                    "voxel_count": sizes[idx],
                    "volume_um3": "" if volumes_um3 is None else volumes_um3[idx],
                    "area_um2": "" if areas_um2 is None else areas_um2[idx],
                    "included_after_filtering": True,
                }
            )


def _component_attachment(
    *,
    csv_path: Path | None,
    component_domain: str,
) -> list[dict[str, Any]]:
    if csv_path is None:
        return []
    label = {
        "lacunae": "Post-filtering per-lacuna voxel counts and physical measurements CSV.",
        "crack": "Post-filtering per-crack voxel counts and physical measurements CSV.",
        "component": "Post-filtering per-component voxel counts and physical measurements CSV.",
    }.get(component_domain, "Post-filtering per-component voxel counts and physical measurements CSV.")
    return [
        {
            "path": str(csv_path),
            "kind": "component_measurements_csv",
            "description": label,
            "parent_arg": "input_path",
        }
    ]


def _filtered_component_ids_and_sizes(
    sizes: list[int],
    *,
    min_size_voxels: int | None,
    max_size_voxels: int | None,
) -> tuple[list[int], list[int]]:
    component_ids: list[int] = []
    filtered_sizes: list[int] = []
    for component_id, size in enumerate(sizes, start=1):
        if min_size_voxels is not None and size < int(min_size_voxels):
            continue
        if max_size_voxels is not None and size > int(max_size_voxels):
            continue
        component_ids.append(component_id)
        filtered_sizes.append(size)
    return component_ids, filtered_sizes


class ConnectedComponentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(..., description=f"Path to a mask/image. Supported inputs: {SUPPORTED_FORMATS}.")
    foreground_value: float | None = Field(
        default=None,
        description="If provided, foreground is exactly pixels/voxels equal to this value. Otherwise foreground is nonzero.",
    )
    connectivity: int | None = Field(
        default=None,
        description="2D: 4 or 8. 3D: 6, 18, or 26. Defaults to 8 in 2D and 26 in 3D.",
    )
    min_size_voxels: int | None = Field(default=None, ge=1)
    max_size_voxels: int | None = Field(default=None, ge=1)
    include_component_sizes: bool = Field(
        default=False,
        description=(
            "Whether to include per-component size and physical-measurement lists directly in the JSON output. "
            "Default is false to keep tool payloads compact; use save_component_csv=True for per-component data. "
            "Large lists are suppressed even if requested."
        ),
    )
    voxel_size_um: list[float] | None = Field(
        default=None,
        description=(
            "Optional pixel/voxel side length in micrometers. For 2D provide [x, y]; "
            "for 3D provide [x, y, z]. This is not voxel volume. If a task gives "
            "voxel volume in mm^3, convert first: side_um = (voxel_volume_mm3 * 1e9) ** (1/3)."
        ),
    )
    save_component_csv: bool = Field(
        default=False,
        description=(
            "If true, write a post-filtering per-component CSV and register it as an attachment. "
            "When true, per-component lists are never returned in the JSON payload."
        ),
    )
    output_csv_path: str | None = Field(
        default=None,
        description="Optional CSV output path. Relative paths resolve under the active session output folder.",
    )
    sample_id: str | None = Field(
        default=None,
        description="Optional stable sample identifier to echo in compact/standard outputs.",
    )
    result_mode: Literal["compact", "standard", "verbose"] = Field(
        default="compact",
        description=(
            "Output detail level. compact is optimized for agent control flow; "
            "standard adds moderate audit fields; verbose returns the full debug payload."
        ),
    )
    session_path: str | None = Field(default=None, description="INTERNAL: optional active session directory.")
    workspace_root: str | None = Field(default=None, description="INTERNAL: optional workspace root.")

    @model_validator(mode="after")
    def _validate_size_bounds(self) -> "ConnectedComponentsArgs":
        if (
            self.min_size_voxels is not None
            and self.max_size_voxels is not None
            and self.min_size_voxels > self.max_size_voxels
        ):
            raise ValueError("min_size_voxels must be <= max_size_voxels.")
        return self


class BoneVolumeOtsuArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(..., description=f"Path to a preprocessed microscopy image/volume. Supported inputs: {SUPPORTED_FORMATS}.")
    include_zero_in_otsu: bool = Field(
        default=False,
        description="If false, compute Otsu threshold only on strictly positive voxels/pixels.",
    )
    voxel_size_um: list[float] | None = Field(
        default=None,
        description=(
            "Optional pixel/voxel side length in micrometers. For 2D provide [x, y]; "
            "for 3D provide [x, y, z]. This is not voxel volume. If a task gives "
            "voxel volume in mm^3, convert first: side_um = (voxel_volume_mm3 * 1e9) ** (1/3)."
        ),
    )
    session_path: str | None = Field(default=None, description="INTERNAL: optional active session directory.")
    workspace_root: str | None = Field(default=None, description="INTERNAL: optional workspace root.")


class CountLacunaeArgs(ConnectedComponentsArgs):
    include_component_sizes: bool = Field(
        default=False,
        description=(
            "Whether to include per-lacuna lists directly in the JSON output. "
            "Default is false to avoid large payloads. Prefer save_component_csv=True."
        ),
    )
    save_component_csv: bool = Field(
        default=True,
        description=(
            "Write post-filtering per-lacuna measurements to CSV and return only a compact "
            "JSON summary plus the CSV attachment path. When true, per-lacuna lists are never "
            "returned in the JSON payload."
        ),
    )
    foreground_value: float | None = Field(
        default=None,
        description="Optional alias for class_value. If omitted, class_value is used.",
    )
    class_value: float = Field(
        default=1,
        description="Lacunae mask class value. Defaults to class 1.",
    )


class CountCracksArgs(ConnectedComponentsArgs):
    include_component_sizes: bool = Field(
        default=False,
        description=(
            "Whether to include per-crack lists directly in the JSON output. "
            "Default is false to avoid large payloads. Prefer save_component_csv=True."
        ),
    )
    save_component_csv: bool = Field(
        default=True,
        description=(
            "Write post-filtering per-crack measurements to CSV and return only a compact "
            "JSON summary plus the CSV attachment path. When true, per-crack lists are never "
            "returned in the JSON payload."
        ),
    )
    foreground_value: float | None = Field(
        default=None,
        description="Optional alias for class_value. If omitted, class_value is used.",
    )
    class_value: float = Field(
        default=2,
        description="Crack mask class value. Defaults to class 2.",
    )


def _count_components_core(
    *,
    component_domain: str,
    input_path: str,
    foreground_value: float | None = None,
    connectivity: int | None = None,
    min_size_voxels: int | None = None,
    max_size_voxels: int | None = None,
    include_component_sizes: bool = False,
    voxel_size_um: list[float] | None = None,
    save_component_csv: bool = False,
    output_csv_path: str | None = None,
    sample_id: str | None = None,
    result_mode: Literal["compact", "standard", "verbose"] = "compact",
    session_path: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Count 2D/3D connected foreground components and compute post-filtering measures."""
    try:
        loaded = _load_array(
            input_path,
            session_path=session_path,
            workspace_root=workspace_root,
            voxel_size_um=voxel_size_um,
        )
        data = np.asarray(loaded.data)
        if data.ndim not in {2, 3}:
            raise ValueError(f"Connected-component analysis supports only 2D or 3D arrays; got shape {data.shape}.")

        foreground = data != 0 if foreground_value is None else data == foreground_value
        structure, normalized_connectivity = _connectivity_structure(data.ndim, connectivity)
        labels_output = np.empty(foreground.shape, dtype=np.int32)
        n_components = ndimage.label(
            foreground,
            structure=structure,
            output=labels_output,
        )
        labels = labels_output
        sizes = _component_sizes(labels, int(n_components))
        component_ids, filtered_sizes = _filtered_component_ids_and_sizes(
            sizes,
            min_size_voxels=min_size_voxels,
            max_size_voxels=max_size_voxels,
        )
        element_measure_um = _voxel_measure(loaded.voxel_size_um)
        filtered_total_voxels = int(sum(filtered_sizes))
        warnings: list[str] = []

        # Large per-component arrays must not enter the JSON payload when a CSV is produced.
        # Even without CSV output, refuse to inline very large lists to protect the agent context.
        requested_inline_lists = bool(include_component_sizes) and not bool(save_component_csv)
        return_component_lists = requested_inline_lists and len(filtered_sizes) <= MAX_INLINE_COMPONENT_LIST_ITEMS
        if requested_inline_lists and not return_component_lists:
            warnings.append(
                "Per-component lists were omitted from the JSON payload because they contain "
                f"{len(filtered_sizes)} items, exceeding MAX_INLINE_COMPONENT_LIST_ITEMS="
                f"{MAX_INLINE_COMPONENT_LIST_ITEMS}. Use save_component_csv=True to export them."
            )

        component_volumes_um3: list[float] | None = None
        total_component_volume_um3: float | None = None
        component_areas_um2: list[float] | None = None
        total_component_area_um2: float | None = None
        voxel_volume_um3: float | None = None
        pixel_area_um2: float | None = None

        if element_measure_um is None:
            warnings.append(
                "Physical component volumes or areas are null because voxel_size_um "
                "was not provided and could not be inferred from the input metadata."
            )
        elif data.ndim == 3:
            voxel_volume_um3 = element_measure_um
            all_component_volumes_um3 = [float(size * element_measure_um) for size in filtered_sizes]
            component_volumes_um3 = all_component_volumes_um3 if return_component_lists else None
            total_component_volume_um3 = float(filtered_total_voxels * element_measure_um)
        else:
            pixel_area_um2 = element_measure_um
            all_component_areas_um2 = [float(size * element_measure_um) for size in filtered_sizes]
            component_areas_um2 = all_component_areas_um2 if return_component_lists else None
            total_component_area_um2 = float(filtered_total_voxels * element_measure_um)

        csv_path = _resolve_component_csv_path(
            loaded_path=loaded.path,
            component_domain=component_domain,
            save_component_csv=save_component_csv,
            output_csv_path=output_csv_path,
            session_path=session_path,
        )
        if csv_path is not None:
            csv_volumes = (
                [float(size * element_measure_um) for size in filtered_sizes]
                if element_measure_um is not None and data.ndim == 3
                else None
            )
            csv_areas = (
                [float(size * element_measure_um) for size in filtered_sizes]
                if element_measure_um is not None and data.ndim == 2
                else None
            )
            _write_component_csv(
                csv_path=csv_path,
                component_ids=component_ids,
                sizes=filtered_sizes,
                volumes_um3=csv_volumes,
                areas_um2=csv_areas,
            )

        result: dict[str, Any] = {
            "ok": True,
            "success": True,
            "tool_kind": TOOL_KIND,
            "message": "Connected-component analysis completed successfully.",
            "input_path": str(loaded.path),
            "dimensionality": int(data.ndim),
            "shape": [int(dim) for dim in data.shape],
            "foreground_value": None if foreground_value is None else float(foreground_value),
            "connectivity": int(normalized_connectivity),
            "n_components": int(len(filtered_sizes)),
            "n_foreground_voxels": filtered_total_voxels,
            "component_sizes_voxels": filtered_sizes if return_component_lists else None,
            "component_ids": component_ids if return_component_lists else None,
            "min_size_voxels": None if min_size_voxels is None else int(min_size_voxels),
            "max_size_voxels": None if max_size_voxels is None else int(max_size_voxels),
            "n_components_after_filtering": int(len(filtered_sizes)),
            "component_sizes_after_filtering_voxels": filtered_sizes if return_component_lists else None,
            "n_components_before_filtering": int(n_components),
            "n_removed_by_filtering": int(n_components) - int(len(filtered_sizes)),
            "total_component_volume_voxels": filtered_total_voxels,
            "voxel_size_um": loaded.voxel_size_um,
            "voxel_size_source": loaded.voxel_size_source,
            "voxel_volume_um3": voxel_volume_um3,
            "pixel_area_um2": pixel_area_um2,
            "component_volumes_um3": component_volumes_um3,
            "total_component_volume_um3": total_component_volume_um3,
            "component_areas_um2": component_areas_um2,
            "total_component_area_um2": total_component_area_um2,
            "component_csv_path": None if csv_path is None else str(csv_path),
            "per_component_data_location": (
                "csv" if csv_path is not None else "json" if return_component_lists else "not_returned"
            ),
            "max_inline_component_list_items": MAX_INLINE_COMPONENT_LIST_ITEMS,
            "warnings": warnings,
            "attachments": _component_attachment(
                csv_path=csv_path,
                component_domain=component_domain,
            ),
        }
        return _project_component_result(
            result,
            component_domain=component_domain,
            result_mode=result_mode,
            sample_id=sample_id,
        )
    except InputPathResolutionError as exc:
        logger.warning(
            "Connected-component counting skipped because input path was not found: %s",
            exc.input_path,
        )
        return _input_path_resolution_failure(exc)
    except Exception as exc:
        logger.exception("Connected-component counting failed")
        return _fail(str(exc), error_type=exc.__class__.__name__)


def count_connected_components_mcp(
    input_path: str,
    foreground_value: float | None = None,
    connectivity: int | None = None,
    min_size_voxels: int | None = None,
    max_size_voxels: int | None = None,
    include_component_sizes: bool = False,
    voxel_size_um: list[float] | None = None,
    save_component_csv: bool = False,
    output_csv_path: str | None = None,
    sample_id: str | None = None,
    result_mode: Literal["compact", "standard", "verbose"] = "compact",
    session_path: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Count 2D/3D connected foreground components in an existing NIfTI or TIFF mask."""
    return _count_components_core(
        component_domain="component",
        input_path=input_path,
        foreground_value=foreground_value,
        connectivity=connectivity,
        min_size_voxels=min_size_voxels,
        max_size_voxels=max_size_voxels,
        include_component_sizes=include_component_sizes,
        voxel_size_um=voxel_size_um,
        save_component_csv=save_component_csv,
        output_csv_path=output_csv_path,
        sample_id=sample_id,
        result_mode=result_mode,
        session_path=session_path,
        workspace_root=workspace_root,
    )


def count_lacunae_mcp(
    input_path: str,
    foreground_value: float | None = None,
    class_value: float = 1,
    connectivity: int | None = None,
    min_size_voxels: int | None = None,
    max_size_voxels: int | None = None,
    include_component_sizes: bool = False,
    voxel_size_um: list[float] | None = None,
    save_component_csv: bool = True,
    output_csv_path: str | None = None,
    sample_id: str | None = None,
    result_mode: Literal["compact", "standard", "verbose"] = "compact",
    session_path: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Count lacunae as connected components in a 2D or 3D lacunar mask."""
    try:
        selected_class_value = class_value if foreground_value is None else foreground_value
        raw = _count_components_core(
            component_domain="lacunae",
            input_path=input_path,
            foreground_value=selected_class_value,
            connectivity=connectivity,
            min_size_voxels=min_size_voxels,
            max_size_voxels=max_size_voxels,
            include_component_sizes=include_component_sizes,
            voxel_size_um=voxel_size_um,
            save_component_csv=save_component_csv,
            output_csv_path=output_csv_path,
            sample_id=sample_id,
            result_mode="verbose",
            session_path=session_path,
            workspace_root=workspace_root,
        )
        if not raw.get("ok"):
            return _project_component_result(raw, component_domain="lacunae", result_mode=result_mode, sample_id=sample_id)

        raw["class_value"] = raw.pop("foreground_value")
        raw["n_lacunae"] = raw["n_components"]
        raw["lacunae_volumes_voxels"] = raw.get("component_sizes_voxels")
        raw["lacunae_volumes_um3"] = raw.get("component_volumes_um3")
        raw["lacunae_areas_um2"] = raw.get("component_areas_um2")
        raw["total_lacunae_volume_voxels"] = raw["total_component_volume_voxels"]
        raw["total_lacunae_volume_um3"] = raw.get("total_component_volume_um3")
        raw["total_lacunae_area_um2"] = raw.get("total_component_area_um2")
        return _project_component_result(raw, component_domain="lacunae", result_mode=result_mode, sample_id=sample_id)
    except Exception as exc:
        logger.exception("Lacunae counting failed")
        return _fail(str(exc), error_type=exc.__class__.__name__)


def count_cracks_mcp(
    input_path: str,
    foreground_value: float | None = None,
    class_value: float = 2,
    connectivity: int | None = None,
    min_size_voxels: int | None = None,
    max_size_voxels: int | None = None,
    include_component_sizes: bool = False,
    voxel_size_um: list[float] | None = None,
    save_component_csv: bool = True,
    output_csv_path: str | None = None,
    sample_id: str | None = None,
    result_mode: Literal["compact", "standard", "verbose"] = "compact",
    session_path: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Count cracks as connected class-2 components in a 2D or 3D mask."""
    try:
        selected_class_value = class_value if foreground_value is None else foreground_value
        raw = _count_components_core(
            component_domain="crack",
            input_path=input_path,
            foreground_value=selected_class_value,
            connectivity=connectivity,
            min_size_voxels=min_size_voxels,
            max_size_voxels=max_size_voxels,
            include_component_sizes=include_component_sizes,
            voxel_size_um=voxel_size_um,
            save_component_csv=save_component_csv,
            output_csv_path=output_csv_path,
            sample_id=sample_id,
            result_mode="verbose",
            session_path=session_path,
            workspace_root=workspace_root,
        )
        if not raw.get("ok"):
            return _project_component_result(raw, component_domain="crack", result_mode=result_mode, sample_id=sample_id)

        raw["class_value"] = raw.pop("foreground_value")
        raw["n_cracks"] = raw["n_components"]
        raw["crack_volumes_voxels"] = raw.get("component_sizes_voxels")
        raw["crack_volumes_um3"] = raw.get("component_volumes_um3")
        raw["crack_areas_um2"] = raw.get("component_areas_um2")
        raw["total_crack_volume_voxels"] = raw["total_component_volume_voxels"]
        raw["total_crack_volume_um3"] = raw.get("total_component_volume_um3")
        raw["total_crack_area_um2"] = raw.get("total_component_area_um2")
        return _project_component_result(raw, component_domain="crack", result_mode=result_mode, sample_id=sample_id)
    except Exception as exc:
        logger.exception("Crack counting failed")
        return _fail(str(exc), error_type=exc.__class__.__name__)


def calculate_bone_volume_otsu_mcp(
    input_path: str,
    include_zero_in_otsu: bool = False,
    voxel_size_um: list[float] | None = None,
    session_path: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Compute bone volume from an Otsu-thresholded preprocessed microscopy volume/image."""
    try:
        loaded = _load_array(
            input_path,
            session_path=session_path,
            workspace_root=workspace_root,
            voxel_size_um=voxel_size_um,
        )
        data = np.asarray(loaded.data, dtype=np.float64)
        if data.ndim not in {2, 3}:
            raise ValueError(f"Otsu bone volume supports only 2D or 3D arrays; got shape {data.shape}.")

        positive_mask = data > 0
        otsu_values = data.ravel() if include_zero_in_otsu else data[positive_mask]
        if otsu_values.size == 0:
            raise ValueError("No voxels/pixels available for Otsu thresholding.")

        # scikit-image does not expose type information for this runtime API.
        otsu_threshold = float(threshold_otsu(otsu_values))  # type: ignore[no-untyped-call]
        bone_mask = data > otsu_threshold
        n_bone = int(np.count_nonzero(bone_mask))
        element_measure_um = _voxel_measure(loaded.voxel_size_um)
        bone_measure_um = None if element_measure_um is None else float(n_bone * element_measure_um)

        result: dict[str, Any] = {
            "ok": True,
            "success": True,
            "tool_kind": TOOL_KIND,
            "input_path": str(loaded.path),
            "dimensionality": int(data.ndim),
            "shape": [int(dim) for dim in data.shape],
            "otsu_threshold": otsu_threshold,
            "threshold_rule": "data > otsu_threshold",
            "otsu_computed_on": "all_voxels_including_zero" if include_zero_in_otsu else "positive_voxels_only",
            "zeros_excluded_from_otsu": not include_zero_in_otsu,
            "n_total_voxels": int(data.size),
            "n_positive_voxels": int(np.count_nonzero(positive_mask)),
            "n_bone_voxels": n_bone,
            "voxel_size_um": loaded.voxel_size_um,
            "voxel_size_source": loaded.voxel_size_source,
            "voxel_volume_um3": element_measure_um if data.ndim == 3 else None,
            "bone_volume": n_bone,
            "bone_volume_um3": bone_measure_um if data.ndim == 3 else None,
            "bone_volume_mm3": (bone_measure_um / 1e9) if bone_measure_um is not None and data.ndim == 3 else None,
            "pixel_size_um": loaded.voxel_size_um if data.ndim == 2 else None,
            "pixel_area_um2": element_measure_um if data.ndim == 2 else None,
            "bone_area_pixels": n_bone if data.ndim == 2 else None,
            "bone_area_um2": bone_measure_um if data.ndim == 2 else None,
            "attachments": [],
        }
        return _clean_payload(result)
    except InputPathResolutionError as exc:
        logger.warning(
            "Otsu bone volume calculation skipped because input path was not found: %s",
            exc.input_path,
        )
        return _input_path_resolution_failure(exc)
    except Exception as exc:
        logger.exception("Otsu bone volume calculation failed")
        return _fail(str(exc), error_type=exc.__class__.__name__)


class CountConnectedComponentsTool(BaseTool):
    name: str = "count_connected_components"
    description: str = (
        "Count connected foreground objects in an existing binary or label mask. "
        "Supports NIfTI volumes, 2D TIFF files, multipage TIFFs, and folders of TIFF slices. "
        "Use as a generic low-priority fallback for non-lacuna and non-crack object counting, "
        "component-size filtering, per-component voxel counts, and physical area/volume totals."
    )
    args_schema: type[BaseModel] = ConnectedComponentsArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        return count_connected_components_mcp(**kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


class CountLacunaeTool(BaseTool):
    name: str = "count_lacunae"
    description: str = (
        "Count lacunae as connected class-1 foreground components in a 2D TIFF mask or 3D NIfTI/TIFF mask. "
        "Defaults to class_value 1, 8-connectivity in 2D, and 26-connectivity in 3D. "
        "Returns a compact JSON summary by default and writes post-filtering per-lacuna voxel counts "
        "and physical volumes/areas to a CSV attachment by default. Per-lacuna lists are never returned "
        "in JSON when CSV output is enabled."
    )
    args_schema: type[BaseModel] = CountLacunaeArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        return count_lacunae_mcp(**kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


class CountCracksTool(BaseTool):
    name: str = "count_cracks"
    description: str = (
        "Count cracks as connected class-2 foreground components in a 2D TIFF mask or 3D NIfTI/TIFF mask. "
        "Defaults to class_value 2, 8-connectivity in 2D, and 26-connectivity in 3D. "
        "Returns a compact JSON summary by default and writes post-filtering per-crack voxel counts "
        "and physical volumes/areas to a CSV attachment by default. Per-crack lists are never returned "
        "in JSON when CSV output is enabled."
    )
    args_schema: type[BaseModel] = CountCracksArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        return count_cracks_mcp(**kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


class CalculateBoneVolumeOtsuTool(BaseTool):
    name: str = "calculate_bone_volume_otsu"
    description: str = (
        "Compute bone tissue volume from a preprocessed microscopy image or volume using Otsu thresholding. "
        "By default Otsu is computed only on strictly positive voxels so zero-valued outside-sample background "
        "does not dominate the threshold."
    )
    args_schema: type[BaseModel] = BoneVolumeOtsuArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        return calculate_bone_volume_otsu_mcp(**kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


LoadedArray.model_rebuild(
    _types_namespace={
        "Any": Any,
        "Path": Path,
        "Literal": Literal,
    }
)
ConnectedComponentsArgs.model_rebuild(
    _types_namespace={
        "Field": Field,
        "Literal": Literal,
    }
)
BoneVolumeOtsuArgs.model_rebuild(
    _types_namespace={
        "Field": Field,
    }
)
CountLacunaeArgs.model_rebuild(
    _types_namespace={
        "Field": Field,
        "Literal": Literal,
    }
)
CountCracksArgs.model_rebuild(
    _types_namespace={
        "Field": Field,
        "Literal": Literal,
    }
)
CountConnectedComponentsTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "ConnectedComponentsArgs": ConnectedComponentsArgs,
    }
)
CountLacunaeTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "CountLacunaeArgs": CountLacunaeArgs,
    }
)
CountCracksTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "CountCracksArgs": CountCracksArgs,
    }
)
CalculateBoneVolumeOtsuTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "BoneVolumeOtsuArgs": BoneVolumeOtsuArgs,
    }
)


EXPORTED_TOOLS: dict[str, BaseTool] = {
    "count_connected_components": CountConnectedComponentsTool(),
    "calculate_bone_volume_otsu": CalculateBoneVolumeOtsuTool(),
    "count_lacunae": CountLacunaeTool(),
    "count_cracks": CountCracksTool(),
}


def register_mcp_tools(mcp_server: Any) -> None:
    """Register plain MCP-callable functions on an existing MCP server instance."""
    if not hasattr(mcp_server, "tool"):
        raise TypeError("mcp_server must expose a .tool() decorator-compatible registration API.")
    mcp_server.tool()(count_connected_components_mcp)
    mcp_server.tool()(calculate_bone_volume_otsu_mcp)
    mcp_server.tool()(count_lacunae_mcp)
    mcp_server.tool()(count_cracks_mcp)
