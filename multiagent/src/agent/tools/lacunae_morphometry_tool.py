import sys
import logging
import csv
import hashlib
import json
import shutil
import time
from numbers import Real
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from agent.tools.path_resolution import missing_input_path_message, resolve_input_path


logger = logging.getLogger(__name__)
LACUNAE_IMPORT_ERROR: Exception | None = None
LACUNAE_CACHE_MAX_ENTRIES = 12
LACUNAE_CACHE_PARTIAL_HASH_BYTES = 1024 * 1024
LACUNAE_CACHE_LOCK_POLL_SECONDS = 1.0
LACUNAE_CACHE_LOCK_STALE_SECONDS = 12 * 60 * 60
TOOL_OUTPUT_FLOAT_DECIMALS = 4

# --- Dynamic Path Setup to import 'lacune_parameters_blocks_nii' ---
BIOMED_AGENT_ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK_ROOT = (BIOMED_AGENT_ROOT / "framework").resolve()
LACUNAE_CACHE_ROOT = (BIOMED_AGENT_ROOT / ".cache" / "lacunae_parameters").resolve()

if not FRAMEWORK_ROOT.exists():
    logger.error(f"Framework folder not found at: {FRAMEWORK_ROOT}")
else:
    if str(FRAMEWORK_ROOT) not in sys.path:
        sys.path.insert(0, str(FRAMEWORK_ROOT))

try:
    import lacune_parameters_blocks_nii  # type: ignore[import-not-found]
except ImportError as e:
    logger.warning(f"Could not import lacune_parameters_blocks_nii: {e}")
    LACUNAE_IMPORT_ERROR = e
    lacune_parameters_blocks_nii = None
# -------------------------------------------------------------------


class LacunaeParametersArgs(BaseModel):
    path_original_volume: str = Field(
        ...,
        description=(
            "Path to the grayscale NIfTI volume to analyze. Relative paths are resolved from the active session first."
        ),
    )
    path_seg: str = Field(
        ...,
        description=(
            "Path to the segmentation NIfTI mask matching the input volume. Relative paths are resolved from the active session first."
        ),
    )
    sample_name: str = Field(
        default="sample_01",
        description=(
            "Sample identifier used to name the generated outputs."
        ),
    )
    nr_slice_per_block: int = Field(
        default=100,
        description=(
            "Number of Z slices processed in each block."
        ),
    )
    overlap_percentage: int = Field(
        default=30,
        ge=0,
        le=99,
        description=(
            "Percentage of overlap between consecutive Z blocks."
        ),
    )
    processed: bool = Field(
        default=True,
        description=(
            "Whether the input grayscale volume should be treated as already preprocessed."
        ),
    )
    output_folder: str | None = Field(
        default=None,
        description=(
            "Optional folder where outputs are written. If omitted, a default output "
            "folder is used."
        ),
    )
    slice_start: int = Field(
        default=0,
        description=(
            "First Z slice to include in the analysis. Slice ranges are half-open: "
            "[slice_start, slice_end)."
        ),
    )
    slice_end: int | None = Field(
        default=None,
        description=(
            "Exclusive upper Z slice boundary for the analysis. Use one past the last "
            "included slice; for example 460-530 includes slices 460 through 529 "
            "and yields 70 slices. If omitted, the analysis runs to the end of the volume."
        ),
    )
    save_block_files: bool = Field(
        default=False,
        description=(
            "Whether to save additional outputs for individual processing blocks."
        ),
    )
    session_path: str | None = Field(
        default=None,
        description=(
            "Internal session folder used to place outputs in the session output directory."
        ),
    )

# =========================
# Shared core logic
# =========================

def _fail(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "tool_kind": "metrics",
        "error": message,
        "attachments": [],
    }


def _resolve_output_folder(
    path_original_volume: str,
    output_folder: str | None = None,
    session_path: str | None = None,
) -> str:
    if session_path:
        return str((Path(session_path).resolve() / "output"))
    if output_folder:
        return str(Path(output_folder).resolve())
    return str(Path(path_original_volume).resolve().parent / "lacunae_analysis")


def _build_attachments_from_framework_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    attachments: list[dict[str, Any]] = []
    output_specs = [
        (
            "individual_lacunae_csv",
            "metrics",
            "Per-lacuna individual morphometry CSV.",
        ),
        (
            "density_distribution_csv",
            "metrics",
            "Block-level lacuna density and tissue-volume statistics CSV.",
        ),
        (
            "global_summary_csv",
            "metrics",
            "Global lacuna morphometry summary CSV.",
        ),
    ]

    for key, kind, description in output_specs:
        raw_path = result.get(key)
        if not raw_path:
            continue
        path_obj = Path(raw_path).resolve()
        if not path_obj.exists() or not path_obj.is_file():
            logger.warning("Skipping missing framework output for key '%s': %s", key, path_obj)
            continue
        attachments.append(
            {
                "path": str(path_obj),
                "kind": kind,
                "description": description,
                "parent_arg": "path_original_volume",
            }
        )
    return attachments


def _coerce_csv_scalar(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _round_tool_output_numbers(value: Any, decimals: int = TOOL_OUTPUT_FLOAT_DECIMALS) -> Any:
    if isinstance(value, dict):
        return {
            key: _round_tool_output_numbers(nested_value, decimals)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_round_tool_output_numbers(item, decimals) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        return round(numeric, decimals)
    return value


def _read_single_row_csv(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_file():
        return {}
    with path_obj.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        try:
            row = next(reader)
        except StopIteration:
            return {}
    return {key: _coerce_csv_scalar(value) for key, value in row.items()}


def _read_numeric_column_summary(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_file():
        return {}
    values_by_column: dict[str, list[float]] = {}
    with path_obj.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                if value is None or value.strip() == "":
                    continue
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                values_by_column.setdefault(key, []).append(numeric)

    summary: dict[str, Any] = {}
    for key, values in values_by_column.items():
        if not values:
            continue
        count = len(values)
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        summary[key] = {
            "count": count,
            "mean": mean,
            "std": variance ** 0.5,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _sampled_file_digest(path: Path) -> dict[str, Any]:
    stat = path.stat()
    size = int(stat.st_size)
    chunk = LACUNAE_CACHE_PARTIAL_HASH_BYTES
    offsets = {0}
    if size > chunk:
        offsets.add(max(0, (size - chunk) // 2))
        offsets.add(max(0, size - chunk))

    digest = hashlib.blake2b(digest_size=32)
    digest.update(str(size).encode("ascii"))
    digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    with path.open("rb") as handle:
        for offset in sorted(offsets):
            handle.seek(offset)
            digest.update(str(offset).encode("ascii"))
            digest.update(handle.read(min(chunk, max(0, size - offset))))

    return {
        "size": size,
        "mtime_ns": int(stat.st_mtime_ns),
        "sampled_blake2b": digest.hexdigest(),
    }


def _nifti_header_fingerprint(path: Path) -> dict[str, Any]:
    try:
        import nibabel as nib

        img: Any = nib.load(str(path))
        return {
            "shape": [int(dim) for dim in img.shape],
            "zooms": [float(value) for value in img.header.get_zooms()[: len(img.shape)]],
            "units": list(img.header.get_xyzt_units()),
        }
    except Exception as exc:
        logger.debug("Could not read NIfTI header while building lacunae cache key: %s", exc)
        return {}


def _file_content_identifier(path: str) -> dict[str, Any]:
    path_obj = Path(path).resolve()
    return {
        "path_name": path_obj.name,
        "file": _sampled_file_digest(path_obj),
        "nifti": _nifti_header_fingerprint(path_obj),
    }


def _effective_slice_end(seg_path: str, slice_end: int | None) -> int:
    if slice_end is not None:
        return int(slice_end)
    import nibabel as nib

    image: Any = nib.load(str(seg_path))
    shape = image.shape
    if len(shape) < 3:
        raise ValueError(f"Segmentation volume must be 3D; got shape {shape}.")
    return int(shape[2])


def _cache_key_payload(
    *,
    resolved_volume: str,
    resolved_seg: str,
    nr_slice_per_block: int,
    overlap_percentage: int,
    processed: bool,
    slice_start: int,
    effective_slice_end: int,
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        volume_future = executor.submit(_file_content_identifier, resolved_volume)
        seg_future = executor.submit(_file_content_identifier, resolved_seg)
        volume_id = volume_future.result()
        seg_id = seg_future.result()

    return {
        "cache_schema_version": 1,
        "volume": volume_id,
        "segmentation": seg_id,
        "parameters": {
            "nr_slice_per_block": int(nr_slice_per_block),
            "overlap_percentage": int(overlap_percentage),
            "processed": bool(processed),
            "slice_start": int(slice_start),
            "slice_end": int(effective_slice_end),
        },
    }


def _cache_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(serialized, digest_size=32).hexdigest()


def _expected_framework_result_paths(
    *,
    resolved_output_folder: str,
    sample_name: str,
    slice_start: int,
    effective_slice_end: int,
) -> dict[str, str]:
    sample_out_dir = Path(resolved_output_folder).resolve() / sample_name / "3D_params_files"
    range_suffix = f"z{int(slice_start)}-{int(effective_slice_end)}"
    return {
        "individual_lacunae_csv": str(
            sample_out_dir / f"{sample_name}_lacunae_individual_morphometry_{range_suffix}.csv"
        ),
        "density_distribution_csv": str(
            sample_out_dir / f"{sample_name}_bone_density_distribution_per_block_{range_suffix}.csv"
        ),
        "global_summary_csv": str(
            sample_out_dir / f"{sample_name}_global_morphometry_summary_{range_suffix}.csv"
        ),
    }


def _cache_payload_paths(entry_dir: Path) -> dict[str, Path]:
    return {
        "individual_lacunae_csv": entry_dir / "individual_lacunae.csv",
        "density_distribution_csv": entry_dir / "density_distribution.csv",
        "global_summary_csv": entry_dir / "global_summary.csv",
    }


def _restore_cached_result(
    entry_dir: Path,
    expected_paths: dict[str, str],
    *,
    require_block_files: bool,
) -> dict[str, Any] | None:
    metadata_path = entry_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if require_block_files and not bool(metadata.get("cached_with_save_block_files")):
        return None

    cached_outputs = metadata.get("cached_outputs") or {}
    cache_paths = _cache_payload_paths(entry_dir)
    restored: dict[str, Any] = {}
    for key, expected_path in expected_paths.items():
        if not cached_outputs.get(key):
            restored[key] = None
            continue
        cached_path = cache_paths[key]
        if not cached_path.exists():
            return None
        destination = Path(expected_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_path, destination)
        restored[key] = str(destination)

    now = time.time()
    try:
        metadata["last_accessed_at"] = now
        metadata["access_count"] = int(metadata.get("access_count") or 0) + 1
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        entry_dir.touch()
    except Exception as exc:
        logger.debug("Could not update lacunae cache LRU metadata: %s", exc)
    return restored


def _store_cached_result(
    *,
    cache_key: str,
    key_payload: dict[str, Any],
    result: dict[str, Any],
    save_block_files: bool,
) -> None:
    LACUNAE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    entry_dir = LACUNAE_CACHE_ROOT / cache_key
    tmp_dir = LACUNAE_CACHE_ROOT / f".{cache_key}.tmp-{time.time_ns()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    cached_outputs: dict[str, bool] = {}
    cache_paths = _cache_payload_paths(tmp_dir)
    try:
        for key, destination in cache_paths.items():
            source_raw = result.get(key)
            if source_raw and Path(source_raw).exists():
                shutil.copy2(Path(source_raw), destination)
                cached_outputs[key] = True
            else:
                cached_outputs[key] = False

        metadata = {
            "cache_key": cache_key,
            "created_at": time.time(),
            "last_accessed_at": time.time(),
            "access_count": 0,
            "key_payload": key_payload,
            "cached_with_save_block_files": bool(save_block_files),
            "cached_outputs": cached_outputs,
        }
        (tmp_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        tmp_dir.rename(entry_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def _prune_lacunae_cache(max_entries: int = LACUNAE_CACHE_MAX_ENTRIES) -> None:
    if not LACUNAE_CACHE_ROOT.exists():
        return
    entries = [
        path
        for path in LACUNAE_CACHE_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".") and not path.name.endswith(".lock")
    ]
    if len(entries) <= max_entries:
        return

    def entry_last_accessed(path: Path) -> float:
        metadata_path = path / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return float(metadata.get("last_accessed_at") or metadata.get("created_at") or path.stat().st_mtime)
        except Exception:
            return float(path.stat().st_mtime)

    for entry in sorted(entries, key=entry_last_accessed)[: max(0, len(entries) - max_entries)]:
        try:
            shutil.rmtree(entry)
        except Exception as exc:
            logger.debug("Could not prune lacunae cache entry %s: %s", entry, exc)


def _acquire_cache_lock(lock_dir: Path) -> bool:
    while True:
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            (lock_dir / "created_at").write_text(str(time.time()), encoding="utf-8")
            return True
        except FileExistsError:
            try:
                age = time.time() - lock_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > LACUNAE_CACHE_LOCK_STALE_SECONDS:
                try:
                    shutil.rmtree(lock_dir)
                    continue
                except Exception:
                    pass
            return False


def _wait_for_cache_entry(
    entry_dir: Path,
    lock_dir: Path,
    timeout_seconds: float = 60.0,
) -> bool:
    started_at = time.time()

    while lock_dir.exists():
        if entry_dir.exists():
            return True

        elapsed = time.time() - started_at
        if elapsed > timeout_seconds:
            logger.warning(
                "Timed out waiting for lacunae cache entry. entry_dir=%s lock_dir=%s",
                entry_dir,
                lock_dir,
            )
            return False

        try:
            age = time.time() - lock_dir.stat().st_mtime
        except FileNotFoundError:
            return entry_dir.exists()

        if age > LACUNAE_CACHE_LOCK_STALE_SECONDS:
            logger.warning(
                "Removing stale lacunae cache lock while waiting: %s age_seconds=%.1f",
                lock_dir,
                age,
            )
            try:
                shutil.rmtree(lock_dir)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(
                    "Could not remove stale lacunae cache lock %s: %s",
                    lock_dir,
                    exc,
                )
                return False

            return entry_dir.exists()

        time.sleep(LACUNAE_CACHE_LOCK_POLL_SECONDS)

    return entry_dir.exists()


def _release_cache_lock(lock_dir: Path) -> None:
    try:
        shutil.rmtree(lock_dir)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("Could not release lacunae cache lock %s: %s", lock_dir, exc)


def _successful_tool_response(
    *,
    result: dict[str, Any],
    resolved_output_folder: str,
    sample_name: str,
) -> dict[str, Any]:
    attachments = _build_attachments_from_framework_result(result)
    global_summary = _read_single_row_csv(result.get("global_summary_csv"))
    density_distribution_summary = _read_numeric_column_summary(
        result.get("density_distribution_csv")
    )
    global_summary = _round_tool_output_numbers(global_summary)
    density_distribution_summary = _round_tool_output_numbers(density_distribution_summary)

    return {
        "success": True,
        "tool_kind": "metrics",
        "message": "Lacunae parameter extraction completed successfully.",
        "attachments": attachments,
        "output_folder": resolved_output_folder,
        "sample_name": sample_name,
        "outputs": result,
        "global_summary": global_summary,
        "density_distribution_summary": density_distribution_summary,
    }


def run_lacunae_parameters(
    *,
    path_original_volume: str,
    path_seg: str,
    sample_name: str = "sample_01",
    nr_slice_per_block: int = 100,
    overlap_percentage: int = 30,
    processed: bool = True,
    output_folder: str | None = None,
    slice_start: int = 0,
    slice_end: int | None = None,
    save_block_files: bool = False,
    session_path: str | None = None,
) -> dict[str, Any]:
    if lacune_parameters_blocks_nii is None:
        if LACUNAE_IMPORT_ERROR is not None:
            return _fail(
                "Framework module 'lacune_parameters_blocks_nii' could not be loaded: "
                f"{LACUNAE_IMPORT_ERROR}. Install the missing dependency and retry."
            )
        return _fail("Framework module 'lacune_parameters_blocks_nii' not loaded.")

    try:
        resolved_volume_path = resolve_input_path(
            path_original_volume,
            session_path=session_path,
        )
        if resolved_volume_path.path is None:
            return _fail(
                missing_input_path_message(
                    "Original volume",
                    path_original_volume,
                    resolved_volume_path.searched_paths,
                )
            )

        resolved_seg_path = resolve_input_path(path_seg, session_path=session_path)
        if resolved_seg_path.path is None:
            return _fail(
                missing_input_path_message(
                    "Segmentation volume",
                    path_seg,
                    resolved_seg_path.searched_paths,
                )
            )

        resolved_volume = str(resolved_volume_path.path)
        resolved_seg = str(resolved_seg_path.path)

        resolved_output_folder = _resolve_output_folder(
            path_original_volume=resolved_volume,
            output_folder=output_folder,
            session_path=session_path,
        )
        Path(resolved_output_folder).mkdir(parents=True, exist_ok=True)

        # Normalize: the LLM occasionally passes 0 (e.g. from "Z0" in filenames)
        # when it means "unset". Treat any non-positive value as None so the full
        # volume depth is used.
        if slice_end is not None and slice_end <= 0:
            slice_end = None

        effective_slice_end = _effective_slice_end(resolved_seg, slice_end)
        expected_paths = _expected_framework_result_paths(
            resolved_output_folder=resolved_output_folder,
            sample_name=sample_name,
            slice_start=slice_start,
            effective_slice_end=effective_slice_end,
        )
        key_payload = _cache_key_payload(
            resolved_volume=resolved_volume,
            resolved_seg=resolved_seg,
            nr_slice_per_block=nr_slice_per_block,
            overlap_percentage=overlap_percentage,
            processed=processed,
            slice_start=slice_start,
            effective_slice_end=effective_slice_end,
        )
        cache_key = _cache_key(key_payload)
        entry_dir = LACUNAE_CACHE_ROOT / cache_key
        lock_dir = LACUNAE_CACHE_ROOT / f"{cache_key}.lock"

        cached_result = _restore_cached_result(
            entry_dir,
            expected_paths,
            require_block_files=save_block_files,
        )
        if cached_result is not None:
            return _successful_tool_response(
                result=cached_result,
                resolved_output_folder=resolved_output_folder,
                sample_name=sample_name,
            )

        owns_lock = _acquire_cache_lock(lock_dir)
        while not owns_lock:
                if _wait_for_cache_entry(entry_dir, lock_dir, timeout_seconds=120.0):
                    cached_result = _restore_cached_result(
                        entry_dir,
                        expected_paths,
                        require_block_files=save_block_files,
                    )
                    if cached_result is not None:
                        return _successful_tool_response(
                            result=cached_result,
                            resolved_output_folder=resolved_output_folder,
                            sample_name=sample_name,
                        )

                owns_lock = _acquire_cache_lock(lock_dir)

                if not owns_lock:
                    return _fail(
                        f"Could not acquire lacunae cache lock after waiting: {lock_dir}"
                    )

        try:
            cached_result = _restore_cached_result(
                entry_dir,
                expected_paths,
                require_block_files=save_block_files,
            )
            if cached_result is not None:
                _release_cache_lock(lock_dir)
                return _successful_tool_response(
                    result=cached_result,
                    resolved_output_folder=resolved_output_folder,
                    sample_name=sample_name,
                )

            result = lacune_parameters_blocks_nii.main(
                nr_slice_per_block=nr_slice_per_block,
                sample_name=sample_name,
                path_original_volume=resolved_volume,
                path_seg=resolved_seg,
                processed=processed,
                parent_folder=resolved_output_folder,
                overlap_percentage=overlap_percentage,
                slice_start=slice_start,
                slice_end=slice_end,
                save_block_files=save_block_files,
            )
        except Exception:
            if owns_lock:
                _release_cache_lock(lock_dir)
            raise

        if not isinstance(result, dict):
            if owns_lock:
                _release_cache_lock(lock_dir)
            return _fail(
                "Framework function returned a non-dict payload; expected an output-path dictionary."
            )

        if owns_lock:
            try:
                _store_cached_result(
                    cache_key=cache_key,
                    key_payload=key_payload,
                    result=result,
                    save_block_files=save_block_files,
                )
                _prune_lacunae_cache()
            finally:
                _release_cache_lock(lock_dir)

        return _successful_tool_response(
            result=result,
            resolved_output_folder=resolved_output_folder,
            sample_name=sample_name,
        )

    except Exception as e:
        logger.exception("Error in lacunae parameter calculation")
        return _fail(str(e))


# =========================
# LangChain Tool
# =========================

class LacunaeParametersTool(BaseTool):
    name: str = "calculate_lacunae_parameters"
    description: str = (
        "Extract 3D lacuna morphometric measurements from a grayscale bone volume and its "
        "matching lacuna segmentation. Use this tool when the image and segmentation are "
        "already available and you need quantitative lacuna-level outputs for downstream "
        "analysis. The result should be interpreted as a morphometric characterization of "
        "the analyzed sample, with tabular outputs that can support later comparison or reporting. "
        "Slice ranges are half-open: slice_start is included and slice_end is excluded."
    )
    args_schema: type[BaseModel] = LacunaeParametersArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        return run_lacunae_parameters(**kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Async not supported")


EXPORTED_TOOLS: dict[str, BaseTool] = {
    "calculate_lacunae_parameters": LacunaeParametersTool()
}



def calculate_lacunae_parameters_mcp(
    path_original_volume: str,
    path_seg: str,
    sample_name: str = "sample_01",
    nr_slice_per_block: int = 100,
    overlap_percentage: int = 30,
    processed: bool = True,
    output_folder: str | None = None,
    slice_start: int = 0,
    slice_end: int | None = None,
    save_block_files: bool = False,
    session_path: str | None = None,
) -> dict[str, Any]:
    """
    Compute 3D lacuna morphometry from a grayscale bone volume and its matching 
    lacuna segmentation. The tool extracts per-lacuna geometric and orientation-related 
    measurements, estimates block-wise lacuna density and tissue-volume statistics, 
    and produces structured CSV outputs for downstream quantitative analysis, reporting, 
    and group comparison workflows.

    Slice ranges are half-open: slice_start is included and slice_end is excluded.
    For example, slice_start=460 and slice_end=530 analyzes slices 460 through 529.
    """
    return run_lacunae_parameters(
        path_original_volume=path_original_volume,
        path_seg=path_seg,
        sample_name=sample_name,
        nr_slice_per_block=nr_slice_per_block,
        overlap_percentage=overlap_percentage,
        processed=processed,
        output_folder=output_folder,
        slice_start=slice_start,
        slice_end=slice_end,
        save_block_files=save_block_files,
        session_path=session_path,
    )
