import logging
import os
from pathlib import Path
from collections.abc import Callable, Iterator
from typing import Any, cast

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.tools import BaseTool

from agent.tools.path_resolution import missing_input_path_message, resolve_input_path

logger = logging.getLogger(__name__)

TOOL_NAME = "calculate_segmentation_metrics"
TOOL_KIND = "metrics"


ROUTING_DESCRIPTION = (
    "Compute segmentation quality metrics by comparing predicted masks against ground truth masks. "
    "Use this tool when you need to evaluate how well a segmentation matches its reference, either "
    "for one prediction/ground-truth pair or for an aligned batch of pairs. "
    "Both inputs must be segmentation masks or label maps, not raw grayscale/intensity images. "
    "The prediction and ground-truth inputs must be distinct files; never compare a mask with itself, "
    "because self-comparison produces meaningless perfect scores and is treated as an invalid evaluation. "
    "This tool works at segmentation-evaluation level and returns class-wise quality scores that support "
    "model assessment, result validation, and comparison across segmentation outputs."
)

MetricFunction = Callable[[int, int, int, int], float]

METRIC_REGISTRY: dict[str, MetricFunction] = {}


def compute_confusion_matrix_elements(pred: np.ndarray, gt: np.ndarray, label_value: int) -> tuple[int, int, int, int]:
    """Compute TP, FP, FN, TN for one class in a one-vs-rest setting."""
    p_mask = pred == label_value
    g_mask = gt == label_value

    tp = int(np.logical_and(p_mask, g_mask).sum())
    fp = int(np.logical_and(p_mask, ~g_mask).sum())
    fn = int(np.logical_and(~p_mask, g_mask).sum())
    tn = int(np.logical_and(~p_mask, ~g_mask).sum())
    return tp, fp, fn, tn



def metric_dice(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom > 0 else 0.0



def metric_iou(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0



def metric_precision(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0



def metric_recall(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0



def metric_accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    total = tp + fp + fn + tn
    return float((tp + tn) / total) if total > 0 else 0.0


METRIC_REGISTRY = {
    "dice": metric_dice,
    "f1": metric_dice,
    "iou": metric_iou,
    "jaccard": metric_iou,
    "precision": metric_precision,
    "recall": metric_recall,
    "sensitivity": metric_recall,
    "accuracy": metric_accuracy,
}

DEFAULT_METRICS = ["dice", "iou", "precision", "recall", "f1"]
DEFAULT_CLASS_MAP = {1: "lacunae", 2: "cracks"}
DEFAULT_NIFTI_CHUNK_SLICES = 4


class SegmentationMetricsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_path: str | None = Field(
        default=None,
        description=(
            "Path to the predicted segmentation mask or label map for single-file evaluation. "
            "Do not pass the raw microscopy volume here. Must not be the same file as ground_truth_path."
        ),
    )
    ground_truth_path: str | None = Field(
        default=None,
        description=(
            "Path to the ground-truth segmentation mask or label map for single-file evaluation. "
            "Do not pass the raw microscopy volume here. Must not be the same file as prediction_path."
        ),
    )
    prediction_files: list[str] | None = Field(
        default=None,
        description="Ordered list of predicted segmentation files for batch evaluation.",
    )
    ground_truth_files: list[str] | None = Field(
        default=None,
        description=(
            "Ordered list of ground-truth segmentation files for batch evaluation. "
            "Each file must align one-to-one with the corresponding entry in 'prediction_files'."
        ),
    )
    metrics: list[str] = Field(
        default_factory=lambda: list(DEFAULT_METRICS),
        description=(
            "Metric names to compute. Supported values are dice, f1, iou, jaccard, "
            "precision, recall, sensitivity, and accuracy."
        ),
    )
    class_map: dict[int, str] = Field(
        default_factory=lambda: dict(DEFAULT_CLASS_MAP),
        description=(
            "Mapping from integer label values to semantic class names used in the reported results."
        ),
    )
    session_path: str | None = Field(
        default=None,
        description="Optional active session directory used to resolve relative input paths.",
    )


def _fail(error: str) -> dict[str, Any]:
    return {
        "success": False,
        "tool_kind": TOOL_KIND,
        "error": error,
        "attachments": [],
    }



def _resolve_path(path_str: str, *, session_path: str | None = None) -> Path:
    resolved = resolve_input_path(path_str, session_path=session_path)
    if resolved.path is None:
        raise FileNotFoundError(
            missing_input_path_message("File", path_str, resolved.searched_paths)
        )
    return resolved.path



def _load_image_as_array(path_str: str, *, session_path: str | None = None) -> np.ndarray:
    path = _resolve_path(path_str, session_path=session_path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes in {".nii", ".nii.gz"}:
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("nibabel is required to load NIfTI files.") from exc

        img: Any = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
        if np.issubdtype(data.dtype, np.integer):
            return cast(np.ndarray, data)
        return cast(np.ndarray, np.rint(data))

    try:
        with Image.open(path) as img:
            return np.array(img).astype(np.int64)
    except Exception as exc:
        raise ValueError(f"Could not load image file {path}: {exc}") from exc


def _is_nifti_path(path: Path) -> bool:
    return "".join(path.suffixes).lower() in {".nii", ".nii.gz"}


def _metric_chunk_slices() -> int:
    raw = os.environ.get("SEGMENTATION_METRICS_CHUNK_SLICES")
    if raw:
        try:
            return max(int(raw), 1)
        except ValueError:
            logger.warning("Ignoring invalid SEGMENTATION_METRICS_CHUNK_SLICES=%r", raw)
    return DEFAULT_NIFTI_CHUNK_SLICES


def _load_label_chunk(dataobj: Any, slicer: tuple[Any, ...]) -> np.ndarray:
    chunk = np.asanyarray(dataobj[slicer])
    if np.issubdtype(chunk.dtype, np.integer):
        return cast(np.ndarray, chunk)
    return cast(np.ndarray, np.rint(chunk))


def _iter_nifti_label_chunks(path: Path) -> Iterator[np.ndarray]:
    import nibabel as nib

    img: Any = nib.load(str(path))
    shape = tuple(int(dim) for dim in img.shape)
    if len(shape) < 3:
        yield _load_label_chunk(img.dataobj, (...,))
        return

    chunk_slices = _metric_chunk_slices()
    total_slices = shape[2]
    for start in range(0, total_slices, chunk_slices):
        end = min(start + chunk_slices, total_slices)
        slicer = (slice(None), slice(None), slice(start, end)) + (slice(None),) * max(len(shape) - 3, 0)
        yield _load_label_chunk(img.dataobj, slicer)


def _load_pair_metadata(prediction_path: Path, ground_truth_path: Path) -> tuple[tuple[int, ...], bool]:
    pred_is_nifti = _is_nifti_path(prediction_path)
    gt_is_nifti = _is_nifti_path(ground_truth_path)
    if pred_is_nifti != gt_is_nifti:
        raise ValueError("Prediction and ground truth must use the same file family: both NIfTI or both image files.")

    if pred_is_nifti:
        import nibabel as nib

        prediction_image: Any = nib.load(str(prediction_path))
        ground_truth_image: Any = nib.load(str(ground_truth_path))
        pred_shape = tuple(int(dim) for dim in prediction_image.shape)
        gt_shape = tuple(int(dim) for dim in ground_truth_image.shape)
        if pred_shape != gt_shape:
            raise ValueError(
                f"Shape mismatch between prediction and ground truth: {pred_shape} vs {gt_shape}."
            )
        return pred_shape, True

    pred_arr = _load_image_as_array(str(prediction_path))
    gt_arr = _load_image_as_array(str(ground_truth_path))
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Shape mismatch between prediction and ground truth: {tuple(pred_arr.shape)} vs {tuple(gt_arr.shape)}."
        )
    return tuple(int(dim) for dim in pred_arr.shape), False


def _is_same_resolved_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return os.path.normcase(str(left.resolve(strict=False))) == os.path.normcase(str(right.resolve(strict=False)))


def _validate_distinct_pair(prediction_path: Path, ground_truth_path: Path) -> None:
    if _is_same_resolved_file(prediction_path, ground_truth_path):
        raise ValueError(
            "Invalid segmentation metrics request: prediction_path and ground_truth_path resolve to the same file. "
            "Segmentation quality evaluation requires a predicted mask and an independent ground-truth mask; "
            "self-comparison would produce meaningless perfect scores."
        )


def _collect_labels_streaming(path: Path) -> list[int]:
    labels: set[int] = set()
    if _is_nifti_path(path):
        for chunk in _iter_nifti_label_chunks(path):
            labels.update(int(label) for label in np.unique(chunk) if int(label) != 0)
    else:
        arr = _load_image_as_array(str(path))
        labels.update(int(label) for label in np.unique(arr) if int(label) != 0)
    return sorted(labels)


def _looks_like_dense_intensity_image(labels: list[int], class_map: dict[int, str]) -> bool:
    if len(labels) < 16:
        return False

    known_labels = set(class_map.keys())
    unknown_labels = [label for label in labels if label not in known_labels]
    if len(unknown_labels) < 8:
        return False

    label_span = labels[-1] - labels[0] + 1
    if label_span <= 0:
        return False

    density = len(labels) / label_span
    return labels[-1] >= 31 and density >= 0.8


def _validate_mask_like_labels(path: Path, labels: list[int], class_map: dict[int, str], role: str) -> None:
    if _looks_like_dense_intensity_image(labels, class_map):
        raise ValueError(
            f"{role} '{path}' appears to be a grayscale/intensity image, not a segmentation mask: "
            f"found {len(labels)} dense non-zero values from {labels[0]} to {labels[-1]}. "
            "Pass a predicted mask as prediction_path and a ground-truth mask as ground_truth_path."
        )


def _compute_confusion_streaming(
    prediction_path: Path,
    ground_truth_path: Path,
    label_value: int,
) -> tuple[int, int, int, int]:
    if _is_nifti_path(prediction_path):
        tp = fp = fn = total = 0
        for pred_chunk, gt_chunk in zip(
            _iter_nifti_label_chunks(prediction_path),
            _iter_nifti_label_chunks(ground_truth_path),
        ):
            if pred_chunk.shape != gt_chunk.shape:
                raise ValueError(
                    f"Chunk shape mismatch between prediction and ground truth: {pred_chunk.shape} vs {gt_chunk.shape}."
                )

            p_mask = pred_chunk == label_value
            g_mask = gt_chunk == label_value
            chunk_tp = int(np.count_nonzero(p_mask & g_mask))
            chunk_p = int(np.count_nonzero(p_mask))
            chunk_g = int(np.count_nonzero(g_mask))

            tp += chunk_tp
            fp += chunk_p - chunk_tp
            fn += chunk_g - chunk_tp
            total += int(pred_chunk.size)

        tn = total - tp - fp - fn
        return tp, fp, fn, tn

    pred_arr = _load_image_as_array(str(prediction_path))
    gt_arr = _load_image_as_array(str(ground_truth_path))
    return compute_confusion_matrix_elements(pred_arr, gt_arr, label_value)



def _supported_metrics_message() -> str:
    return ", ".join(sorted(METRIC_REGISTRY.keys()))


def _unsupported_metrics_warning(unsupported_metrics: list[str]) -> str:
    quoted = ", ".join(f"'{metric_name}'" for metric_name in unsupported_metrics)
    return (
        f"The following requested metrics are not implemented and were skipped: {quoted}. "
        f"Supported metrics: {_supported_metrics_message()}."
    )


def _normalize_metrics(metrics: list[str]) -> tuple[list[str], list[str]]:
    if not metrics:
        raise ValueError("metrics must contain at least one supported metric name.")

    normalized: list[str] = []
    unsupported: list[str] = []
    for metric_name in metrics:
        key = str(metric_name).strip().lower()
        if key not in METRIC_REGISTRY:
            if key and key not in unsupported:
                unsupported.append(key)
            continue
        if key not in normalized:
            normalized.append(key)

    if not normalized:
        raise ValueError(
            "None of the requested metrics are implemented. "
            + _unsupported_metrics_warning(unsupported)
        )

    return normalized, unsupported



def _normalize_class_map(class_map: dict[int, str]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_label, raw_name in class_map.items():
        label = int(raw_label)
        if label < 0:
            raise ValueError("class_map labels must be non-negative integers.")
        normalized[label] = str(raw_name)
    return normalized



def _build_mode_and_pairs(
    prediction_path: str | None,
    ground_truth_path: str | None,
    prediction_files: list[str] | None,
    ground_truth_files: list[str] | None,
) -> tuple[str, list[tuple[str, str]]]:
    single_requested = prediction_path is not None or ground_truth_path is not None
    batch_requested = prediction_files is not None or ground_truth_files is not None

    if single_requested and batch_requested:
        raise ValueError(
            "Provide either single-file inputs (prediction_path, ground_truth_path) or batch inputs "
            "(prediction_files, ground_truth_files), not both."
        )

    if single_requested:
        if not prediction_path or not ground_truth_path:
            raise ValueError("Single-file mode requires both prediction_path and ground_truth_path.")
        return "single", [(prediction_path, ground_truth_path)]

    if batch_requested:
        if not prediction_files or not ground_truth_files:
            raise ValueError("Batch mode requires both prediction_files and ground_truth_files.")
        if len(prediction_files) != len(ground_truth_files):
            raise ValueError(
                f"Batch mismatch: {len(prediction_files)} prediction files vs {len(ground_truth_files)} ground-truth files."
            )
        if len(prediction_files) == 0:
            raise ValueError("Batch mode requires at least one file pair.")
        return "batch", list(zip(prediction_files, ground_truth_files))

    raise ValueError(
        "Provide either prediction_path and ground_truth_path for single mode, or prediction_files and ground_truth_files for batch mode."
    )



def _process_single_pair(
    prediction_path: str,
    ground_truth_path: str,
    metrics: list[str],
    class_map: dict[int, str],
    session_path: str | None = None,
) -> dict[str, Any]:
    resolved_prediction = _resolve_path(prediction_path, session_path=session_path)
    resolved_ground_truth = _resolve_path(ground_truth_path, session_path=session_path)
    _validate_distinct_pair(resolved_prediction, resolved_ground_truth)
    shape, _ = _load_pair_metadata(resolved_prediction, resolved_ground_truth)

    gt_labels = _collect_labels_streaming(resolved_ground_truth)
    pred_labels = _collect_labels_streaming(resolved_prediction)
    _validate_mask_like_labels(resolved_ground_truth, gt_labels, class_map, "Ground truth")
    _validate_mask_like_labels(resolved_prediction, pred_labels, class_map, "Prediction")
    hallucinated_labels = [label for label in pred_labels if label not in gt_labels]

    results: dict[str, Any] = {}
    ordered_labels = gt_labels + hallucinated_labels

    for label in ordered_labels:
        class_name = class_map.get(label, f"class_{label}")
        if label in hallucinated_labels and label not in gt_labels and label not in class_map:
            class_name = f"class_{label}_hallucinated"

        tp, fp, fn, tn = _compute_confusion_streaming(resolved_prediction, resolved_ground_truth, label)
        class_metrics: dict[str, Any] = {
            metric_name: round(float(METRIC_REGISTRY[metric_name](tp, fp, fn, tn)), 4)
            for metric_name in metrics
        }
        class_metrics["_counts"] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }
        results[class_name] = class_metrics

    return {
        "prediction_path": str(resolved_prediction),
        "ground_truth_path": str(resolved_ground_truth),
        "shape": list(shape),
        "active_ground_truth_labels": gt_labels,
        "hallucinated_prediction_labels": hallucinated_labels,
        "metrics": results,
    }



def _build_summary(
    detailed_results: list[dict[str, Any]],
    metrics: list[str],
    mode: str,
    unsupported_metrics: list[str] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested_metrics": metrics,
        "unsupported_metrics": unsupported_metrics or [],
        "successful_pairs": sum(1 for item in detailed_results if item["status"] == "success"),
        "failed_pairs": sum(1 for item in detailed_results if item["status"] == "error"),
        "classes": {},
    }

    if mode != "batch":
        return summary

    accumulator: dict[str, dict[str, list[float]]] = {}

    for item in detailed_results:
        if item["status"] != "success":
            continue
        for class_name, class_values in item["result"]["metrics"].items():
            accumulator.setdefault(class_name, {metric_name: [] for metric_name in metrics})
            for metric_name in metrics:
                accumulator[class_name][metric_name].append(float(class_values[metric_name]))

    for class_name, metric_values in accumulator.items():
        summary["classes"][class_name] = {}
        for metric_name, values in metric_values.items():
            if values:
                summary["classes"][class_name][metric_name] = round(float(np.mean(values)), 4)
                summary["classes"][class_name][f"{metric_name}_std"] = round(float(np.std(values)), 4)

    return summary



def calculate_segmentation_metrics_mcp(
    prediction_path: str | None = None,
    ground_truth_path: str | None = None,
    prediction_files: list[str] | None = None,
    ground_truth_files: list[str] | None = None,
    metrics: list[str] | None = None,
    class_map: dict[int, str] | None = None,
    session_path: str | None = None,
) -> dict[str, Any]:
    """
    Compute segmentation quality metrics by comparing predicted masks against ground truth masks.

    Use this tool when you need to evaluate how well a segmentation matches its reference,
    either for one prediction/ground-truth pair or for an aligned batch of pairs. It
    rejects pairs where the prediction and ground truth resolve to the same file, because
    self-comparison is not a valid quality evaluation. It works at segmentation-evaluation
    level and returns class-wise quality scores that support
    model assessment, result validation, and comparison across segmentation outputs.
    """
    try:
        normalized_metrics, unsupported_metrics = _normalize_metrics(
            metrics if metrics is not None else list(DEFAULT_METRICS)
        )
        normalized_class_map = _normalize_class_map(class_map if class_map is not None else dict(DEFAULT_CLASS_MAP))
        mode, pairs = _build_mode_and_pairs(
            prediction_path=prediction_path,
            ground_truth_path=ground_truth_path,
            prediction_files=prediction_files,
            ground_truth_files=ground_truth_files,
        )

        detailed_results: list[dict[str, Any]] = []
        for pred_path, gt_path in pairs:
            pair_label = Path(pred_path).name
            try:
                result = _process_single_pair(
                    pred_path,
                    gt_path,
                    normalized_metrics,
                    normalized_class_map,
                    session_path=session_path,
                )
                detailed_results.append(
                    {
                        "file": pair_label,
                        "status": "success",
                        "result": result,
                    }
                )
            except Exception as exc:
                detailed_results.append(
                    {
                        "file": pair_label,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        if not any(item["status"] == "success" for item in detailed_results):
            errors = "; ".join(item["error"] for item in detailed_results if item["status"] == "error")
            return _fail(f"No valid file pairs were processed. {errors}")

        return {
            "success": True,
            "tool_kind": TOOL_KIND,
            "mode": mode,
            "requested_metrics": normalized_metrics,
            "unsupported_metrics": unsupported_metrics,
            "warnings": (
                [_unsupported_metrics_warning(unsupported_metrics)]
                if unsupported_metrics
                else []
            ),
            "class_map": normalized_class_map,
            "total_pairs": len(pairs),
            "detailed_results": detailed_results,
            "summary": _build_summary(
                detailed_results,
                normalized_metrics,
                mode,
                unsupported_metrics=unsupported_metrics,
            ),
            "attachments": [],
        }
    except Exception as exc:
        logger.exception("Segmentation metrics computation failed")
        return _fail(str(exc))


class SegmentationMetricsTool(BaseTool):
    name: str = TOOL_NAME
    description: str = ROUTING_DESCRIPTION
    args_schema: type[BaseModel] = SegmentationMetricsArgs

    def _run(
        self,
        prediction_path: str | None = None,
        ground_truth_path: str | None = None,
        prediction_files: list[str] | None = None,
        ground_truth_files: list[str] | None = None,
        metrics: list[str] | None = None,
        class_map: dict[int, str] | None = None,
        session_path: str | None = None,
    ) -> dict[str, Any]:
        return calculate_segmentation_metrics_mcp(
            prediction_path=prediction_path,
            ground_truth_path=ground_truth_path,
            prediction_files=prediction_files,
            ground_truth_files=ground_truth_files,
            metrics=metrics,
            class_map=class_map,
            session_path=session_path,
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


EXPORTED_TOOLS: dict[str, BaseTool] = {
    TOOL_NAME: SegmentationMetricsTool(),
}


def register_mcp_tools(mcp_server: Any) -> None:
    """Register the plain MCP-callable function on an already existing MCP server instance."""
    if not hasattr(mcp_server, "tool"):
        raise TypeError("mcp_server must expose a .tool() decorator-compatible registration API.")
    mcp_server.tool()(calculate_segmentation_metrics_mcp)
