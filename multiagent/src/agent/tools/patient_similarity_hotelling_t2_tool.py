import json
import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from agent.tools.path_resolution import missing_input_path_message, resolve_input_path

logger = logging.getLogger(__name__)

EPSILON = 1e-12
TOOL_KIND = "block_similarity"
TOOL_NAME = "find_similar_patients_by_block_hotelling_t2"

DEFAULT_MIN_REF_N = 5
DEFAULT_MIN_QUERY_N = 5
CACHE_SCHEMA_VERSION = 1
CACHE_FILE_NAME = ".knn_cache.json"
DEFAULT_FEATURE_POLICY_ID = "default_22_morphometry_v1"
DEFAULT_SHRINKAGE = 0.05
DEFAULT_EXPECTED_BLOCK_DEPTH_PX = 70

DEFAULT_EXCLUDE_COLUMNS = [
    "Lacunar ID",
    "Scan Width (px)",
    "Scan Height (px)",
    "Block Depth (px)",
    "Total Number of Lacunae",
]
DEFAULT_FEATURE_COLS = [
    "Centroid X",
    "Centroid Y",
    "Centroid Z",
    "Lacuna BBox Size X (um)",
    "Lacuna BBox Size Y (um)",
    "Lacuna BBox Size Z (um)",
    "Major Axis (um) (radius) (from PCA on volume)",
    "Minor Axis (um) (radius) (from PCA on volume)",
    "Minor to Major Axis Ratio",
    "Surface Area (um^2)",
    "Surface Area to Vol Ratio (raw)",
    "Surface Area to Vol Ratio (um)",
    "Volume (um^3)",
    "Voxel Size (mm)",
    "index Lc.Or1_x",
    "index Lc.Or1_y",
    "index Lc.Or1_z",
    "index Lc.Or2_x",
    "index Lc.Or2_y",
    "index Lc.Or2_z",
    "index Lc_Ob",
    "index Lc_St",
]


class SimilarPatientsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_block_csv: str = Field(
        description=(
            "Path to the query block CSV containing the raw per-lacuna rows for the block to classify. "
            "It must contain the same feature columns declared in the reference manifest."
        )
    )
    reference_library: str = Field(
        description=(
            "Path to an external writable morphometry reference library. The repository does not "
            "include reference data. Its required layout is documented in "
            "multiagent/KNN_REFERENCE_LIBRARY.md."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of nearest reference blocks to retain for ranking, patient summary, and block class prediction."
        ),
    )
    exclude_self_match: bool = Field(
        default=False,
        description="Exclude the reference row that is the same block as the query.",
    )
    exclude_reference_id: str | None = Field(
        default=None,
        description="Optional reference block id to exclude from ranking.",
    )
    reference_filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional metadata filters for reference_library entries. Use this to restrict neighbors before ranking; "
            "all keys are ANDed, and list values mean allowed values. Example: "
            "{'section_name': 'T2', 'slice_start': 989, 'slice_end': 1059, 'step': 0}."
        ),
    )
    exclude_patient_id: str | None = Field(
        default=None,
        description=(
            "Optional exact patient_id to exclude before ranking. For LOO, use the full query patient_id "
            "before '_step' (e.g. T1_S26), never a prefix like T1."
        ),
    )
    session_path: str | None = Field(
        default=None,
        description="Optional active session directory used to resolve relative query/reference paths.",
    )


DESCRIPTION = (
    "Compare a query morphometry block against a labeled reference block library using Hotelling T^2 similarity, "
    "retrieve the nearest reference blocks, summarize the most supported patients among the nearest matches, and "
    "predict the most likely class of the query block from those nearest labeled references. The tool is intended "
    "for block-level retrieval and block classification workflows. reference_library is required: "
    "OsteoPilot never includes a reference library or patient data. See "
    "multiagent/KNN_REFERENCE_LIBRARY.md for the required external-library layout.\n\n"
    "How to read the output: top_block_matches is the primary block-level ranking, ordered from the most similar "
    "reference block to the least similar within the retained top-k. patient_summary is an aggregation over those "
    "retained top-k block matches, with one row per patient represented among them. class_inference is the final "
    "block class prediction derived from the retained nearest labeled reference blocks. For each retained block, "
    "top3_different_features compares query means against the stored reference mean vector using the active "
    "feature-policy order.\n\n"
    "For final answers, prefer answer_ready_summary for common fields. Use the full payload only if the requested "
    "field is absent. For patient-level similarity, use answer_ready_summary.best_patient.weighted_score from "
    "patient_summary, not block-level similarity_weight from top_block_matches."
)


class FindSimilarPatientsTool(BaseTool):
    name: str = TOOL_NAME
    description: str = DESCRIPTION
    args_schema: type[BaseModel] = SimilarPatientsArgs

    def _run(
        self,
        query_block_csv: str,
        reference_library: str,
        top_k: int = 5,
        exclude_self_match: bool = False,
        exclude_reference_id: str | None = None,
        reference_filters: dict[str, Any] | None = None,
        exclude_patient_id: str | None = None,
        session_path: str | None = None,
    ) -> dict[str, Any]:
        return find_similar_patients_by_block_hotelling_t2(
            query_block_csv=query_block_csv,
            reference_library=reference_library,
            top_k=top_k,
            exclude_self_match=exclude_self_match,
            exclude_reference_id=exclude_reference_id,
            reference_filters=reference_filters,
            exclude_patient_id=exclude_patient_id,
            session_path=session_path,
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution not supported.")


def _error_payload(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "tool_kind": TOOL_KIND,
        "error": message,
        "attachments": [],
    }


def _query_block_id_from_path(query_block_csv: str) -> str:
    return Path(query_block_csv).stem


def _query_patient_id_from_block_id(block_id: str) -> str | None:
    if "_step" not in block_id:
        return None
    patient_id = block_id.split("_step", 1)[0].strip()
    return patient_id or None


def _require_existing_file(
    path: str,
    label: str,
    *,
    session_path: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(path, str) or not path.strip():
        return _error_payload(f"{label} must be a non-empty string path."), None

    resolved = resolve_input_path(path, session_path=session_path)
    if resolved.path is None:
        return (
            _error_payload(
                missing_input_path_message(label, path, resolved.searched_paths)
            ),
            None,
        )
    return None, str(resolved.path)


def _require_existing_dir(
    path: str,
    label: str,
    *,
    session_path: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(path, str) or not path.strip():
        return _error_payload(f"{label} must be a non-empty string path."), None

    resolved = resolve_input_path(path, session_path=session_path)
    if resolved.path is None:
        return (
            _error_payload(
                missing_input_path_message(label, path, resolved.searched_paths)
            ),
            None,
        )
    if not resolved.path.is_dir():
        return _error_payload(f"{label} is not a directory: {resolved.path}"), None
    return None, str(resolved.path)


def _load_feature_policy(library_dir: str | None = None) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "feature_policy_id": DEFAULT_FEATURE_POLICY_ID,
        "feature_cols": list(DEFAULT_FEATURE_COLS),
        "exclude_columns": list(DEFAULT_EXCLUDE_COLUMNS),
        "shrinkage": DEFAULT_SHRINKAGE,
        "expected_block_depth_px": DEFAULT_EXPECTED_BLOCK_DEPTH_PX,
    }

    if library_dir:
        policy_path = Path(library_dir) / "feature_policy.json"
        if policy_path.exists():
            with policy_path.open("r", encoding="utf-8") as handle:
                user_policy = json.load(handle)
            if not isinstance(user_policy, dict):
                raise ValueError("feature_policy.json must contain a JSON object.")
            policy.update({key: value for key, value in user_policy.items() if value is not None})

    feature_cols = policy.get("feature_cols")
    if not isinstance(feature_cols, list) or not all(isinstance(column, str) and column for column in feature_cols):
        raise ValueError("Feature policy must define a non-empty feature_cols list.")
    policy["feature_cols"] = list(feature_cols)
    policy["shrinkage"] = float(policy.get("shrinkage", DEFAULT_SHRINKAGE))
    expected_depth = policy.get("expected_block_depth_px")
    policy["expected_block_depth_px"] = int(expected_depth) if expected_depth is not None else None
    if not isinstance(policy.get("feature_policy_id"), str) or not policy["feature_policy_id"].strip():
        policy["feature_policy_id"] = DEFAULT_FEATURE_POLICY_ID
    return policy


def _compute_stats_from_csv(
    csv_path: str,
    feature_cols: list[str],
    shrinkage: float,
) -> dict[str, Any]:
    feature_set = set(feature_cols)
    df = pd.read_csv(
        csv_path,
        usecols=lambda column: column in feature_set or column == "Block Depth (px)",
    )
    missing_cols = [column for column in feature_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required feature columns: {missing_cols}")

    x_values = df[feature_cols].to_numpy(dtype=np.float64, copy=True)
    valid_mask = np.isfinite(x_values).all(axis=1)
    x_values = x_values[valid_mask]
    n, p = x_values.shape
    if n <= 0:
        return {
            "n": 0,
            "p": p,
            "block_depth_px": _extract_block_depth_px(df),
            "mean": np.array([], dtype=np.float64),
            "cov": np.array([], dtype=np.float64),
        }

    mean = x_values.mean(axis=0)
    cov = np.cov(x_values, rowvar=False, bias=False) if n > 1 else np.zeros((p, p), dtype=np.float64)
    cov = _cov_shrinkage(cov, shrinkage)
    return {
        "n": int(n),
        "p": int(p),
        "block_depth_px": _extract_block_depth_px(df),
        "mean": mean,
        "cov": cov,
    }


def _serialize_vector(values: np.ndarray) -> str:
    return ",".join(f"{value:.10g}" for value in values.tolist())


def _serialize_matrix(values: np.ndarray) -> str:
    return ",".join(f"{value:.10g}" for value in values.reshape(-1).tolist())


def _source_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"source_size": int(stat.st_size), "source_mtime_ns": int(stat.st_mtime_ns)}


def _cache_payload_matches_policy(
    cache: dict[str, Any],
    *,
    source_csv: str,
    signature: dict[str, int],
    feature_policy: dict[str, Any],
) -> bool:
    return (
        int(cache.get("schema_version", -1)) == CACHE_SCHEMA_VERSION
        and cache.get("cache_kind") == "hotelling_t2_block_stats"
        and cache.get("source_csv") == source_csv
        and int(cache.get("source_size", -1)) == int(signature["source_size"])
        and int(cache.get("source_mtime_ns", -1)) == int(signature["source_mtime_ns"])
        and cache.get("feature_policy_id") == feature_policy["feature_policy_id"]
        and list(cache.get("feature_cols", [])) == list(feature_policy["feature_cols"])
        and float(cache.get("shrinkage", -1.0)) == float(feature_policy["shrinkage"])
        and cache.get("expected_block_depth_px") == feature_policy.get("expected_block_depth_px")
    )


def _load_or_build_sample_cache(
    *,
    sample_dir: Path,
    morphometry_csv: Path,
    feature_policy: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    cache_path = sample_dir / CACHE_FILE_NAME
    source_csv = os.path.relpath(morphometry_csv, sample_dir)
    signature = _source_signature(morphometry_csv)

    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                cache = json.load(handle)
            if isinstance(cache, dict) and _cache_payload_matches_policy(
                cache,
                source_csv=source_csv,
                signature=signature,
                feature_policy=feature_policy,
            ):
                return cache, True
        except Exception:
            pass

    feature_cols = list(feature_policy["feature_cols"])
    stats = _compute_stats_from_csv(
        str(morphometry_csv),
        feature_cols=feature_cols,
        shrinkage=float(feature_policy["shrinkage"]),
    )
    cache = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_kind": "hotelling_t2_block_stats",
        "source_csv": source_csv,
        **signature,
        "feature_policy_id": feature_policy["feature_policy_id"],
        "feature_cols": feature_cols,
        "shrinkage": float(feature_policy["shrinkage"]),
        "expected_block_depth_px": feature_policy.get("expected_block_depth_px"),
        "n": int(stats["n"]),
        "p": int(stats["p"]),
        "block_depth_px": stats.get("block_depth_px"),
        "mean_flat": _serialize_vector(stats["mean"]),
        "cov_flat": _serialize_matrix(stats["cov"]),
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, cache_path)
    return cache, False


def _discover_morphometry_library(library_dir: str) -> list[dict[str, Any]]:
    root = Path(library_dir).resolve()
    entries: list[dict[str, Any]] = []
    for sample_dir in sorted(child for child in root.iterdir() if child.is_dir()):
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata.json must contain an object: {metadata_path}")

        sample_identifier = metadata.get("sample_identifier")
        if not isinstance(sample_identifier, str) or not sample_identifier.strip():
            raise ValueError(f"metadata.json missing sample_identifier: {metadata_path}")
        if sample_identifier != sample_dir.name:
            raise ValueError(
                f"sample_identifier mismatch for {metadata_path}: "
                f"metadata has {sample_identifier!r}, folder is {sample_dir.name!r}"
            )

        patient_id = metadata.get("patient_id")
        class_label = metadata.get("class_label") or metadata.get("disease")
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError(f"metadata.json missing patient_id: {metadata_path}")
        if not isinstance(class_label, str) or not class_label.strip():
            raise ValueError(f"metadata.json missing class_label: {metadata_path}")

        for key in ("step", "z_index", "slice_start", "slice_end"):
            if not isinstance(metadata.get(key), int):
                raise ValueError(f"metadata.json field {key} must be an integer: {metadata_path}")
        if int(metadata["slice_start"]) >= int(metadata["slice_end"]):
            raise ValueError(f"metadata.json slice_start must be lower than slice_end: {metadata_path}")

        csv_rel = metadata.get("morphometry_csv", "morphometry.csv")
        if not isinstance(csv_rel, str) or not csv_rel.strip():
            raise ValueError(f"metadata.json morphometry_csv must be a path string: {metadata_path}")
        morphometry_csv = (sample_dir / csv_rel).resolve()
        if sample_dir.resolve() not in morphometry_csv.parents and morphometry_csv != sample_dir.resolve():
            raise ValueError(f"morphometry_csv must resolve inside the sample folder: {metadata_path}")
        if not morphometry_csv.exists() or not morphometry_csv.is_file():
            raise ValueError(f"morphometry_csv not found for {sample_identifier}: {morphometry_csv}")

        entries.append(
            {
                "sample_identifier": sample_identifier,
                "sample_dir": sample_dir,
                "metadata_path": metadata_path,
                "morphometry_csv": morphometry_csv,
                "metadata": metadata,
                "patient_id": patient_id,
                "class_label": class_label,
            }
        )
    return entries


def _extract_block_depth_px(df: pd.DataFrame) -> int | None:
    column = "Block Depth (px)"
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return int(round(float(values.iloc[0])))


def _cov_shrinkage(covariance: np.ndarray, lam: float) -> np.ndarray:
    p = covariance.shape[0]
    if p == 0:
        return covariance
    mu = float(np.trace(covariance) / p)
    return (1.0 - lam) * covariance + lam * mu * np.eye(p, dtype=covariance.dtype)


def _parse_flat_vector(serialized: str, expected_len: int) -> np.ndarray:
    values = np.fromstring(serialized, sep=",", dtype=np.float64)
    if values.size != expected_len:
        raise ValueError(f"Bad mean length: got {values.size}, expected {expected_len}")
    return values


def _parse_flat_matrix(serialized: str, p: int) -> np.ndarray:
    values = np.fromstring(serialized, sep=",", dtype=np.float64)
    if values.size != p * p:
        raise ValueError(f"Bad cov length: got {values.size}, expected {p * p}")
    return values.reshape((p, p))


def hotelling_t2_from_stats(
    n1: int,
    m1: np.ndarray,
    s1: np.ndarray,
    n2: int,
    m2: np.ndarray,
    s2: np.ndarray,
) -> float:
    p = m1.size
    if m2.size != p:
        raise ValueError("Mean dimension mismatch")
    if s1.shape != (p, p) or s2.shape != (p, p):
        raise ValueError("Cov dimension mismatch")

    delta = (m1 - m2).reshape(-1, 1)
    denominator = n1 + n2 - 2
    if denominator <= 0:
        pooled = np.zeros((p, p), dtype=np.float64)
    else:
        pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / denominator

    pooled_inv = np.linalg.pinv(pooled)
    scale = (n1 * n2) / (n1 + n2) if (n1 + n2) > 0 else 0.0
    return float(scale * (delta.T @ pooled_inv @ delta).squeeze())


def _build_feature_mean_differences(
    feature_cols: list[str],
    query_mean: np.ndarray,
    reference_mean: np.ndarray,
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_cols):
        query_value = float(query_mean[index])
        reference_value = float(reference_mean[index])
        abs_diff = abs(query_value - reference_value)
        if not all(math.isfinite(value) for value in (query_value, reference_value, abs_diff)):
            continue
        differences.append(
            {
                "feature": feature,
                "query_mean": round(query_value, 4),
                "reference_mean": round(reference_value, 4),
                "abs_diff": round(abs_diff, 4),
            }
        )
    differences.sort(key=lambda item: (-float(item["abs_diff"]), item["feature"]))
    return differences[:top_n]


def _build_patient_summary(top_matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for match in top_matches:
        patient = match.get("patient") or "<unknown_patient>"
        entry = grouped.setdefault(
            patient,
            {
                "patient": patient,
                "match_count": 0,
                "best_t2": None,
                "weighted_score": 0.0,
                "best_block": None,
                "class_counts": {},
            },
        )
        entry["match_count"] += 1
        entry["weighted_score"] += float(match["similarity_weight"])
        t2 = float(match["t2"])
        if entry["best_t2"] is None or t2 < entry["best_t2"]:
            entry["best_t2"] = t2
            entry["best_block"] = match.get("reference_block_id")

        class_label = match.get("class_label")
        if class_label is not None:
            counts = entry["class_counts"]
            counts[class_label] = counts.get(class_label, 0) + 1

    patient_summary: list[dict[str, Any]] = []
    for entry in grouped.values():
        raw_counts = entry.pop("class_counts")
        entry["class_counts"] = [
            {"class_label": class_label, "count": count}
            for class_label, count in sorted(raw_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        entry["weighted_score"] = round(float(entry["weighted_score"]), 8)
        patient_summary.append(entry)

    patient_summary.sort(key=lambda item: (-item["weighted_score"], item["best_t2"], item["patient"]))
    return patient_summary


def _build_class_inference(top_matches: list[dict[str, Any]]) -> dict[str, Any]:
    labeled_matches = [match for match in top_matches if match.get("class_label") is not None]
    if not labeled_matches:
        return {
            "available": False,
            "predicted_class": None,
            "reason": "No class label column found in the reference data for the selected top matches.",
            "decision_rule": None,
            "weight_formula": None,
            "confidence": None,
            "class_votes": [],
        }

    aggregated: dict[str, dict[str, Any]] = {}
    for match in labeled_matches:
        class_label = str(match["class_label"])
        entry = aggregated.setdefault(
            class_label,
            {"class_label": class_label, "count": 0, "weight": 0.0, "best_t2": None},
        )
        entry["count"] += 1
        entry["weight"] += float(match["similarity_weight"])
        t2 = float(match["t2"])
        if entry["best_t2"] is None or t2 < entry["best_t2"]:
            entry["best_t2"] = t2

    class_votes = list(aggregated.values())
    class_votes.sort(key=lambda item: (-item["weight"], -item["count"], item["best_t2"], item["class_label"]))
    total_weight = sum(vote["weight"] for vote in class_votes)
    confidence = (class_votes[0]["weight"] / total_weight) if total_weight > 0 else None

    return {
        "available": True,
        "predicted_class": class_votes[0]["class_label"],
        "reason": "Computed from a weighted vote over the nearest reference blocks.",
        "decision_rule": "weighted_vote_over_top_block_matches",
        "weight_formula": "1 / (t2 + 1e-12)",
        "confidence": round(float(confidence), 8) if confidence is not None else None,
        "class_votes": [
            {
                "class_label": vote["class_label"],
                "count": int(vote["count"]),
                "weight": round(float(vote["weight"]), 8),
                "best_t2": float(vote["best_t2"]),
            }
            for vote in class_votes
        ],
    }


def _class_label_from_patient_entry(entry: dict[str, Any]) -> str | None:
    class_counts = entry.get("class_counts") or []
    if len(class_counts) != 1:
        return None
    class_label = class_counts[0].get("class_label")
    return str(class_label) if class_label is not None else None


def _project_block_match(match: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rank",
        "reference_block_id",
        "reference_relpath",
        "patient",
        "patient_id",
        "class_label",
        "t2",
        "hotelling_t2_distance",
        "similarity_weight",
        "reference_n",
        "step",
        "z_index",
        "slice_start",
        "slice_end",
        "section_name",
        "reference_metadata",
        "top3_different_features",
    ]
    return {key: match[key] for key in keys if key in match}


def _build_answer_ready_summary(
    *,
    query_block_csv: str,
    query_block_id: str,
    query_patient_id: str | None,
    top_block_matches: list[dict[str, Any]],
    patient_summary: list[dict[str, Any]],
    class_inference: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    nearest_block = _project_block_match(top_block_matches[0]) if top_block_matches else None
    best_patient = None
    if patient_summary:
        patient_entry = patient_summary[0]
        best_patient = {
            "patient": patient_entry.get("patient"),
            "patient_id": patient_entry.get("patient"),
            "class_label": _class_label_from_patient_entry(patient_entry),
            "match_count": patient_entry.get("match_count"),
            "best_t2": patient_entry.get("best_t2"),
            "weighted_score": patient_entry.get("weighted_score"),
            "best_block": patient_entry.get("best_block"),
            "class_counts": patient_entry.get("class_counts", []),
            "weighted_score_source": "patient_summary[0].weighted_score",
        }

    return {
        "query": {
            "query_block_csv": query_block_csv,
            "query_block_id": query_block_id,
            "query_patient_id": query_patient_id,
            "top_k": top_k,
        },
        "classification": {
            "available": class_inference.get("available"),
            "predicted_class": class_inference.get("predicted_class"),
            "confidence": class_inference.get("confidence"),
            "class_votes": class_inference.get("class_votes", []),
            "decision_rule": class_inference.get("decision_rule"),
            "weight_formula": class_inference.get("weight_formula"),
        },
        "nearest_block": nearest_block,
        "best_patient": best_patient,
        "top_blocks_preview": [
            _project_block_match(match)
            for match in top_block_matches[: min(5, len(top_block_matches))]
        ],
        "feature_differences_rank1": (
            top_block_matches[0].get("top3_different_features", [])
            if top_block_matches
            else []
        ),
        "notes": {
            "projection_only": True,
            "full_precision_payload_fields": [
                "top_block_matches",
                "patient_summary",
                "class_inference",
            ],
            "patient_weighted_score_rule": (
                "Use best_patient.weighted_score for patient-level similarity; similarity_weight is block-level."
            ),
        },
    }


def _is_self_reference_match(
    *,
    query_path: str,
    reference_relpath: str,
    reference_abs_path: str | None,
) -> bool:
    query = Path(query_path).expanduser()
    query_abs = query.resolve()

    if reference_abs_path:
        try:
            if Path(reference_abs_path).expanduser().resolve() == query_abs:
                return True
        except Exception:
            pass

    rel = Path(reference_relpath)
    return query.name == rel.name or query_abs.as_posix().endswith(rel.as_posix())


def _matches_excluded_reference_id(match: dict[str, Any], excluded_id: str | None) -> bool:
    if not excluded_id:
        return False
    excluded = excluded_id.strip()
    if not excluded:
        return False
    candidates = [
        match.get("reference_block_id"),
        match.get("reference_relpath"),
        Path(str(match.get("reference_relpath", ""))).name,
    ]
    return any(str(candidate) == excluded for candidate in candidates if candidate is not None)


def _metadata_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set)):
        return any(_metadata_value_matches(actual, item) for item in expected)
    if actual is None:
        return expected is None
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return str(actual) == str(expected)


def _entry_matches_reference_filters(entry: dict[str, Any], reference_filters: dict[str, Any] | None) -> bool:
    if not reference_filters:
        return True
    metadata = entry.get("metadata") or {}
    for key, expected in reference_filters.items():
        if key in metadata:
            actual = metadata.get(key)
        else:
            actual = entry.get(key)
        if not _metadata_value_matches(actual, expected):
            return False
    return True


def _metadata_for_match(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry.get("metadata") or {}
    keys = [
        "sample_identifier",
        "patient_id",
        "class_label",
        "step",
        "z_index",
        "slice_start",
        "slice_end",
        "section_name",
        "block_depth_px",
    ]
    output = {key: metadata[key] for key in keys if key in metadata}
    output.setdefault("sample_identifier", entry.get("sample_identifier"))
    output.setdefault("patient_id", entry.get("patient_id"))
    output.setdefault("class_label", entry.get("class_label"))
    return {key: value for key, value in output.items() if value is not None}


def _patient_id_matches(actual: Any, excluded_patient_id: str | None) -> bool:
    if excluded_patient_id is None:
        return False
    excluded = str(excluded_patient_id).strip()
    return bool(excluded) and str(actual) == excluded


def _success_payload(
    *,
    query_block_csv: str,
    top_k: int,
    top_block_matches: list[dict[str, Any]],
    patient_summary: list[dict[str, Any]],
    class_inference: dict[str, Any],
    metadata: dict[str, Any],
    total_ranked_count: int,
) -> dict[str, Any]:
    query_block_id = _query_block_id_from_path(query_block_csv)
    query_patient_id = _query_patient_id_from_block_id(query_block_id)
    answer_ready_summary = _build_answer_ready_summary(
        query_block_csv=query_block_csv,
        query_block_id=query_block_id,
        query_patient_id=query_patient_id,
        top_block_matches=top_block_matches,
        patient_summary=patient_summary,
        class_inference=class_inference,
        top_k=top_k,
    )
    return {
        "success": True,
        "tool_kind": TOOL_KIND,
        "message": (
            f"Ranked {total_ranked_count} usable reference blocks and retained the top {len(top_block_matches)} "
            "nearest block matches for patient summary and class inference."
        ),
        "query_block_csv": query_block_csv,
        "query_block_id": query_block_id,
        "query_patient_id": query_patient_id,
        "top_k": top_k,
        "top_block_matches": top_block_matches,
        "rank1_top3_different_features": (
            top_block_matches[0].get("top3_different_features", [])
            if top_block_matches
            else []
        ),
        "patient_summary": patient_summary,
        "class_inference": class_inference,
        "answer_ready_summary": answer_ready_summary,
        "metadata": metadata,
        "attachments": [],
    }


def _persist_tool_payload_artifact(
    payload: dict[str, Any],
    *,
    session_path: str | None,
    query_block_csv: str,
) -> dict[str, Any]:
    if not session_path:
        return payload

    session_root = Path(session_path).expanduser()
    output_dir = session_root / "session_output" / "tool_payloads" / TOOL_NAME
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        query_stem = Path(query_block_csv).stem or "query"
        safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in query_stem)
        output_path = output_dir / f"{safe_stem}__{uuid.uuid4().hex[:8]}.json"
        artifact_payload = dict(payload)
        artifact_payload["attachments"] = []
        output_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to persist %s payload artifact: %s", TOOL_NAME, exc)
        return payload

    attachments = list(payload.get("attachments") or [])
    attachments.append(
        {
            "path": str(output_path),
            "kind": "tool_payload_json",
            "description": "Full JSON payload returned by find_similar_patients_by_block_hotelling_t2.",
        }
    )
    payload["attachments"] = attachments
    payload["payload_artifact_path"] = str(output_path)
    return payload


def _run_against_morphometry_library(
    *,
    query_block_csv: str,
    reference_library: str,
    top_k: int,
    exclude_self_match: bool,
    exclude_reference_id: str | None,
    reference_filters: dict[str, Any] | None,
    exclude_patient_id: str | None,
    session_path: str | None,
) -> dict[str, Any]:
    error, resolved_query_block_csv = _require_existing_file(
        query_block_csv,
        "query_block_csv",
        session_path=session_path,
    )
    if error is not None:
        return error

    error, resolved_reference_library = _require_existing_dir(
        reference_library,
        "reference_library",
        session_path=session_path,
    )
    if error is not None:
        return error
    assert resolved_reference_library is not None

    feature_policy = _load_feature_policy(resolved_reference_library)
    feature_cols = list(feature_policy["feature_cols"])
    shrinkage = float(feature_policy["shrinkage"])
    expected_block_depth_px = feature_policy.get("expected_block_depth_px")
    p = len(feature_cols)

    try:
        query_stats = _compute_stats_from_csv(
            resolved_query_block_csv or query_block_csv,
            feature_cols=feature_cols,
            shrinkage=shrinkage,
        )
    except ValueError as exc:
        message = str(exc)
        if "missing required feature columns" in message.lower():
            return _error_payload(message.replace("CSV missing", "Query missing"))
        return _error_payload(message)

    query_block_depth_px = query_stats.get("block_depth_px")
    if expected_block_depth_px is not None and query_block_depth_px is not None:
        if int(query_block_depth_px) != int(expected_block_depth_px):
            return _error_payload(
                "Query block_depth_px mismatch: "
                f"got {query_block_depth_px}, expected {int(expected_block_depth_px)}"
            )

    query_n = int(query_stats["n"])
    if query_n < DEFAULT_MIN_QUERY_N:
        return _error_payload(
            f"Query has too few valid rows: n={query_n} (min_query_n={DEFAULT_MIN_QUERY_N})"
        )
    query_mean = query_stats["mean"]
    query_cov = query_stats["cov"]

    library_entries = _discover_morphometry_library(resolved_reference_library)
    if not library_entries:
        return _error_payload(f"No usable morphometry samples found in reference_library: {resolved_reference_library}")

    discovered_sample_count = len(library_entries)
    reference_filter_excluded_count = 0
    if reference_filters:
        filtered_entries = []
        for entry in library_entries:
            if _entry_matches_reference_filters(entry, reference_filters):
                filtered_entries.append(entry)
            else:
                reference_filter_excluded_count += 1
        library_entries = filtered_entries
        if not library_entries:
            return _error_payload("No reference entries remain after applying reference_filters.")

    exclude_patient_id_count = 0
    if exclude_patient_id:
        filtered_entries = []
        for entry in library_entries:
            if _patient_id_matches(entry.get("patient_id"), exclude_patient_id):
                exclude_patient_id_count += 1
                continue
            filtered_entries.append(entry)
        library_entries = filtered_entries
        if not library_entries:
            return _error_payload("No reference entries remain after excluding exclude_patient_id.")

    ranked_matches: list[dict[str, Any]] = []
    cache_hits = 0
    cache_rebuilt = 0
    skipped_reference_rows = 0

    for entry_index, entry in enumerate(library_entries):
        cache, cache_hit = _load_or_build_sample_cache(
            sample_dir=entry["sample_dir"],
            morphometry_csv=entry["morphometry_csv"],
            feature_policy=feature_policy,
        )
        if cache_hit:
            cache_hits += 1
        else:
            cache_rebuilt += 1

        reference_n = int(cache["n"])
        reference_p = int(cache["p"])
        if reference_p != p:
            return _error_payload(f"Reference p mismatch at sample {entry['sample_identifier']}: ref_p={reference_p}, expected_p={p}")

        if expected_block_depth_px is not None and cache.get("block_depth_px") is not None:
            block_depth_px = int(cache["block_depth_px"])
            if block_depth_px != int(expected_block_depth_px):
                return _error_payload(
                    f"Reference block_depth_px mismatch at sample {entry['sample_identifier']}: "
                    f"got {block_depth_px}, expected {int(expected_block_depth_px)}"
                )

        if reference_n < DEFAULT_MIN_REF_N:
            skipped_reference_rows += 1
            continue

        reference_mean = _parse_flat_vector(str(cache["mean_flat"]), p)
        reference_cov = _parse_flat_matrix(str(cache["cov_flat"]), p)
        t2 = hotelling_t2_from_stats(query_n, query_mean, query_cov, reference_n, reference_mean, reference_cov)
        if not math.isfinite(t2):
            return _error_payload(f"Non-finite Hotelling T^2 at sample {entry['sample_identifier']}")

        ranked_matches.append(
            {
                "rank_source_row": int(entry_index),
                "reference_block_id": entry["sample_identifier"],
                "reference_relpath": entry["sample_identifier"],
                "reference_morphometry_csv": str(entry["morphometry_csv"]),
                "patient": str(entry["patient_id"]),
                "patient_id": str(entry["patient_id"]),
                "class_label": str(entry["class_label"]),
                "reference_metadata": _metadata_for_match(entry),
                "step": entry["metadata"].get("step"),
                "z_index": entry["metadata"].get("z_index"),
                "slice_start": entry["metadata"].get("slice_start"),
                "slice_end": entry["metadata"].get("slice_end"),
                "section_name": entry["metadata"].get("section_name"),
                "t2": float(t2),
                "hotelling_t2_distance": float(t2),
                "reference_n": reference_n,
                "top3_different_features": _build_feature_mean_differences(
                    feature_cols,
                    query_mean,
                    reference_mean,
                    top_n=3,
                ),
            }
        )

    if not ranked_matches:
        return _error_payload("No usable reference entries to compare (min_ref_n too high or empty library).")

    excluded_reference_id_count = 0
    if exclude_reference_id:
        kept_matches = []
        for match in ranked_matches:
            if _matches_excluded_reference_id(match, exclude_reference_id):
                excluded_reference_id_count += 1
                continue
            kept_matches.append(match)
        ranked_matches = kept_matches
        if not ranked_matches:
            return _error_payload("No usable reference entries remain after excluding exclude_reference_id.")

    self_match_excluded_count = 0
    if exclude_self_match:
        filtered_matches = []
        for match in ranked_matches:
            if _is_self_reference_match(
                query_path=resolved_query_block_csv or query_block_csv,
                reference_relpath=str(match["reference_relpath"]),
                reference_abs_path=match.get("reference_morphometry_csv"),
            ):
                self_match_excluded_count += 1
                continue
            filtered_matches.append(match)
        ranked_matches = filtered_matches
        if not ranked_matches:
            return _error_payload("No usable reference entries remain after excluding the self-match.")

    ranked_matches.sort(key=lambda item: (item["t2"], item["patient"], item["reference_block_id"]))
    top_matches = ranked_matches[:top_k]
    for rank, match in enumerate(top_matches, start=1):
        match["rank"] = rank
        match["similarity_weight"] = float(1.0 / (match["t2"] + EPSILON))

    patient_summary = _build_patient_summary(top_matches)
    class_inference = _build_class_inference(top_matches)
    metadata = {
        "query_valid_row_count": query_n,
        "reference_usable_block_count": len(ranked_matches),
        "reference_discovered_sample_count": discovered_sample_count,
        "reference_filtered_sample_count": len(library_entries),
        "reference_filters": reference_filters or {},
        "reference_filter_excluded_count": int(reference_filter_excluded_count),
        "reference_skipped_block_count": skipped_reference_rows,
        "feature_count": p,
        "feature_cols": feature_cols,
        "feature_policy_id": feature_policy["feature_policy_id"],
        "query_block_depth_px": query_block_depth_px,
        "shrinkage": shrinkage,
        "label_source": "metadata.json.class_label",
        "patient_source": "metadata.json.patient_id",
        "reference_join_key": "sample_identifier",
        "reference_library": resolved_reference_library,
        "reference_mode": "morphometry_library",
        "cache_file": CACHE_FILE_NAME,
        "cache_hits": cache_hits,
        "cache_rebuilt": cache_rebuilt,
        "reference_mean_encoding": "mean_flat aligned to feature_policy.feature_cols",
            "exclude_self_match": bool(exclude_self_match),
            "self_match_excluded_count": int(self_match_excluded_count),
            "exclude_reference_id": exclude_reference_id,
            "exclude_reference_id_count": int(excluded_reference_id_count),
            "exclude_patient_id": exclude_patient_id,
            "exclude_patient_id_count": int(exclude_patient_id_count),
        }
    payload = _success_payload(
        query_block_csv=resolved_query_block_csv or query_block_csv,
        top_k=top_k,
        top_block_matches=top_matches,
        patient_summary=patient_summary,
        class_inference=class_inference,
        metadata=metadata,
        total_ranked_count=len(ranked_matches),
    )
    return _persist_tool_payload_artifact(
        payload,
        session_path=session_path,
        query_block_csv=resolved_query_block_csv or query_block_csv,
    )


def find_similar_patients_by_block_hotelling_t2(
    query_block_csv: str,
    reference_library: str,
    top_k: int = 5,
    exclude_self_match: bool = False,
    exclude_reference_id: str | None = None,
    reference_filters: dict[str, Any] | None = None,
    exclude_patient_id: str | None = None,
    session_path: str | None = None,
) -> dict[str, Any]:
    """Plain importable function for direct use or MCP registration."""
    try:
        validated_args = SimilarPatientsArgs(
            query_block_csv=query_block_csv,
            reference_library=reference_library,
            top_k=top_k,
            exclude_self_match=exclude_self_match,
            exclude_reference_id=exclude_reference_id,
            reference_filters=reference_filters,
            exclude_patient_id=exclude_patient_id,
            session_path=session_path,
        )
    except Exception as exc:
        return _error_payload(f"Invalid arguments: {exc}")

    try:
        return _run_against_morphometry_library(
            query_block_csv=validated_args.query_block_csv,
            reference_library=validated_args.reference_library,
            top_k=validated_args.top_k,
            exclude_self_match=validated_args.exclude_self_match,
            exclude_reference_id=validated_args.exclude_reference_id,
            reference_filters=validated_args.reference_filters,
            exclude_patient_id=validated_args.exclude_patient_id,
            session_path=validated_args.session_path,
        )
    except ValueError as exc:
        return _error_payload(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in %s", TOOL_NAME)
        return _error_payload(f"Unexpected error: {exc}")


def register_mcp_tools(mcp_server: Any) -> None:
    """Register the plain callable into an existing MCP server without starting the server."""
    if not hasattr(mcp_server, "tool"):
        raise TypeError("mcp_server must expose a .tool decorator for registration.")

    decorator = mcp_server.tool(name=TOOL_NAME, description=DESCRIPTION)
    decorator(find_similar_patients_by_block_hotelling_t2)


EXPORTED_TOOLS = {
    TOOL_NAME: FindSimilarPatientsTool(),
}
