#!/usr/bin/env python3
"""Create Matplotlib plots for aggregate evaluation scores."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import textwrap
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_DIR.parent
DEFAULT_RUN_INDEX_CSV = TOOL_DIR / "output" / "run_index.csv"
DEFAULT_EVALUATION_DIR = TOOL_DIR / "output" / "runs"
MPLCONFIGDIR = TOOL_DIR / "output" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib  # noqa: E402  # MPLCONFIGDIR must be configured before importing Matplotlib.

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


DEFAULT_AGGREGATE_DIR = TOOL_DIR / "output" / "runs" / "aggregate_scores"
DEFAULT_AGGREGATE_JSON = DEFAULT_AGGREGATE_DIR / "aggregate_scores.json"
_osteobench_dir = os.environ.get("OSTEOBENCH_DIR")
DEFAULT_SPEC_DIR = (
    Path(_osteobench_dir).expanduser()
    if _osteobench_dir
    else PROJECT_ROOT / "osteobench"
)

DIMENSION_LABELS = {
    "A": "Routing",
    "B": "Task Judge",
    "C": "Tools",
    "D": "Artifacts",
    "E": "Final Output",
}
DIMENSION_ORDER = {dimension: index for index, dimension in enumerate(DIMENSION_LABELS)}

COLORS = {
    "blue": "#1f5a7a",
    "blue_light": "#8fb8cf",
    "green": "#2f7d5c",
    "red": "#b94b4b",
    "yellow": "#d3a22f",
    "gray": "#64707d",
    "dark_gray": "#2f343a",
    "light_gray": "#e4e8ec",
    "panel": "#f8fafb",
}

plt.rcParams.update(
    {
        "axes.edgecolor": COLORS["light_gray"],
        "axes.labelcolor": COLORS["dark_gray"],
        "axes.titlecolor": COLORS["dark_gray"],
        "axes.titlesize": 14,
        "axes.labelsize": 10,
        "font.size": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.color": COLORS["dark_gray"],
        "ytick.color": COLORS["dark_gray"],
    }
)


LABEL_OVERRIDES: dict[str, str] = {
    "data_qa": "DataQA",
    "preprocessing_qa": "PreprocessingQA",
    "segmentation_qa": "SegmentationQA",
    "phenotyping_qa": "PhenotypingQA",
    "statistics_qa": "StatisticsQA",
}


def short_name(name: str) -> str:
    return name.removesuffix(".json").replace("_qa", "").replace("_", " ")


def effective_label(json_file: str) -> str:
    """Return the display stem for a json_file, applying LABEL_OVERRIDES."""
    stem = json_file.removesuffix(".json")
    return LABEL_OVERRIDES.get(stem, stem)


def merge_label_groups(data: dict[str, Any]) -> dict[str, Any]:
    """Pool per-file rows that share the same display label per LABEL_OVERRIDES."""
    if not LABEL_OVERRIDES:
        return data

    file_buckets: dict[str, dict[str, Any]] = {}
    for row in data["per_json_file"]:
        key = f"{effective_label(row['json_file'])}.json"
        if key not in file_buckets:
            file_buckets[key] = {
                "json_file": key,
                "status": "SUCCESS",
                "n_tasks": 0, "n_passed": 0, "n_failed": 0, "n_errors": 0, "n_excluded": 0,
                "earned": 0.0, "max": 0.0, "observation_percentages": [],
            }
        b = file_buckets[key]
        b["n_tasks"] += row["n_tasks"]
        b["n_passed"] += row["n_passed"]
        b["n_failed"] += row["n_failed"]
        b["n_errors"] += row["n_errors"]
        b["n_excluded"] += row.get("n_excluded", 0)
        b["earned"] += row["earned"]
        b["max"] += row["max"]
        b["observation_percentages"].extend(row.get("observation_percentages", []))
        if row["status"] != "SUCCESS":
            b["status"] = row["status"]
    for b in file_buckets.values():
        b["percentage"] = rounded(pct(b["earned"], b["max"]))
        b["ci95_percentage"] = confidence_interval(b["observation_percentages"])
        b["n_observations"] = len(b["observation_percentages"])
        b["label"] = f"{b['earned']:.0f}/{b['max']:.0f}"
    new_per_file = list(file_buckets.values())

    def pool_section_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key = (f"{effective_label(row['json_file'])}.json", row["section_type"], row["section"])
            if key not in buckets:
                buckets[key] = {**row, "json_file": key[0], "earned": 0.0, "max": 0.0, "observation_percentages": []}
            buckets[key]["earned"] += row["earned"]
            buckets[key]["max"] += row["max"]
            buckets[key]["observation_percentages"].extend(row.get("observation_percentages", []))
        for b in buckets.values():
            b["percentage"] = rounded(pct(b["earned"], b["max"]))
            b["ci95_percentage"] = confidence_interval(b["observation_percentages"])
            b["n_observations"] = len(b["observation_percentages"])
        return list(buckets.values())

    def pool_criterion_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (f"{effective_label(row['json_file'])}.json", row["criterion_id"])
            if key not in buckets:
                buckets[key] = {**row, "json_file": key[0], "earned": 0.0, "max": 0.0, "observation_percentages": []}
            buckets[key]["earned"] += row["earned"]
            buckets[key]["max"] += row["max"]
            buckets[key]["observation_percentages"].extend(row.get("observation_percentages", []))
        for b in buckets.values():
            b["percentage"] = rounded(pct(b["earned"], b["max"]))
            b["ci95_percentage"] = confidence_interval(b["observation_percentages"])
            b["n_observations"] = len(b["observation_percentages"])
        return list(buckets.values())

    new_dimensions = pool_section_rows(data["per_json_dimension_scores"])
    new_categories = pool_section_rows(data["per_json_category_scores"])
    new_criteria = pool_criterion_rows(data["per_json_criterion_scores"])

    section_rows = new_dimensions + new_categories
    section_means: list[dict[str, Any]] = []
    for section_type, section in sorted({(r["section_type"], r["section"]) for r in section_rows}):
        matching = [r for r in section_rows if r["section_type"] == section_type and r["section"] == section]
        earned = sum(r["earned"] for r in matching)
        max_score = sum(r["max"] for r in matching)
        observations = [v for r in matching for v in r.get("observation_percentages", [])]
        section_means.append({
            "section_type": section_type, "section": section,
            "n_json_files": len(matching),
            "n_observations": len(observations),
            "pooled_earned": earned, "pooled_max": max_score,
            "pooled_percentage": rounded(pct(earned, max_score)),
            "ci95_percentage": confidence_interval(observations),
        })

    criterion_means: list[dict[str, Any]] = []
    for criterion_id in sorted({r["criterion_id"] for r in new_criteria}):
        matching = [r for r in new_criteria if r["criterion_id"] == criterion_id]
        earned = sum(r["earned"] for r in matching)
        max_score = sum(r["max"] for r in matching)
        observations = [v for r in matching for v in r.get("observation_percentages", [])]
        first = matching[0]
        criterion_means.append({
            "criterion_id": criterion_id, "criterion_label": first["criterion_label"],
            "dimension": first["dimension"], "category": first["category"],
            "n_json_files": len(matching),
            "n_observations": len(observations),
            "pooled_earned": earned, "pooled_max": max_score,
            "pooled_percentage": rounded(pct(earned, max_score)),
            "ci95_percentage": confidence_interval(observations),
        })

    return {
        **data,
        "per_json_file": new_per_file,
        "per_json_dimension_scores": new_dimensions,
        "per_json_category_scores": new_categories,
        "per_json_criterion_scores": new_criteria,
        "section_means_across_json_files": section_means,
        "criterion_means_across_json_files": criterion_means,
    }


def wrap_labels(labels: list[str], width: int = 18) -> list[str]:
    return ["\n".join(textwrap.wrap(label, width=width, break_long_words=False)) for label in labels]


def save(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_percent_labels(
    ax: plt.Axes,
    values: list[float],
    horizontal: bool = False,
    errors: list[float] | None = None,
) -> None:
    errors = errors or [0.0] * len(values)
    for index, value in enumerate(values):
        label = f"{value * 100:.2f}%"
        if errors[index]:
            label += f" ± {errors[index] * 100:.2f}%"
        if horizontal:
            ax.text(min(value + errors[index] + 0.015, 0.99), index, label, va="center", fontsize=9)
        else:
            ax.text(index, min(value + errors[index] + 0.018, 1.12), label, ha="center", va="bottom", fontsize=9)


def pct(earned: float, max_score: float) -> float:
    return earned / max_score if max_score else 0.0


def rounded(value: float) -> float:
    return round(value, 4)


def add_score(bucket: dict[str, float], earned: float, max_score: float) -> None:
    bucket["earned"] += earned
    bucket["max"] += max_score


def confidence_interval(values: list[float], confidence: float = 0.95) -> float:
    """Half-width of a Student's t confidence interval on the mean.

    Widens automatically for small n (unlike a fixed z=1.96), which matters
    here since some criterion/dimension buckets only have a handful of
    observations. Returns 0.0 when n < 2 (variance is not estimable).
    """
    n = len(values)
    if n < 2:
        return 0.0
    sem = float(np.std(values, ddof=1)) / np.sqrt(n)
    t_critical = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    return rounded(t_critical * sem)


def add_observation(bucket: dict[str, Any], earned: float, max_score: float) -> None:
    if max_score:
        bucket.setdefault("observations", []).append(pct(earned, max_score))


def bounded_yerr(values: list[float], errors: list[float]) -> np.ndarray:
    lower = [min(error, value) for value, error in zip(values, errors)]
    upper = [min(error, 1.0 - value) for value, error in zip(values, errors)]
    return np.array([lower, upper])


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.grid(axis=grid_axis, color=COLORS["light_gray"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def task_id_from_run_id(run_id: str) -> str:
    parts = run_id.split("_", 1)
    return parts[1] if len(parts) == 2 else run_id


def resolve_run_index_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    candidates = [path if path.is_absolute() else DEFAULT_EVALUATION_DIR / path]

    # Some recorded paths come from another machine/mount (e.g. a different cluster
    # or user account), and some are relative but already include a redundant
    # "evaluation/..." prefix that isn't relative to DEFAULT_EVALUATION_DIR.
    # Both cases can be rebased onto this machine's TOOL_DIR using that directory
    # name as a stable anchor, since it's the one path segment guaranteed to match.
    parts = path.parts
    if TOOL_DIR.name in parts:
        last_index = len(parts) - 1 - parts[::-1].index(TOOL_DIR.name)
        candidates.append(TOOL_DIR.joinpath(*parts[last_index + 1 :]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def is_run_index_csv(path: Path) -> bool:
    return path.is_file() and path.suffix == ".csv" and not path.name.endswith("_latest_run_status.csv")


def benchmark_from_task_id(task_id: str) -> str:
    if DEFAULT_SPEC_DIR is None:
        return "indexed_runs"
    for spec_file in sorted(DEFAULT_SPEC_DIR.glob("*.json"), key=lambda item: len(item.stem), reverse=True):
        if task_id == spec_file.stem or task_id.startswith(f"{spec_file.stem}_"):
            return spec_file.stem
    return "indexed_runs"


def benchmark_from_index_path(path_text: str, task_id: str) -> str:
    path = resolve_run_index_path(path_text)
    try:
        relative = path.resolve().relative_to(DEFAULT_EVALUATION_DIR.resolve())
        if relative.parts:
            return relative.parts[0]
    except ValueError:
        pass

    # path structure: {eval_dir}/{benchmark}/{batch}/{run} — 2 levels up from run dir
    if path.parent.parent.name:
        return path.parent.parent.name

    raw_path = Path(path_text)
    if not raw_path.is_absolute() and raw_path.parts:
        return raw_path.parts[0]

    return benchmark_from_task_id(task_id)


def source_task_id_lookup() -> dict[str, str]:
    if DEFAULT_SPEC_DIR is None:
        return {}
    try:
        from run_index import load_spec_tasks
    except Exception:
        return {}

    lookup: dict[str, str] = {}
    for spec_file in sorted(DEFAULT_SPEC_DIR.glob("*.json")):
        try:
            tasks = load_spec_tasks(spec_file)
        except Exception:
            continue
        for task in tasks:
            task_id = str(task.get("id", ""))
            if task_id:
                lookup[task_id] = str(task.get("source_task_id") or task_id)
    return lookup


def has_glob_syntax(value: str) -> bool:
    return any(char in value for char in "*?[]")


def load_run_id_file(path: Path) -> set[str]:
    run_ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            run_ids.add(value)
    return run_ids


@dataclass(frozen=True)
class RunIndexFilter:
    completed_run_paths_by_task_id: dict[str, Path]
    run_id_by_task_id: dict[str, str]
    exact_run_ids: set[str]
    include_run_id_patterns: tuple[str, ...]
    include_run_id_regexes: tuple[re.Pattern[str], ...]
    include_task_id_patterns: tuple[str, ...]
    include_source_task_id_patterns: tuple[str, ...]
    include_benchmark_patterns: tuple[str, ...]

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        *,
        exact_run_ids: set[str] | None = None,
        include_run_id_patterns: list[str] | None = None,
        include_run_id_regexes: list[str] | None = None,
        include_task_id_patterns: list[str] | None = None,
        include_source_task_id_patterns: list[str] | None = None,
        include_benchmark_patterns: list[str] | None = None,
    ) -> "RunIndexFilter":
        if not csv_path.exists():
            raise SystemExit(f"Run index CSV not found: {csv_path}")

        completed_run_paths_by_task_id: dict[str, Path] = {}
        run_id_by_task_id: dict[str, str] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                run_id = (row.get("run_id") or "").strip()
                path_text = (row.get("path") or "").strip()
                if not run_id or not path_text:
                    continue
                task_id = task_id_from_run_id(run_id)
                completed_run_paths_by_task_id[task_id] = resolve_run_index_path(path_text)
                run_id_by_task_id[task_id] = run_id

        return cls(
            completed_run_paths_by_task_id=completed_run_paths_by_task_id,
            run_id_by_task_id=run_id_by_task_id,
            exact_run_ids=exact_run_ids or set(),
            include_run_id_patterns=tuple(include_run_id_patterns or []),
            include_run_id_regexes=tuple(re.compile(pattern) for pattern in include_run_id_regexes or []),
            include_task_id_patterns=tuple(include_task_id_patterns or []),
            include_source_task_id_patterns=tuple(include_source_task_id_patterns or []),
            include_benchmark_patterns=tuple(include_benchmark_patterns or []),
        )

    @property
    def has_selector(self) -> bool:
        return bool(
            self.exact_run_ids
            or self.include_run_id_patterns
            or self.include_run_id_regexes
            or self.include_task_id_patterns
            or self.include_source_task_id_patterns
            or self.include_benchmark_patterns
        )

    def selected_text(self, value: str, patterns: tuple[str, ...]) -> bool:
        return any(
            fnmatch.fnmatchcase(value, pattern) if has_glob_syntax(pattern) else pattern in value
            for pattern in patterns
        )

    def selected_row(self, *, benchmark: str, run_id: str, task_id: str, source_task_id: str) -> bool:
        if self.selected_text(benchmark, self.include_benchmark_patterns):
            return True
        if run_id in self.exact_run_ids:
            return True
        return (
            self.selected_text(run_id, self.include_run_id_patterns)
            or self.selected_text(task_id, self.include_task_id_patterns)
            or self.selected_text(source_task_id, self.include_source_task_id_patterns)
            or any(pattern.search(run_id) for pattern in self.include_run_id_regexes)
        )

    def run_path_for_task(self, task_id: str) -> Path | None:
        return self.completed_run_paths_by_task_id.get(task_id)

    def run_id_for_task(self, task_id: str) -> str:
        return self.run_id_by_task_id.get(task_id, task_id)


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_score_label(label: str) -> tuple[float, float] | None:
    if "/" not in label:
        return None
    earned_text, max_text = label.split("/", 1)
    try:
        return float(earned_text), float(max_text)
    except ValueError:
        return None


def score_from_status_row(row: dict[str, str], result: dict[str, Any] | None) -> dict[str, Any]:
    result_score = (result or {}).get("score")
    if not isinstance(result_score, dict):
        result_score = {}

    earned = result_score.get("earned")
    max_score = result_score.get("max")
    percentage = result_score.get("percentage")
    label = result_score.get("label") or row.get("latest_result_score_label", "")

    if (earned is None or max_score is None) and label:
        parsed = parse_score_label(label)
        if parsed is not None:
            earned, max_score = parsed

    if percentage in {None, ""}:
        percentage = row.get("latest_result_score_percentage", "")

    if earned is None or max_score is None:
        try:
            percentage_float = float(percentage)
        except (TypeError, ValueError):
            percentage_float = 0.0
        earned = percentage_float
        max_score = 1.0
    else:
        earned = float(earned)
        max_score = float(max_score)
        percentage_float = pct(earned, max_score)

    return {
        "earned": earned,
        "max": max_score,
        "percentage": rounded(percentage_float),
        "label": label or f"{earned:g}/{max_score:g}",
        "criteria": result_score.get("criteria", []),
    }


def latest_status_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.name.endswith("_latest_run_status.csv"):
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*_latest_run_status.csv"))
    return []


def run_index_selection_metadata(run_index_filter: RunIndexFilter | None) -> dict[str, Any]:
    return {
        "enabled": run_index_filter is not None,
        "selected_run_ids": sorted(run_index_filter.exact_run_ids) if run_index_filter else [],
        "include_run_id_patterns": list(run_index_filter.include_run_id_patterns) if run_index_filter else [],
        "include_run_id_regexes": [pattern.pattern for pattern in run_index_filter.include_run_id_regexes] if run_index_filter else [],
        "include_task_id_patterns": list(run_index_filter.include_task_id_patterns) if run_index_filter else [],
        "include_source_task_id_patterns": list(run_index_filter.include_source_task_id_patterns) if run_index_filter else [],
        "include_benchmark_patterns": list(run_index_filter.include_benchmark_patterns) if run_index_filter else [],
    }


def indexed_run_evidence(run_dir: Path) -> dict[str, str]:
    result = load_json_object(run_dir / "result.json")
    judge = load_json_object(run_dir / "judge_response.json")

    result_status = str((result or {}).get("status") or "")
    judge_status = str((judge or {}).get("status") or "")
    result_complete = result_status in {"passed", "failed"}
    judge_complete = judge_status in {"passed", "failed"}
    evaluation_complete = result_complete and judge_complete

    if evaluation_complete and result_status == "passed":
        judge_outcome = "judge_passed"
    elif evaluation_complete and result_status == "failed":
        judge_outcome = "judge_failed"
    else:
        judge_outcome = "judge_unavailable"

    result_score = (result or {}).get("score")
    if not isinstance(result_score, dict):
        result_score = {}

    proof_parts = [
        f"run_index_path='{run_dir}'",
        f"result.json.status='{result_status or 'missing_or_invalid'}'",
        f"judge_response.json.status='{judge_status or 'missing_or_invalid'}'",
    ]
    if result_score.get("label"):
        proof_parts.append(f"result.json.score.label='{result_score['label']}'")
    if result_score.get("percentage") not in {None, ""}:
        proof_parts.append(f"result.json.score.percentage={result_score['percentage']}")

    return {
        "rerun_status": "passed" if evaluation_complete else "run_error",
        # Keep selected index rows in the plotted task set; incomplete evidence is
        # represented by judge_unavailable and counted as an error in the plot.
        "run_completion_status": "run_completed",
        "judge_outcome": judge_outcome,
        "latest_judge_status": judge_status,
        "latest_result_status": result_status,
        "latest_judge_raw_score_1_to_5": str((judge or {}).get("raw_score_1_to_5") or ""),
        "latest_judge_pass_threshold_1_to_5": str((judge or {}).get("pass_threshold_1_to_5") or ""),
        "latest_result_score_label": str(result_score.get("label") or ""),
        "latest_result_score_percentage": str(result_score.get("percentage") or ""),
        "proof": " | ".join(proof_parts),
    }


def apply_run_index_filter(
    rows: list[dict[str, str]],
    *,
    benchmark: str,
    run_index_filter: RunIndexFilter | None,
) -> list[dict[str, str]]:
    if run_index_filter is None:
        return rows

    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        task_id = row.get("task_id", "")
        indexed_path = run_index_filter.run_path_for_task(task_id)
        if indexed_path is None:
            continue

        run_id = run_index_filter.run_id_for_task(task_id)
        if run_index_filter.has_selector and not run_index_filter.selected_row(
            benchmark=benchmark,
            run_id=run_id,
            task_id=task_id,
            source_task_id=row.get("source_task_id", ""),
        ):
            continue

        filtered_row = dict(row)
        filtered_row["run_id"] = run_id
        filtered_row.update(indexed_run_evidence(indexed_path))
        filtered_row["latest_completed_run_dir"] = str(indexed_path)
        filtered_row["latest_run_dir"] = str(indexed_path)
        filtered_rows.append(filtered_row)

    return filtered_rows


def resolve_run_dir(run_dir_text: str, csv_path: Path) -> Path:
    """Return the run directory path, rebasing it to the local evaluation root when the
    original absolute path (from another machine) no longer exists."""
    path = Path(run_dir_text)
    if not run_dir_text or path.exists():
        return path
    # Rebase: take last 3 components (benchmark/batch/run) under csv_path.parent.parent
    parts = path.parts
    if len(parts) >= 3:
        rebased = csv_path.parent.parent.joinpath(*parts[-3:])
        if rebased.exists():
            return rebased
    return path


def load_deduplicated_latest_status_data(
    input_path: Path,
    run_index_filter: RunIndexFilter | None = None,
) -> dict[str, Any]:
    csv_paths = latest_status_csvs(input_path)
    if not csv_paths:
        raise SystemExit(f"No *_latest_run_status.csv files found under {input_path}")

    rows_by_benchmark: dict[str, list[dict[str, str]]] = {}
    for csv_path in csv_paths:
        benchmark = csv_path.name.removesuffix("_latest_run_status.csv")
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows_by_benchmark[benchmark] = apply_run_index_filter(
            rows,
            benchmark=benchmark,
            run_index_filter=run_index_filter,
        )

    return load_deduplicated_status_rows(
        rows_by_benchmark,
        source_path=input_path,
        source_kind="latest_run_status_csv",
        run_index_filter=run_index_filter,
    )


def load_deduplicated_run_index_data(
    input_path: Path,
    run_index_filter: RunIndexFilter | None = None,
) -> dict[str, Any]:
    if not input_path.exists():
        raise SystemExit(f"Run index CSV not found: {input_path}")

    lookup = source_task_id_lookup()
    rows_by_benchmark: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = (row.get("run_id") or "").strip()
            path_text = (row.get("path") or "").strip()
            print(path_text)
            if not run_id:
                continue
            run_requested = (row.get("run_requested") or "true").strip().lower()
            if run_requested != "false":
                continue
            task_id = task_id_from_run_id(run_id)
            source_task_id = lookup.get(task_id, task_id)
            if not path_text:
                continue

            benchmark = benchmark_from_index_path(path_text, task_id)
            if run_index_filter is not None and run_index_filter.has_selector and not run_index_filter.selected_row(
                benchmark=benchmark,
                run_id=run_id,
                task_id=task_id,
                source_task_id=source_task_id,
            ):
                continue

            run_dir = resolve_run_index_path(path_text)
            status_row = {
                "run_id": run_id,
                "task_id": task_id,
                "source_task_id": source_task_id,
                "variant_path": "",
                "latest_run_folder": run_dir.name,
                "latest_completed_run_folder": run_dir.name,
                "latest_run_dir": str(run_dir),
                "latest_completed_run_dir": str(run_dir),
                "present_run_count": "1",
                "completed_run_count": "1",
            }
            status_row.update(indexed_run_evidence(run_dir))
            rows_by_benchmark[benchmark].append(status_row)

    return load_deduplicated_status_rows(
        dict(rows_by_benchmark),
        source_path=input_path,
        source_kind="run_index_csv",
        run_index_filter=run_index_filter,
    )


def load_deduplicated_status_rows(
    rows_by_benchmark: dict[str, list[dict[str, str]]],
    *,
    source_path: Path,
    source_kind: str,
    run_index_filter: RunIndexFilter | None = None,
    pre_excluded_total: int = 0,
) -> dict[str, Any]:
    per_file: list[dict[str, Any]] = []
    per_file_dimensions: list[dict[str, Any]] = []
    per_file_categories: list[dict[str, Any]] = []
    per_file_criteria: list[dict[str, Any]] = []

    overall_score = {"earned": 0.0, "max": 0.0}
    completed_total = 0
    excluded_total = pre_excluded_total

    for benchmark, rows in sorted(rows_by_benchmark.items()):
        completed_rows = [row for row in rows if row.get("run_completion_status") == "run_completed"]
        excluded_total += len(rows) - len(completed_rows)
        completed_total += len(completed_rows)

        file_score = {"earned": 0.0, "max": 0.0}
        by_dimension: dict[str, dict[str, Any]] = defaultdict(lambda: {"earned": 0.0, "max": 0.0, "observations": []})
        by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"earned": 0.0, "max": 0.0, "observations": []})
        by_criterion: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"earned": 0.0, "max": 0.0, "observations": [], "label": "", "dimension": "", "category": ""}
        )

        passed = failed = errors = 0
        file_observations: list[float] = []
        for row in completed_rows:
            if row.get("judge_outcome") == "judge_passed":
                passed += 1
            elif row.get("judge_outcome") == "judge_failed":
                failed += 1
            else:
                errors += 1

            result_path = resolve_run_dir(row.get("latest_completed_run_dir", ""), source_path) / "result.json"
            result = load_json_object(result_path)
            score = score_from_status_row(row, result)
            add_score(file_score, score["earned"], score["max"])
            add_score(overall_score, score["earned"], score["max"])
            add_observation({"observations": file_observations}, score["earned"], score["max"])

            row_by_dimension: dict[str, dict[str, float]] = defaultdict(lambda: {"earned": 0.0, "max": 0.0})
            row_by_category: dict[str, dict[str, float]] = defaultdict(lambda: {"earned": 0.0, "max": 0.0})
            for criterion in score.get("criteria", []):
                if not isinstance(criterion, dict):
                    continue
                earned = float(criterion.get("earned", 0) or 0)
                max_score = float(criterion.get("max", 0) or 0)
                dimension = str(criterion.get("dimension", "unknown"))
                category = str(criterion.get("category", "unknown"))
                criterion_id = str(criterion.get("id", "unknown"))

                add_score(by_dimension[dimension], earned, max_score)
                add_score(by_category[category], earned, max_score)
                add_score(by_criterion[criterion_id], earned, max_score)
                add_score(row_by_dimension[dimension], earned, max_score)
                add_score(row_by_category[category], earned, max_score)
                add_observation(by_criterion[criterion_id], earned, max_score)
                by_criterion[criterion_id]["label"] = str(criterion.get("label", criterion_id))
                by_criterion[criterion_id]["dimension"] = dimension
                by_criterion[criterion_id]["category"] = category

            for dimension, values in row_by_dimension.items():
                add_observation(by_dimension[dimension], values["earned"], values["max"])
            for category, values in row_by_category.items():
                add_observation(by_category[category], values["earned"], values["max"])

        per_file.append(
            {
                "json_file": f"{benchmark}.json",
                "status": "SUCCESS" if errors == 0 else "PARTIAL",
                "n_tasks": len(completed_rows),
                "n_passed": passed,
                "n_failed": failed,
                "n_errors": errors,
                "n_excluded": len(rows) - len(completed_rows),
                "earned": file_score["earned"],
                "max": file_score["max"],
                "percentage": rounded(pct(file_score["earned"], file_score["max"])),
                "ci95_percentage": confidence_interval(file_observations),
                "n_observations": len(file_observations),
                "observation_percentages": list(file_observations),
                "label": f"{file_score['earned']:.0f}/{file_score['max']:.0f}",
            }
        )

        for dimension, values in sorted(by_dimension.items()):
            per_file_dimensions.append(
                {
                    "json_file": f"{benchmark}.json",
                    "section_type": "dimension",
                    "section": dimension,
                    "earned": values["earned"],
                    "max": values["max"],
                    "percentage": rounded(pct(values["earned"], values["max"])),
                    "ci95_percentage": confidence_interval(values["observations"]),
                    "n_observations": len(values["observations"]),
                    "observation_percentages": [rounded(value) for value in values["observations"]],
                }
            )

        for category, values in sorted(by_category.items()):
            per_file_categories.append(
                {
                    "json_file": f"{benchmark}.json",
                    "section_type": "category",
                    "section": category,
                    "earned": values["earned"],
                    "max": values["max"],
                    "percentage": rounded(pct(values["earned"], values["max"])),
                    "ci95_percentage": confidence_interval(values["observations"]),
                    "n_observations": len(values["observations"]),
                    "observation_percentages": [rounded(value) for value in values["observations"]],
                }
            )

        for criterion_id, values in sorted(by_criterion.items()):
            per_file_criteria.append(
                {
                    "json_file": f"{benchmark}.json",
                    "criterion_id": criterion_id,
                    "criterion_label": values["label"],
                    "dimension": values["dimension"],
                    "category": values["category"],
                    "earned": values["earned"],
                    "max": values["max"],
                    "percentage": rounded(pct(values["earned"], values["max"])),
                    "ci95_percentage": confidence_interval(values["observations"]),
                    "n_observations": len(values["observations"]),
                    "observation_percentages": [rounded(value) for value in values["observations"]],
                }
            )

    section_rows = per_file_dimensions + per_file_categories
    section_means = []
    for section_type, section in sorted({(row["section_type"], row["section"]) for row in section_rows}):
        matching = [row for row in section_rows if row["section_type"] == section_type and row["section"] == section]
        earned = sum(row["earned"] for row in matching)
        max_score = sum(row["max"] for row in matching)
        observations = [
            value
            for row in matching
            for value in row.get("observation_percentages", [row["percentage"]])
        ]
        json_percentages = [row["percentage"] for row in matching]
        section_means.append(
            {
                "section_type": section_type,
                "section": section,
                "n_json_files": len(matching),
                "n_observations": len(observations),
                "pooled_earned": earned,
                "pooled_max": max_score,
                "pooled_percentage": rounded(pct(earned, max_score)),
                "ci95_percentage": confidence_interval(observations),
                "ci95_percentage_across_json_files": confidence_interval(json_percentages),
            }
        )

    criterion_means = []
    for criterion_id in sorted({row["criterion_id"] for row in per_file_criteria}):
        matching = [row for row in per_file_criteria if row["criterion_id"] == criterion_id]
        earned = sum(row["earned"] for row in matching)
        max_score = sum(row["max"] for row in matching)
        observations = [
            value
            for row in matching
            for value in row.get("observation_percentages", [row["percentage"]])
        ]
        json_percentages = [row["percentage"] for row in matching]
        first = matching[0]
        criterion_means.append(
            {
                "criterion_id": criterion_id,
                "criterion_label": first["criterion_label"],
                "dimension": first["dimension"],
                "category": first["category"],
                "n_json_files": len(matching),
                "n_observations": len(observations),
                "pooled_earned": earned,
                "pooled_max": max_score,
                "pooled_percentage": rounded(pct(earned, max_score)),
                "ci95_percentage": confidence_interval(observations),
                "ci95_percentage_across_json_files": confidence_interval(json_percentages),
            }
        )

    file_percentages = [row["percentage"] for row in per_file]
    if run_index_filter is not None and completed_total == 0:
        raise SystemExit("Run-index filter selected no completed runs.")
    if completed_total == 0:
        raise SystemExit(f"No completed runs found in {source_path}")

    return {
        "source_dir": str(source_path),
        "source_kind": source_kind,
        "run_index_filter": run_index_selection_metadata(run_index_filter),
        "n_json_files": len(per_file),
        "n_completed_tasks": completed_total,
        "n_excluded_tasks": excluded_total,
        "overall": {
            "pooled_earned": overall_score["earned"],
            "pooled_max": overall_score["max"],
            "pooled_percentage": rounded(pct(overall_score["earned"], overall_score["max"])),
            "macro_mean_percentage_across_json_files": rounded(float(np.mean(file_percentages))) if file_percentages else 0,
            "macro_ci95_percentage_across_json_files": confidence_interval(file_percentages),
        },
        "per_json_file": per_file,
        "per_json_dimension_scores": per_file_dimensions,
        "per_json_category_scores": per_file_categories,
        "per_json_criterion_scores": per_file_criteria,
        "section_means_across_json_files": section_means,
        "criterion_means_across_json_files": criterion_means,
    }


def load_plot_data(input_path: Path, run_index_filter: RunIndexFilter | None = None) -> tuple[dict[str, Any], Path]:
    if input_path.is_file() and input_path.suffix == ".json":
        if run_index_filter is not None:
            raise SystemExit("Run-index filtering is only supported for latest-run-status CSV input, not aggregate JSON input.")
        data = json.loads(input_path.read_text(encoding="utf-8"))
        return data, input_path.parent / "plots"

    if is_run_index_csv(input_path):
        data = load_deduplicated_run_index_data(input_path, run_index_filter=run_index_filter)
        return data, DEFAULT_AGGREGATE_DIR / "plots"

    data = load_deduplicated_latest_status_data(input_path, run_index_filter=run_index_filter)
    output_dir = input_path.parent / "plots" if input_path.is_file() else input_path / "plots"
    return data, output_dir


def plot_per_json_scores(data: dict, output_dir: Path) -> None:
    rows = sorted(data["per_json_file"], key=lambda row: row["percentage"])
    labels = [short_name(row["json_file"]) for row in rows]
    values = [row["percentage"] for row in rows]
    errors = [row.get("ci95_percentage", 0.0) for row in rows]
    colors = [COLORS["red"] if row["status"] != "SUCCESS" else COLORS["blue"] for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(
        wrap_labels(labels, 28),
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        xerr=bounded_yerr(values, errors),
        ecolor=COLORS["dark_gray"],
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    overall = data["overall"]["pooled_percentage"]
    ax.axvline(overall, color=COLORS["dark_gray"], linestyle="--", linewidth=1.4)
    ax.text(
        min(overall + 0.012, 0.88),
        len(labels) - 0.5,
        f"Overall {overall * 100:.1f}%",
        color=COLORS["dark_gray"],
        fontsize=9.5,
        fontweight="bold",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": COLORS["light_gray"], "alpha": 0.95},
    )
    add_percent_labels(ax, values, horizontal=True, errors=errors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score percentage")
    ax.set_title("Final Evaluation Score by JSON File")
    ax.text(0.0, -0.12, "Error bars: 95% confidence interval of the mean (Student's t) across selected runs within each JSON file.", transform=ax.transAxes, fontsize=8.5, color=COLORS["gray"])
    style_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    save(fig, output_dir / "per_json_scores.png")


def plot_task_outcomes(data: dict, output_dir: Path) -> None:
    rows = data["per_json_file"]
    labels = [short_name(row["json_file"]) for row in rows]
    passed = np.array([row["n_passed"] for row in rows])
    failed = np.array([row["n_failed"] for row in rows])
    errors = np.array([row["n_errors"] for row in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(x, passed, label="Passed", color=COLORS["green"], edgecolor="white", linewidth=0.7)
    ax.bar(x, failed, bottom=passed, label="Failed", color=COLORS["yellow"], edgecolor="white", linewidth=0.7)
    ax.bar(x, errors, bottom=passed + failed, label="Errors", color=COLORS["red"], edgecolor="white", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(labels, 14), rotation=0, ha="center")
    ax.set_ylabel("Number of tasks")
    ax.set_title("Task Outcomes by JSON File")
    ax.legend(frameon=False, ncols=3, loc="upper right")
    style_axes(ax, grid_axis="y")
    save(fig, output_dir / "task_outcomes_by_json.png")


def plot_section_means(data: dict, output_dir: Path, section_type: str, file_name: str, title: str) -> None:
    rows = [row for row in data["section_means_across_json_files"] if row["section_type"] == section_type]
    if section_type == "dimension":
        rows = sorted(rows, key=lambda row: DIMENSION_ORDER.get(row["section"], len(DIMENSION_ORDER)))
        labels = [DIMENSION_LABELS.get(row["section"], row["section"]) for row in rows]
    else:
        rows = sorted(rows, key=lambda row: row["pooled_percentage"], reverse=True)
        labels = [row["section"] for row in rows]
    values = [row["pooled_percentage"] for row in rows]
    errors = [row.get("ci95_percentage", 0.0) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(labels))
    ax.bar(
        x,
        values,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.8,
        yerr=bounded_yerr(values, errors),
        ecolor=COLORS["dark_gray"],
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
    )
    add_percent_labels(ax, values, errors=errors)
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(labels, 14), rotation=0, ha="center")
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Pooled percentage")
    ax.set_title(title, pad=16)
    ax.text(0.0, -0.16, "Error bars: 95% confidence interval of the mean (Student's t) across selected run observations.", transform=ax.transAxes, fontsize=8.5, color=COLORS["gray"])
    style_axes(ax, grid_axis="y")
    save(fig, output_dir / file_name)


def plot_dimension_heatmap(data: dict, output_dir: Path) -> None:
    rows = data["per_json_dimension_scores"]
    json_files = [row["json_file"] for row in data["per_json_file"]]
    dimensions = sorted({row["section"] for row in rows}, key=lambda dimension: DIMENSION_ORDER.get(dimension, len(DIMENSION_ORDER)))
    if not dimensions:
        print("Skipping dimension heatmap: no dimension data available.")
        return
    lookup = {(row["json_file"], row["section"]): row["percentage"] for row in rows}
    ci_lookup = {(row["json_file"], row["section"]): row.get("ci95_percentage", 0.0) for row in rows}
    matrix = np.array([[lookup.get((json_file, dimension), np.nan) for dimension in dimensions] for json_file in json_files])

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_xticklabels(wrap_labels([DIMENSION_LABELS.get(dimension, dimension) for dimension in dimensions], 12))
    ax.set_yticks(np.arange(len(json_files)))
    ax.set_yticklabels(wrap_labels([short_name(name) for name in json_files], 24))
    ax.set_title("Dimension Scores by JSON File")

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            if not np.isnan(value):
                ci95 = ci_lookup.get((json_files[y], dimensions[x]), 0.0)
                label = f"{value * 100:.2f}%"
                if ci95:
                    label += f"\n± {ci95 * 100:.2f}%"
                text_color = "white" if value < 0.55 else "black"
                ax.text(x, y, label, ha="center", va="center", color=text_color, fontsize=7)

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Score percentage")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    save(fig, output_dir / "dimension_scores_heatmap.png")


def plot_criterion_means(data: dict, output_dir: Path) -> None:
    rows = sorted(
        data["criterion_means_across_json_files"],
        key=lambda row: (DIMENSION_ORDER.get(row["dimension"], len(DIMENSION_ORDER)), row["criterion_id"]),
    )
    labels = [row["criterion_id"].replace("_", " ") for row in rows]
    values = [row["pooled_percentage"] for row in rows]
    errors = [row.get("ci95_percentage", 0.0) for row in rows]

    fig, ax = plt.subplots(figsize=(9.8, 7.2))
    ax.barh(
        wrap_labels(labels, 34),
        values,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.7,
        xerr=bounded_yerr(values, errors),
        ecolor=COLORS["dark_gray"],
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax.invert_yaxis()
    add_percent_labels(ax, values, horizontal=True, errors=errors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Pooled percentage")
    ax.set_title("Criterion Means Across JSON Files", pad=14)
    ax.text(0.0, -0.08, "Error bars: 95% confidence interval of the mean (Student's t) across selected run observations.", transform=ax.transAxes, fontsize=8.5, color=COLORS["gray"])
    style_axes(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    save(fig, output_dir / "criterion_means.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_RUN_INDEX_CSV,
        type=Path,
        help=(
            "Path to run_index.csv, aggregate_scores.json, an aggregate_scores "
            "directory containing *_latest_run_status.csv files, or one *_latest_run_status.csv file."
        ),
    )
    parser.add_argument(
        "--completed-run-index",
        type=Path,
        default=None,
        help=(
            "Optional run_index.csv for filtering compatibility latest-run-status CSV input. "
            "Run-index CSV input already reads result.json directly from indexed paths."
        ),
    )
    parser.add_argument(
        "--include-run-id",
        action="append",
        default=[],
        help=(
            "Keep indexed runs whose run_id contains this text. Glob patterns are also "
            "accepted. Can be repeated, for example T1_S26 and T2_S53."
        ),
    )
    parser.add_argument(
        "--include-run-id-regex",
        action="append",
        default=[],
        help="Keep indexed runs whose run_id matches this regular expression. Can be repeated.",
    )
    parser.add_argument(
        "--include-task-id",
        action="append",
        default=[],
        help=(
            "Keep indexed runs whose expanded task_id contains this text. Glob patterns "
            "are accepted. Useful for global tasks such as statistics_qa_5."
        ),
    )
    parser.add_argument(
        "--include-source-task-id",
        action="append",
        default=[],
        help=(
            "Keep indexed runs whose source_task_id contains this text. Glob patterns "
            "are accepted. Useful for selecting a whole original question before expansion."
        ),
    )
    parser.add_argument(
        "--include-benchmark",
        action="append",
        default=[],
        help=(
            "Keep indexed runs from benchmarks whose CSV stem contains this text, for "
            "example statistics_qa. Glob patterns are accepted."
        ),
    )
    parser.add_argument(
        "--run-id-file",
        action="append",
        type=Path,
        default=[],
        help="Text file containing exact run_id values, one per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for plots and deduplicated_plot_data.json.",
    )
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    exact_run_ids = set()
    for run_id_file in args.run_id_file:
        exact_run_ids.update(load_run_id_file(run_id_file))

    run_index_filter = None
    if (
        args.completed_run_index
        or args.include_run_id
        or args.include_run_id_regex
        or args.include_task_id
        or args.include_source_task_id
        or args.include_benchmark
        or exact_run_ids
    ):
        selector_csv = args.completed_run_index or (input_path if is_run_index_csv(input_path) else DEFAULT_RUN_INDEX_CSV)
        run_index_filter = RunIndexFilter.from_csv(
            selector_csv.resolve(),
            exact_run_ids=exact_run_ids,
            include_run_id_patterns=args.include_run_id,
            include_run_id_regexes=args.include_run_id_regex,
            include_task_id_patterns=args.include_task_id,
            include_source_task_id_patterns=args.include_source_task_id,
            include_benchmark_patterns=args.include_benchmark,
        )

    data, default_output_dir = load_plot_data(input_path, run_index_filter=run_index_filter)
    data = merge_label_groups(data)
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_per_json_scores(data, output_dir)
    plot_task_outcomes(data, output_dir)
    plot_section_means(data, output_dir, "dimension", "dimension_means.png", "Dimension Means for Selected Runs")
    plot_dimension_heatmap(data, output_dir)
    plot_criterion_means(data, output_dir)

    (output_dir / "deduplicated_plot_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Matplotlib plots to {output_dir}")


if __name__ == "__main__":
    main()
