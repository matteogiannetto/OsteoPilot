#!/usr/bin/env python3
"""Create a latest-run-status report for an OsteoBench JSON specification."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
TOOL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TOOL_DIR))

from run_index import load_spec_tasks  # noqa: E402  # Requires the path bootstrap above.


DEFAULT_OUTPUT_DIR = TOOL_DIR / "output" / "runs" / "aggregate_scores"


CSV_FIELDS = [
    "rerun_status",
    "run_completion_status",
    "judge_outcome",
    "task_id",
    "source_task_id",
    "question",
    "variant_path",
    "latest_batch",
    "latest_run_folder",
    "latest_judge_status",
    "latest_result_status",
    "latest_judge_raw_score_1_to_5",
    "latest_judge_pass_threshold_1_to_5",
    "latest_result_score_label",
    "latest_result_score_percentage",
    "present_run_count",
    "completed_run_count",
    "latest_completed_batch",
    "latest_completed_run_folder",
    "proof",
    "latest_run_dir",
    "latest_completed_run_dir",
]


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"invalid_json: {exc}"
    if not isinstance(parsed, dict):
        return None, f"not_object: {type(parsed).__name__}"
    return parsed, None


def compact_text(value: Any, limit: int = 220) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def batch_datetime(batch_name: str) -> datetime:
    match = re.search(r"_(\d{2}-\d{2}-\d{4})_(\d{2}-\d{2}-\d{2})$", batch_name)
    if not match:
        return datetime.min
    return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%d-%m-%Y_%H-%M-%S")


def run_sort_key(run_dir: Path) -> tuple[datetime, float, str]:
    batch = run_dir.parent.name
    try:
        mtime = run_dir.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (batch_datetime(batch), mtime, run_dir.name)


def judge_outcome(judge: dict[str, Any] | None) -> str:
    judge_status = (judge or {}).get("status")
    if judge_status == "passed":
        return "judge_passed"
    if judge_status == "failed":
        return "judge_failed"
    return "judge_unavailable"


def run_completion_status(
    run_dir: Path,
    result: dict[str, Any] | None,
    result_error: str | None,
) -> str:
    result_status = (result or {}).get("status")

    if result_status in {"passed", "failed"}:
        return "run_completed"
    if result_status == "error":
        return "run_error"
    if result is None:
        return "run_error"

    run_summary_exists = (run_dir / "run_summary.json").exists()
    judge_exists = (run_dir / "judge_response.json").exists()
    if run_summary_exists or judge_exists:
        return "run_completed"

    return "run_error"


def find_present_runs(run_root: Path) -> list[Path]:
    run_dirs: set[Path] = set()
    for file_name in ("judge_response.json", "result.json", "run_summary.json", "execution_trace.json"):
        for path in run_root.glob(f"*/*/{file_name}"):
            run_dirs.add(path.parent)
    return sorted(run_dirs, key=run_sort_key)


def run_record(run_dir: Path) -> dict[str, Any]:
    judge, judge_error = load_json(run_dir / "judge_response.json")
    result, result_error = load_json(run_dir / "result.json")
    task_id = (judge or {}).get("task_id") or (result or {}).get("task_id") or ""
    completion_status = run_completion_status(run_dir, result, result_error)
    rerun_status = "passed" if completion_status == "run_completed" else completion_status
    verdict = judge_outcome(judge)

    proof = []
    if judge is None:
        proof.append(f"judge_response.json={judge_error}")
    else:
        proof.append(f"judge_response.json.status={(judge or {}).get('status')!r}")
        if judge.get("raw_score_1_to_5") is not None:
            proof.append(f"judge_response.json.raw_score_1_to_5={judge.get('raw_score_1_to_5')!r}")
        if judge.get("pass_threshold_1_to_5") is not None:
            proof.append(f"judge_response.json.pass_threshold_1_to_5={judge.get('pass_threshold_1_to_5')!r}")

    if result is None:
        proof.append(f"result.json={result_error}")
    else:
        proof.append(f"result.json.status={result.get('status')!r}")
        score = result.get("score")
        if isinstance(score, dict):
            if score.get("label") is not None:
                proof.append(f"result.json.score.label={score.get('label')!r}")
            if score.get("percentage") is not None:
                proof.append(f"result.json.score.percentage={score.get('percentage')!r}")
        errors = result.get("errors")
        if isinstance(errors, list) and errors:
            first_error = errors[0]
            if isinstance(first_error, dict):
                proof.append(
                    "result.json.errors[0]="
                    f"{compact_text(str(first_error.get('type', 'error')) + ': ' + str(first_error.get('message', '')))}"
                )

    score = (result or {}).get("score")
    if not isinstance(score, dict):
        score = {}

    return {
        "task_id": task_id,
        "run_completion_status": completion_status,
        "rerun_status": rerun_status,
        "judge_outcome": verdict,
        "batch": run_dir.parent.name,
        "run_folder": run_dir.name,
        "judge_status": (judge or {}).get("status"),
        "result_status": (result or {}).get("status"),
        "judge_raw_score_1_to_5": (judge or {}).get("raw_score_1_to_5"),
        "judge_pass_threshold_1_to_5": (judge or {}).get("pass_threshold_1_to_5"),
        "result_score_label": score.get("label"),
        "result_score_percentage": score.get("percentage"),
        "proof": " | ".join(proof),
        "run_dir": str(run_dir),
        "_sort_key": run_sort_key(run_dir),
    }


def variant_path_for_task(task: dict[str, Any]) -> str:
    for key in ("volume_relative_path", "dataset_relative_path", "volume_relative_path_variant", "dataset_relative_path_variant"):
        value = task.get(key)
        if isinstance(value, str):
            return value
    return ""


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def build_rows(spec_file: Path, run_root: Path) -> list[dict[str, Any]]:
    tasks = load_spec_tasks(spec_file)
    records_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_dir in find_present_runs(run_root):
        record = run_record(run_dir)
        if record["task_id"]:
            records_by_task[record["task_id"]].append(record)

    rows = []
    for task in tasks:
        task_id = str(task["id"])
        records = sorted(records_by_task.get(task_id, []), key=lambda item: item["_sort_key"])
        latest = records[-1] if records else None
        completed_runs = [record for record in records if record["run_completion_status"] == "run_completed"]
        latest_completed = completed_runs[-1] if completed_runs else None

        row = {
            "rerun_status": latest["rerun_status"] if latest else "missing_run",
            "run_completion_status": latest["run_completion_status"] if latest else "missing_run",
            "judge_outcome": latest["judge_outcome"] if latest else "judge_unavailable",
            "task_id": task_id,
            "source_task_id": task.get("source_task_id", task_id),
            "question": task.get("question", ""),
            "variant_path": variant_path_for_task(task),
            "latest_batch": latest["batch"] if latest else "",
            "latest_run_folder": latest["run_folder"] if latest else "",
            "latest_judge_status": "" if not latest or latest["judge_status"] is None else latest["judge_status"],
            "latest_result_status": "" if not latest or latest["result_status"] is None else latest["result_status"],
            "latest_judge_raw_score_1_to_5": "" if not latest or latest["judge_raw_score_1_to_5"] is None else latest["judge_raw_score_1_to_5"],
            "latest_judge_pass_threshold_1_to_5": "" if not latest or latest["judge_pass_threshold_1_to_5"] is None else latest["judge_pass_threshold_1_to_5"],
            "latest_result_score_label": "" if not latest or latest["result_score_label"] is None else latest["result_score_label"],
            "latest_result_score_percentage": "" if not latest or latest["result_score_percentage"] is None else latest["result_score_percentage"],
            "present_run_count": len(records),
            "completed_run_count": len(completed_runs),
            "latest_completed_batch": latest_completed["batch"] if latest_completed else "",
            "latest_completed_run_folder": latest_completed["run_folder"] if latest_completed else "",
            "proof": latest["proof"] if latest else f"No present run folder matched this expanded {spec_file.stem}.json task_id.",
            "latest_run_dir": latest["run_dir"] if latest else "",
            "latest_completed_run_dir": latest_completed["run_dir"] if latest_completed else "",
        }
        rows.append(row)
    return rows


def write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    spec_file: Path,
    run_root: Path,
    benchmark_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{benchmark_name}_latest_run_status.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})

    counts = Counter(row["rerun_status"] for row in rows)
    completion_counts = Counter(row["run_completion_status"] for row in rows)
    judge_counts = Counter(row["judge_outcome"] for row in rows)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_source[row["source_task_id"]][row["rerun_status"]] += 1

    report_path = output_dir / f"{benchmark_name}_latest_run_status_report.md"
    lines = [
        f"# {benchmark_name} Latest Present Run Completion Status",
        "",
        f"Spec: `{spec_file}`",
        f"Run root: `{run_root}`",
        "",
        f"For each expanded `{spec_file.name}` task, this keeps the latest present run folder. "
        "Latest is ordered by the timestamp in the batch folder name, then directory mtime.",
        "",
        "Rerun rule: `passed` means the latest run reached an evaluable endpoint. "
        "A completed run is counted as `passed` even when the judge outcome is `judge_failed`. "
        "Only `run_error` and `missing_run` are rerun candidates.",
        "",
        "Completion evidence rule: a latest present run is `run_completed` when "
        "`result.json.status` is `passed` or `failed`, or when final artifacts such as "
        "`run_summary.json`/`judge_response.json` exist. `result.json.status='error'`, "
        "missing `result.json` without final artifacts, or invalid `result.json` is `run_error`.",
        "",
        "## Totals",
        "",
        markdown_table(
            ["expanded_tasks", "passed", "run_error", "missing_run"],
            [[len(rows), counts.get("passed", 0), counts.get("run_error", 0), counts.get("missing_run", 0)]],
        ),
        "",
        "## Evidence Status Counts",
        "",
        markdown_table(
            ["run_completed", "run_error", "missing_run", "judge_passed", "judge_failed", "judge_unavailable"],
            [[
                completion_counts.get("run_completed", 0),
                completion_counts.get("run_error", 0),
                completion_counts.get("missing_run", 0),
                judge_counts.get("judge_passed", 0),
                judge_counts.get("judge_failed", 0),
                judge_counts.get("judge_unavailable", 0),
            ]],
        ),
        "",
        "## By Source Question",
        "",
        markdown_table(
            ["source_task_id", "passed", "run_error", "missing_run", "total"],
            [
                [
                    source_task_id,
                    counts_for_source.get("passed", 0),
                    counts_for_source.get("run_error", 0),
                    counts_for_source.get("missing_run", 0),
                    sum(counts_for_source.values()),
                ]
                for source_task_id, counts_for_source in sorted(by_source.items())
            ],
        ),
        "",
        "## Tasks To Rerun",
        "",
    ]

    rerun_rows = [row for row in rows if row["rerun_status"] != "passed"]
    if rerun_rows:
        lines.append(
            markdown_table(
                [
                    "rerun_status",
                    "run_completion_status",
                    "task_id",
                    "variant_path",
                    "latest_run_folder",
                    "result_status",
                    "proof",
                ],
                [
                    [
                        row["rerun_status"],
                        row["run_completion_status"],
                        row["task_id"],
                        row["variant_path"],
                        row["latest_run_folder"],
                        row["latest_result_status"],
                        row["proof"],
                    ]
                    for row in rerun_rows
                ],
            )
        )
    else:
        lines.append("No latest run errors or missing tasks.")

    lines.extend(
        [
            "",
            "## Latest Run Per Expanded Task",
            "",
            markdown_table(
                [
                    "rerun_status",
                    "run_completion_status",
                    "judge_outcome",
                    "task_id",
                    "variant_path",
                    "latest_run_folder",
                    "present_runs",
                    "completed_runs",
                    "latest_completed_run_folder",
                ],
                [
                    [
                        row["rerun_status"],
                        row["run_completion_status"],
                        row["judge_outcome"],
                        row["task_id"],
                        row["variant_path"],
                        row["latest_run_folder"],
                        row["present_run_count"],
                        row["completed_run_count"],
                        row["latest_completed_run_folder"],
                    ]
                    for row in rows
                ],
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path = output_dir / f"{benchmark_name}_latest_run_status_summary.json"
    summary = {
        "spec_file": str(spec_file),
        "run_root": str(run_root),
        "csv": str(csv_path),
        "report": str(report_path),
        "expanded_tasks": len(rows),
        "by_rerun_status": dict(sorted(counts.items())),
        "by_run_completion_status": dict(sorted(completion_counts.items())),
        "by_judge_outcome": dict(sorted(judge_counts.items())),
        "by_source_task_id": {
            source_task_id: dict(sorted(counts_for_source.items()))
            for source_task_id, counts_for_source in sorted(by_source.items())
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-file", type=Path, required=True)
    parser.add_argument("--benchmark-name")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    benchmark_name = args.benchmark_name or args.spec_file.stem
    run_root = args.run_root or TOOL_DIR / "output" / "runs" / benchmark_name
    rows = build_rows(args.spec_file, run_root)
    write_outputs(rows, args.output_dir, args.spec_file, run_root, benchmark_name)

    counts = Counter(row["rerun_status"] for row in rows)
    print(f"expanded_tasks={len(rows)}")
    for key in ("passed", "run_error", "missing_run"):
        print(f"{key}={counts.get(key, 0)}")
    print(f"wrote={args.output_dir / f'{benchmark_name}_latest_run_status.csv'}")
    print(f"wrote={args.output_dir / f'{benchmark_name}_latest_run_status_report.md'}")


if __name__ == "__main__":
    main()
