#!/usr/bin/env python3
"""Maintain the evaluation per-run CSV index."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
from pathlib import Path
from typing import Any


TOOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_DIR.parent
DEFAULT_CSV = TOOL_DIR / "output" / "run_index.csv"
DEFAULT_EVALUATION_DIR = TOOL_DIR / "output" / "runs"

FIELDNAMES = ["run_requested", "run_id", "path"]
COMPLETED_STATUSES = {"passed", "failed"}


def bool_text(value: str | bool | None, *, default: bool = True) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None or value == "":
        return "true" if default else "false"
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return "true"
    if normalized in {"0", "false", "no", "n", "off"}:
        return "false"
    return "true" if default else "false"


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "run_id": row.get("run_id", ""),
                    "path": row.get("path", ""),
                    "run_requested": bool_text(row.get("run_requested"), default=True),
                }
            )
        return rows


def write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def variant_id_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "patient"


def path_variant_id_slug(value: str) -> str:
    path = Path(value.strip().rstrip("/"))
    name = path.name or value.strip()
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = Path(name).stem or name
    if name.endswith("_original"):
        name = name[: -len("_original")]
    return variant_id_slug(name)


def dataset_path_variant_id_slug(value: str) -> str:
    path = Path(value.strip().rstrip("/"))
    if path.parent.name and path.name in {"group_1", "group_2"}:
        return variant_id_slug(f"{path.parent.name}_{path.name}")
    return variant_id_slug(path.name or value.strip())


def task_without_key(task: dict[str, Any], key_to_remove: str) -> dict[str, Any]:
    return {key: value for key, value in task.items() if key != key_to_remove}


def infer_template_patient_id(task: dict[str, Any], patient_ids: list[str]) -> str:
    for field_name in ("sample_id", "sample_identifier"):
        value = task.get(field_name)
        if isinstance(value, str):
            for patient_id in patient_ids:
                if patient_id in value:
                    return patient_id

    serialized = json.dumps(
        task_without_key(task, "patient_id"),
        ensure_ascii=False,
        sort_keys=True,
    )
    scored = [(serialized.count(patient_id), patient_id) for patient_id in patient_ids]
    scored.sort(reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return patient_ids[0]


def replace_patient_id_in_value(
    value: Any,
    template_patient_id: str,
    variant_patient_id: str,
) -> Any:
    if isinstance(value, str):
        return value.replace(template_patient_id, variant_patient_id)
    if isinstance(value, list):
        return [
            replace_patient_id_in_value(item, template_patient_id, variant_patient_id)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: replace_patient_id_in_value(item, template_patient_id, variant_patient_id)
            for key, item in value.items()
        }
    return value


def expand_patient_id_variants(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            expanded_tasks.append(task)
            continue

        patient_id_value = task.get("patient_id")
        if not isinstance(patient_id_value, list):
            expanded_tasks.append(task)
            continue

        patient_ids = []
        seen_patient_ids = set()
        for item in patient_id_value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"task '{task.get('id', '<unknown')}' patient_id list must contain "
                    "only non-empty strings"
                )
            patient_id = item.strip()
            if patient_id not in seen_patient_ids:
                seen_patient_ids.add(patient_id)
                patient_ids.append(patient_id)

        template_patient_id = infer_template_patient_id(task, patient_ids)
        base_task_id = str(task.get("id", "task"))
        uses_patient_id_placeholder = "{patient_id}" in json.dumps(
            task_without_key(task, "patient_id"),
            ensure_ascii=False,
        )
        for patient_id in patient_ids:
            variant = copy.deepcopy(task)
            if not uses_patient_id_placeholder:
                variant = replace_patient_id_in_value(
                    variant,
                    template_patient_id,
                    patient_id,
                )
            variant["patient_id"] = patient_id
            variant["source_task_id"] = base_task_id
            variant["patient_id_template"] = template_patient_id
            if len(patient_ids) > 1:
                variant["id"] = f"{base_task_id}_{variant_id_slug(patient_id)}"
                if isinstance(variant.get("name"), str) and variant["name"]:
                    variant["name"] = f"{variant['name']} [{patient_id}]"
            expanded_tasks.append(variant)

    return expanded_tasks


def expand_list_valued_task_fields(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    list_fields = {
        "volume_relative_path": path_variant_id_slug,
        "mask_relative_path": path_variant_id_slug,
        "dataset_relative_path": dataset_path_variant_id_slug,
    }
    expanded_tasks: list[dict[str, Any]] = []

    for task in tasks:
        variants = [task]
        for field_name, slugger in list_fields.items():
            next_variants: list[dict[str, Any]] = []
            for variant_source in variants:
                if not isinstance(variant_source, dict):
                    next_variants.append(variant_source)
                    continue

                field_value = variant_source.get(field_name)
                if not isinstance(field_value, list):
                    next_variants.append(variant_source)
                    continue

                concrete_values = []
                seen_values = set()
                for item in field_value:
                    if not isinstance(item, str) or not item.strip():
                        raise ValueError(
                            f"task '{variant_source.get('id', '<unknown>')}' "
                            f"{field_name} list must contain only non-empty strings"
                        )
                    concrete_value = item.strip()
                    if concrete_value not in seen_values:
                        seen_values.add(concrete_value)
                        concrete_values.append(concrete_value)

                base_task_id = str(variant_source.get("id", "task"))
                original_source_task_id = str(
                    variant_source.get("source_task_id", base_task_id)
                )
                sample_id_values = variant_source.get("sample_id")
                use_parallel_sample_ids = (
                    isinstance(sample_id_values, list)
                    and len(sample_id_values) == len(field_value)
                    and all(isinstance(item, str) and item.strip() for item in sample_id_values)
                )
                for concrete_value in concrete_values:
                    variant = copy.deepcopy(variant_source)
                    variant[field_name] = concrete_value
                    if use_parallel_sample_ids:
                        original_index = field_value.index(concrete_value)
                        variant["sample_id_template"] = sample_id_values
                        variant["sample_id"] = sample_id_values[original_index].strip()
                    elif field_name == "volume_relative_path" and isinstance(
                        variant.get("sample_id"),
                        str,
                    ):
                        variant["sample_id_template"] = variant["sample_id"]
                        variant["sample_id"] = slugger(concrete_value)
                    variant["source_task_id"] = original_source_task_id
                    variant[f"{field_name}_variant"] = concrete_value
                    if len(concrete_values) > 1:
                        variant["id"] = f"{base_task_id}_{slugger(concrete_value)}"
                    next_variants.append(variant)
            variants = next_variants
        expanded_tasks.extend(variants)

    return expanded_tasks


PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_task_template_value(value: Any, task_context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return PLACEHOLDER_PATTERN.sub(
            lambda match: str(task_context.get(match.group(1), match.group(0))),
            value,
        )
    if isinstance(value, list):
        return [render_task_template_value(item, task_context) for item in value]
    if isinstance(value, dict):
        return {
            key: render_task_template_value(item, task_context)
            for key, item in value.items()
        }
    return value


def preprocess_task_spec(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered_tasks: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            rendered_tasks.append(task)
            continue
        rendered_task = copy.deepcopy(task)
        for _ in range(5):
            next_rendered_task = render_task_template_value(rendered_task, rendered_task)
            if next_rendered_task == rendered_task:
                break
            rendered_task = next_rendered_task
        rendered_tasks.append(rendered_task)
    return rendered_tasks


def safe_path_slug(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return slug or fallback


def load_spec_tasks(spec_file: Path) -> list[dict[str, Any]]:
    tasks = json.loads(spec_file.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("task spec root must be a JSON list")
    return preprocess_task_spec(expand_list_valued_task_fields(expand_patient_id_variants(tasks)))


def run_id_for_task(index: int, task_id: str) -> str:
    return f"{index:03d}_{safe_path_slug(task_id, fallback=f'task_{index:03d}')}"


def spec_entries(spec_file: Path) -> list[dict[str, str]]:
    entries = []
    for index, task in enumerate(load_spec_tasks(spec_file), start=1):
        task_id = str(task["id"])
        entries.append(
            {
                "run_id": run_id_for_task(index, task_id),
                "path": "",
                "run_requested": "true",
            }
        )
    return entries


def ensure_entries(csv_path: Path, spec_files: list[Path]) -> list[dict[str, str]]:
    existing_rows = read_rows(csv_path)
    by_id = {row["run_id"]: row for row in existing_rows if row.get("run_id")}
    ordered_ids = [row["run_id"] for row in existing_rows if row.get("run_id")]

    for spec_file in spec_files:
        for entry in spec_entries(spec_file):
            run_id = entry["run_id"]
            if run_id not in by_id:
                by_id[run_id] = entry
                ordered_ids.append(run_id)

    rows = [by_id[run_id] for run_id in ordered_ids if run_id in by_id]
    write_rows(csv_path, rows)
    return rows


def task_id_from_run_id(run_id: str) -> str:
    parts = run_id.split("_", 1)
    return parts[1] if len(parts) == 2 else run_id


def evaluation_dir() -> Path:
    return Path(os.environ.get("EVALUATION_DIR", DEFAULT_EVALUATION_DIR))


def resolve_index_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return evaluation_dir() / path


def existing_path_is_valid(path_text: str) -> bool:
    if not path_text.strip():
        return False
    return resolve_index_path(path_text).exists()


def parse_task_id_filter(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip(" ,"):
        return None
    return {item for item in raw.replace(",", " ").split() if item}


def plan_tasks(
    csv_path: Path,
    spec_file: Path,
    *,
    requested_only: bool,
    task_ids_raw: str | None,
) -> list[str]:
    rows = ensure_entries(csv_path, [spec_file])
    row_by_id = {row["run_id"]: row for row in rows}
    allowed_task_ids = parse_task_id_filter(task_ids_raw)
    selected: list[str] = []

    for entry in spec_entries(spec_file):
        row = row_by_id.get(entry["run_id"], entry)
        task_id = task_id_from_run_id(entry["run_id"])
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue

        if requested_only:
            if bool_text(row.get("run_requested"), default=True) == "true":
                selected.append(task_id)
            continue

        if bool_text(row.get("run_requested"), default=True) == "false":
            continue

        if not existing_path_is_valid(row.get("path", "")):
            selected.append(task_id)

    return selected


def task_id_from_run_dir_name(folder_name: str) -> str | None:
    match = re.match(r"^\d+_(.+)_run-[0-9a-f]+$", folder_name)
    return match.group(1) if match else None


def rescan_from_dir(csv_path: Path, spec_file: Path, spec_dir: Path) -> None:
    rows = ensure_entries(csv_path, [spec_file])
    row_by_id = {row["run_id"]: row for row in rows}
    run_id_by_task_id = {
        task_id_from_run_id(entry["run_id"]): entry["run_id"]
        for entry in spec_entries(spec_file)
    }

    found: dict[str, Path] = {}
    for batch_dir in sorted(spec_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        for run_dir in sorted(batch_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            task_id = task_id_from_run_dir_name(run_dir.name)
            if task_id and task_id in run_id_by_task_id:
                found[task_id] = run_dir  # last batch dir seen wins

    for task_id, run_dir in found.items():
        run_id = run_id_by_task_id[task_id]
        row = row_by_id.get(run_id)
        if row is None:
            continue
        row["path"] = str(run_dir)
        row["run_requested"] = "false"

    write_rows(csv_path, rows)


def update_from_summary(csv_path: Path, spec_file: Path, summary_path: Path) -> None:
    rows = ensure_entries(csv_path, [spec_file])
    row_by_id = {row["run_id"]: row for row in rows}
    run_id_by_task_id = {
        task_id_from_run_id(entry["run_id"]): entry["run_id"]
        for entry in spec_entries(spec_file)
    }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    for result in summary.get("results", []):
        task_id = result.get("task_id")
        run_id = run_id_by_task_id.get(task_id)
        if not run_id:
            continue
        row = row_by_id.get(run_id)
        if row is None:
            continue
        if result.get("status") in COMPLETED_STATUSES:
            run_dir = result.get("evaluator_run_dir")
            if run_dir:
                row["path"] = str(run_dir)
            row["run_requested"] = "false"

    write_rows(csv_path, rows)


def command_init(args: argparse.Namespace) -> int:
    ensure_entries(args.csv, args.spec_file)
    print(args.csv)
    return 0


def command_plan(args: argparse.Namespace) -> int:
    for task_id in plan_tasks(
        args.csv,
        args.spec_file,
        requested_only=args.requested_only,
        task_ids_raw=args.task_ids_raw,
    ):
        print(task_id)
    return 0


def command_rescan(args: argparse.Namespace) -> int:
    spec_dir = args.spec_dir or (
        Path(os.environ.get("EVALUATION_DIR", str(DEFAULT_EVALUATION_DIR))) / args.spec_file.stem
    )
    rescan_from_dir(args.csv, args.spec_file, spec_dir)
    print(args.csv)
    return 0


def command_update(args: argparse.Namespace) -> int:
    update_from_summary(args.csv, args.spec_file, args.summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--spec-file", action="append", type=Path, required=True)
    init_parser.set_defaults(func=command_init)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--spec-file", type=Path, required=True)
    plan_parser.add_argument("--requested-only", action="store_true")
    plan_parser.add_argument("--task-ids-raw")
    plan_parser.set_defaults(func=command_plan)

    rescan_parser = subparsers.add_parser("rescan")
    rescan_parser.add_argument("--spec-file", type=Path, required=True)
    rescan_parser.add_argument("--spec-dir", type=Path, default=None)
    rescan_parser.set_defaults(func=command_rescan)

    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("--spec-file", type=Path, required=True)
    update_parser.add_argument("--summary", type=Path, required=True)
    update_parser.set_defaults(func=command_update)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
