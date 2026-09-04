#!/usr/bin/env bash
set -u

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$TOOL_DIR/.." && pwd)"
DEFAULT_OSTEOBENCH_DIR="$REPOSITORY_ROOT/osteobench"
OSTEOBENCH_DIR="${OSTEOBENCH_DIR:-$DEFAULT_OSTEOBENCH_DIR}"
EVALUATOR="$TOOL_DIR/evaluator.py"
RUN_INDEX_TOOL="$TOOL_DIR/run_index.py"
if [ ! -d "$OSTEOBENCH_DIR" ]; then
  echo "OsteoBench directory not found: $OSTEOBENCH_DIR"
  echo "Set OSTEOBENCH_DIR only to override the bundled $DEFAULT_OSTEOBENCH_DIR specifications."
  exit 1
fi
JSON_DIR="$OSTEOBENCH_DIR"
if [ -n "${EVALUATION_PYTHON:-}" ]; then
  PYTHON_BIN="$EVALUATION_PYTHON"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi
EVALUATION_GRAPH="${EVALUATION_GRAPH:-full_weak}"
# Each graph gets its own output tree so runs never collide.
if [ "$EVALUATION_GRAPH" = "full_weak" ]; then
  _default_export_root="$TOOL_DIR/output/runs"
  _default_run_index_csv="$TOOL_DIR/output/run_index.csv"
else
  _default_export_root="$TOOL_DIR/output/runs_${EVALUATION_GRAPH}"
  _default_run_index_csv="$TOOL_DIR/output/run_index_${EVALUATION_GRAPH}.csv"
fi
EXPORT_ROOT="${EVALUATION_DIR:-$_default_export_root}"
RUN_INDEX_CSV="${EVALUATION_RUN_INDEX_CSV:-$_default_run_index_csv}"
RUN_INDEX_ENABLED="${EVALUATION_RUN_INDEX_ENABLED:-0}"
RUN_REQUESTED_ONLY="${EVALUATION_RUN_REQUESTED_ONLY:-0}"
SUCCESS_TOKEN="SUCCESS"
TASK_IDS_RAW="${EVALUATION_TASK_IDS:-}"

current_status_file=""

mark_suspended() {
  if [ -n "$current_status_file" ]; then
    {
      echo "SUSPENDED"
      echo "updated_at=$(date '+%Y-%m-%d %H:%M:%S')"
    } > "$current_status_file"
  fi
}

trap 'mark_suspended; echo; echo "Run suspended."; exit 130' INT TERM

safe_name() {
  basename "$1" .json
}

is_truthy() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

is_excluded() {
  case "$(safe_name "$1")" in
    test)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

status_for_name() {
  local name="$1"
  local status_file="$EXPORT_ROOT/$name/RUN_STATUS.txt"
  if [ -f "$status_file" ] && grep -q "^$SUCCESS_TOKEN$" "$status_file"; then
    echo "completed"
  elif [ -f "$status_file" ] && grep -q "^PARTIAL_SUCCESS$" "$status_file"; then
    echo "partial"
  elif [ -f "$status_file" ] && grep -q "^FAILED$" "$status_file"; then
    echo "failed"
  elif [ -f "$status_file" ]; then
    echo "suspended"
  else
    echo "new"
  fi
}

latest_batch_dir() {
  local name="$1"
  local spec_out="$EXPORT_ROOT/$name"
  ls -td "$spec_out"/"$name"_* 2>/dev/null | head -n 1
}

build_task_id_args() {
  local raw="$1"
  local -n out_ref="$2"

  out_ref=()
  if [ -z "$(echo "$raw" | tr -d '[:space:],')" ]; then
    return 0
  fi

  local normalized
  normalized="$(echo "$raw" | tr ',' ' ')"
  read -ra task_ids <<< "$normalized"
  for task_id in "${task_ids[@]}"; do
    if [ -n "$task_id" ]; then
      out_ref+=("--task-id" "$task_id")
    fi
  done
}

mkdir -p "$EXPORT_ROOT"
task_id_args=()
build_task_id_args "$TASK_IDS_RAW" task_id_args

specs=()
while IFS= read -r spec; do
  if ! is_excluded "$spec"; then
    specs+=("$spec")
  fi
done < <(find "$JSON_DIR" -maxdepth 1 -type f -name '*.json' | sort)

if [ "${#specs[@]}" -eq 0 ]; then
  echo "No JSON specs found in $JSON_DIR"
  exit 1
fi

if is_truthy "$RUN_REQUESTED_ONLY" && ! is_truthy "$RUN_INDEX_ENABLED"; then
  echo "EVALUATION_RUN_REQUESTED_ONLY requires EVALUATION_RUN_INDEX_ENABLED=1"
  exit 1
fi

if is_truthy "$RUN_INDEX_ENABLED"; then
  run_index_init_args=()
  for spec in "${specs[@]}"; do
    run_index_init_args+=("--spec-file" "$spec")
  done
  "$PYTHON_BIN" "$RUN_INDEX_TOOL" --csv "$RUN_INDEX_CSV" init "${run_index_init_args[@]}" >/dev/null
fi

echo "Evaluation group: $EXPORT_ROOT  [graph=$EVALUATION_GRAPH]"
if is_truthy "$RUN_INDEX_ENABLED"; then
  echo "Run index CSV: $RUN_INDEX_CSV"
  if is_truthy "$RUN_REQUESTED_ONLY"; then
    echo "Run request filter: enabled"
  fi
fi
echo
echo "Available JSON specs:"
for i in "${!specs[@]}"; do
  number=$((i + 1))
  name="$(safe_name "${specs[$i]}")"
  printf "  %2d) %-40s [%s]\n" "$number" "$name" "$(status_for_name "$name")"
done

echo
printf "Select specs to run, for example 1,3 or all: "
read -r selection

selected_indexes=()
if [ -z "$(echo "$selection" | tr -d '[:space:]')" ]; then
  echo "No specs selected."
  exit 1
fi

if [ "$selection" = "all" ] || [ "$selection" = "ALL" ]; then
  for i in "${!specs[@]}"; do
    selected_indexes+=("$i")
  done
else
  IFS=',' read -ra requested <<< "$selection"
  for raw in "${requested[@]}"; do
    item="$(echo "$raw" | tr -d '[:space:]')"
    if ! echo "$item" | grep -Eq '^[0-9]+$'; then
      echo "Invalid selection: $raw"
      exit 1
    fi
    index=$((item - 1))
    if [ "$index" -lt 0 ] || [ "$index" -ge "${#specs[@]}" ]; then
      echo "Selection out of range: $item"
      exit 1
    fi
    selected_indexes+=("$index")
  done
fi

if [ "${#selected_indexes[@]}" -eq 0 ]; then
  echo "No specs selected."
  exit 1
fi

echo
if is_truthy "$RUN_INDEX_ENABLED"; then
  echo "Starting selected specs. Run index controls task-level skipping."
else
  echo "Starting selected specs. Completed specs are skipped."
fi
if [ "${#task_id_args[@]}" -gt 0 ]; then
  echo "Task id filter: $TASK_IDS_RAW"
fi
echo

for index in "${selected_indexes[@]}"; do
  spec="${specs[$index]}"
  name="$(safe_name "$spec")"
  spec_out="$EXPORT_ROOT/$name"
  status_file="$spec_out/RUN_STATUS.txt"
  current_task_id_args=("${task_id_args[@]}")
  current_task_ids_label="$TASK_IDS_RAW"

  mkdir -p "$spec_out"

  if is_truthy "$RUN_INDEX_ENABLED"; then
    plan_args=(--csv "$RUN_INDEX_CSV" plan --spec-file "$spec")
    if is_truthy "$RUN_REQUESTED_ONLY"; then
      plan_args+=(--requested-only)
    fi
    if [ -n "$(echo "$TASK_IDS_RAW" | tr -d '[:space:],')" ]; then
      plan_args+=(--task-ids-raw "$TASK_IDS_RAW")
    fi

    mapfile -t planned_task_ids < <("$PYTHON_BIN" "$RUN_INDEX_TOOL" "${plan_args[@]}")
    current_task_id_args=()
    if [ "${#planned_task_ids[@]}" -eq 0 ]; then
      echo "[SKIP] $name has no tasks selected by the run index."
      continue
    fi
    current_task_ids_label="$(IFS=,; echo "${planned_task_ids[*]}")"
    for planned_task_id in "${planned_task_ids[@]}"; do
      current_task_id_args+=("--task-id" "$planned_task_id")
    done
    echo "[PLAN] $name -> ${#planned_task_ids[@]} task(s) selected by run index"
  elif [ "${#task_id_args[@]}" -eq 0 ] && [ -f "$status_file" ] && grep -q "^$SUCCESS_TOKEN$" "$status_file"; then
    echo "[SKIP] $name already completed. Remove $status_file to rerun it."
    continue
  fi

  current_status_file="$status_file"
  {
    echo "STARTED"
    echo "spec=$spec"
    if [ "${#current_task_id_args[@]}" -gt 0 ]; then
      echo "task_ids=$current_task_ids_label"
    fi
    if is_truthy "$RUN_INDEX_ENABLED"; then
      echo "run_index=$RUN_INDEX_CSV"
    fi
    echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  } > "$status_file"

  echo "[RUN] $name"
  evaluator_cmd=(
    "$PYTHON_BIN" "$EVALUATOR"
    --spec-file "$spec"
    --render-task-templates
    --human
    --run-prefix "$name"
    --evaluator-runs-dir "$spec_out"
    --graph "$EVALUATION_GRAPH"
  )
  if [ "${#current_task_id_args[@]}" -gt 0 ]; then
    evaluator_cmd+=("${current_task_id_args[@]}")
  fi
  "${evaluator_cmd[@]}"
  evaluator_exit_code=$?

  batch_dir="$(latest_batch_dir "$name")"
  summary_file="$batch_dir/summary.json"

  if is_truthy "$RUN_INDEX_ENABLED" && [ -n "$batch_dir" ] && [ -f "$summary_file" ]; then
    "$PYTHON_BIN" "$RUN_INDEX_TOOL" --csv "$RUN_INDEX_CSV" update --spec-file "$spec" --summary "$summary_file"
  fi

  if [ -n "$batch_dir" ] && [ -f "$summary_file" ] && [ "$evaluator_exit_code" -eq 0 ]; then
    cp "$summary_file" "$spec_out/latest_summary.json"
    {
      if [ "${#current_task_id_args[@]}" -gt 0 ]; then
        echo "PARTIAL_SUCCESS"
      else
        echo "$SUCCESS_TOKEN"
      fi
      echo "spec=$spec"
      if [ "${#current_task_id_args[@]}" -gt 0 ]; then
        echo "task_ids=$current_task_ids_label"
      fi
      if is_truthy "$RUN_INDEX_ENABLED"; then
        echo "run_index=$RUN_INDEX_CSV"
      fi
      echo "batch_dir=$batch_dir"
      echo "summary=$summary_file"
      echo "latest_summary=$spec_out/latest_summary.json"
      echo "completed_at=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "evaluator_exit_code=$evaluator_exit_code"
    } > "$status_file"
    if [ "${#current_task_id_args[@]}" -gt 0 ]; then
      echo "[PARTIAL DONE] $name -> $summary_file"
    else
      echo "[DONE] $name -> $summary_file"
    fi
  elif [ -n "$batch_dir" ] && [ -f "$summary_file" ]; then
    cp "$summary_file" "$spec_out/latest_summary.json"
    {
      echo "FAILED"
      echo "spec=$spec"
      if [ "${#current_task_id_args[@]}" -gt 0 ]; then
        echo "task_ids=$current_task_ids_label"
      fi
      if is_truthy "$RUN_INDEX_ENABLED"; then
        echo "run_index=$RUN_INDEX_CSV"
      fi
      echo "batch_dir=$batch_dir"
      echo "summary=$summary_file"
      echo "latest_summary=$spec_out/latest_summary.json"
      echo "updated_at=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "evaluator_exit_code=$evaluator_exit_code"
      echo "reason=evaluator completed but reported failing tasks"
    } > "$status_file"
    echo "[FAILED] $name -> $summary_file"
  else
    {
      echo "SUSPENDED"
      echo "spec=$spec"
      echo "updated_at=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "evaluator_exit_code=$evaluator_exit_code"
      echo "reason=no summary.json was produced"
    } > "$status_file"
    echo "[SUSPENDED] $name did not produce summary.json"
  fi

  current_status_file=""
  echo
done

echo "Group folder ready at: $EXPORT_ROOT"
