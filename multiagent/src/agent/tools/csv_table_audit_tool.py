from __future__ import annotations

import math
import operator
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd  # type: ignore[import-untyped]
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.tools.path_resolution import missing_input_path_message, resolve_input_path


CsvOperation = Literal[
    "count",
    "missing",
    "mean",
    "std",
    "min",
    "max",
    "median",
    "sum",
]
DEFAULT_CSV_OPERATIONS: tuple[CsvOperation, ...] = (
    "count",
    "mean",
    "std",
    "min",
    "max",
)
ComparisonOperator = Literal[">", ">=", "<", "<=", "==", "!="]
OPERATOR_ALIASES: dict[str, ComparisonOperator] = {
    "gt": ">",
    "greater_than": ">",
    "gte": ">=",
    "ge": ">=",
    "greater_than_or_equal": ">=",
    "lt": "<",
    "less_than": "<",
    "lte": "<=",
    "le": "<=",
    "less_than_or_equal": "<=",
    "eq": "==",
    "equals": "==",
    "ne": "!=",
    "neq": "!=",
    "not_equal": "!=",
}
THRESHOLD_KEY_OPERATORS: dict[str, ComparisonOperator] = {
    "gt": ">",
    "gte": ">=",
    "ge": ">=",
    "lt": "<",
    "lte": "<=",
    "le": "<=",
    "eq": "==",
    "ne": "!=",
    "neq": "!=",
}


class ConditionalCountSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Output key for this conditional count.",
        min_length=1,
    )
    column: str = Field(
        ...,
        description="CSV column to test.",
    )
    operator: ComparisonOperator = Field(
        ...,
        description=(
            "Comparison operator. Preferred values are the literal symbols: >, >=, <, <=, ==, or !=. "
            "Aliases such as gt, gte, lt, lte, eq, and ne are accepted and normalized."
        ),
    )
    value: int | float | str | bool = Field(
        ...,
        description="Literal value to compare against.",
    )


_CONDITION_RE = re.compile(
    r"^\s*(?P<column>.+?)\s*(?P<operator>>=|<=|==|!=|>|<)\s*(?P<value>.+?)\s*$"
)


def _parse_condition_value(raw_value: str) -> int | float | str | bool:
    value = raw_value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    quoted = (
        (value.startswith("'") and value.endswith("'"))
        or (value.startswith('"') and value.endswith('"'))
    )
    if quoted and len(value) >= 2:
        return value[1:-1]

    try:
        parsed_float = float(value)
    except ValueError:
        return value
    if parsed_float.is_integer() and re.fullmatch(r"[+-]?\d+", value):
        return int(parsed_float)
    return parsed_float


def _condition_from_expression(expression: str) -> dict[str, Any]:
    match = _CONDITION_RE.match(expression)
    if match is None:
        raise ValueError(
            "Conditional count expressions must look like '<column> <operator> <value>', "
            "for example 'Volume_um3 >= 50'."
        )
    return {
        "name": expression.strip(),
        "column": match.group("column").strip(),
        "operator": match.group("operator"),
        "value": _parse_condition_value(match.group("value")),
    }


def _normalize_operator(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    return OPERATOR_ALIASES.get(normalized.lower(), normalized)


def _threshold_condition_name(column: str, operator: ComparisonOperator, value: Any, base_name: Any = None) -> str:
    if isinstance(base_name, str) and base_name.strip():
        return f"{base_name.strip()} {operator} {value}"
    return f"{column} {operator} {value}"


def _normalize_conditional_counts(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
        normalized = []
        for expression, enabled in value.items():
            if enabled:
                normalized.append(_condition_from_expression(str(expression)))
        return normalized

    if not isinstance(value, list):
        return value

    normalized = []
    for item in value:
        if isinstance(item, str):
            normalized.append(_condition_from_expression(item))
            continue
        if isinstance(item, dict):
            spec = dict(item)
            if "operator" not in spec and "op" in spec:
                spec["operator"] = spec.pop("op")
            if "operator" in spec:
                spec["operator"] = _normalize_operator(spec["operator"])
            threshold_keys = [key for key in THRESHOLD_KEY_OPERATORS if key in spec]
            if "column" in spec and threshold_keys:
                for key in threshold_keys:
                    operator_symbol = THRESHOLD_KEY_OPERATORS[key]
                    threshold_value = spec[key]
                    normalized.append(
                        {
                            "name": _threshold_condition_name(
                                str(spec["column"]),
                                operator_symbol,
                                threshold_value,
                                spec.get("name"),
                            ),
                            "column": spec["column"],
                            "operator": operator_symbol,
                            "value": threshold_value,
                        }
                    )
                continue
            if "name" not in spec and {"column", "operator", "value"}.issubset(spec):
                spec["name"] = f"{spec['column']} {spec['operator']} {spec['value']}"
            normalized.append(spec)
            continue
        normalized.append(item)
    return normalized


class CsvTableAuditArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: str = Field(
        ...,
        description="CSV file path, absolute or relative to the active session/workspace.",
    )
    session_path: str = Field(
        ...,
        description="INTERNAL: absolute active session path injected by the subgraph.",
    )
    workspace_root: str | None = Field(
        default=None,
        description="INTERNAL: optional read-only workspace root for resolving relative paths.",
    )
    columns: list[str] | None = Field(
        default=None,
        description="Optional columns to aggregate. If omitted, numeric columns are used.",
    )
    operations: list[CsvOperation] = Field(
        default_factory=lambda: list(DEFAULT_CSV_OPERATIONS),
        description="Aggregate operations to compute per selected column.",
    )
    include_audit: bool = Field(
        default=True,
        description=(
            "Whether to include CSV structure metadata: row/column shape, exact column names, "
            "dtypes, numeric columns, and missing-value counts. Keep true when you need to "
            "inspect or discover exact column names before passing columns, group_by, or "
            "conditional_counts."
        ),
    )
    group_by: list[str] | None = Field(
        default=None,
        description="Optional categorical columns for grouped aggregation.",
    )
    max_groups: int = Field(
        default=25,
        ge=1,
        le=200,
        description="Maximum grouped aggregate rows to return.",
    )
    preview_rows: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Optional first-row preview count. Use 0 unless needed; 1-10 is usually enough, maximum 50.",
    )
    std_ddof: int = Field(
        default=1,
        ge=0,
        le=1,
        description="Delta degrees of freedom for std: 1 for sample std, 0 for population std.",
    )
    conditional_counts: list[ConditionalCountSpec] | None = Field(
        default=None,
        description=(
            "Optional declarative row counts. Preferred forms are a string list like "
            "['Volume [um3] > 50', 'Volume [um3] < 5000'] or objects like "
            "{'name': 'large_volume', 'column': 'Volume [um3]', 'operator': '>', 'value': 50}. "
            "Use literal operator symbols >, >=, <, <=, ==, !=; aliases gt/gte/lt/lte/eq/ne are "
            "accepted. Shorthand objects such as {'column': 'Volume [um3]', 'gt': 50, 'lt': 5000} "
            "are expanded into separate simple threshold counts, not a combined between-count. "
            "Each condition excludes missing values in its target column and returns count, "
            "valid_count, and fraction."
        ),
    )

    @field_validator("conditional_counts", mode="before")
    @classmethod
    def normalize_conditional_counts(cls, value: Any) -> Any:
        return _normalize_conditional_counts(value)


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _audit(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    numeric_columns = list(df.select_dtypes(include="number").columns)
    return {
        "path": str(path),
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "numeric_columns": numeric_columns,
        "missing_by_column": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if int(count) > 0
        },
    }


def _select_columns(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested is None:
        return list(df.select_dtypes(include="number").columns)

    missing = [column for column in requested if column not in df.columns]
    if missing:
        raise ValueError(f"Unknown CSV columns: {missing}")
    return list(requested)


def _validate_group_by(df: pd.DataFrame, group_by: list[str] | None) -> list[str]:
    if not group_by:
        return []
    missing = [column for column in group_by if column not in df.columns]
    if missing:
        raise ValueError(f"Unknown group_by columns: {missing}")
    return list(group_by)


def _canonical_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column.lower())


def _resolve_condition_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested

    requested_key = _canonical_column_name(requested)
    matches = [
        column
        for column in df.columns
        if _canonical_column_name(str(column)) == requested_key
    ]
    if len(matches) == 1:
        return str(matches[0])
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous conditional count column {requested!r}; matching columns: {matches}"
        )
    raise ValueError(f"Unknown conditional count column: {requested!r}")


def _column_aggregates(
    df: pd.DataFrame,
    columns: list[str],
    operations: list[CsvOperation],
    std_ddof: int,
) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    numeric_ops = {"mean", "std", "min", "max", "median", "sum"}

    for column in columns:
        series = df[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        column_result: dict[str, Any] = {}
        for operation in operations:
            if operation == "count":
                column_result["count"] = int(series.count())
            elif operation == "missing":
                column_result["missing"] = int(series.isna().sum())
            elif operation in numeric_ops:
                if not is_numeric:
                    column_result[operation] = None
                    continue
                if operation == "mean":
                    value = series.mean()
                elif operation == "std":
                    value = series.std(ddof=std_ddof)
                elif operation == "min":
                    value = series.min()
                elif operation == "max":
                    value = series.max()
                elif operation == "median":
                    value = series.median()
                else:
                    value = series.sum()
                column_result[operation] = _json_value(value)
        aggregates[column] = column_result

    return aggregates


def _grouped_aggregates(
    df: pd.DataFrame,
    columns: list[str],
    operations: list[CsvOperation],
    group_by: list[str],
    max_groups: int,
    std_ddof: int,
) -> list[dict[str, Any]]:
    if not group_by:
        return []

    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_by, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = {
            "group": {
                column: _json_value(value)
                for column, value in zip(group_by, key_values)
            },
            "rows": int(len(group)),
            "aggregates": _column_aggregates(group, columns, operations, std_ddof),
        }
        rows.append(row)
        if len(rows) >= max_groups:
            break
    return rows


def _compare_series(series: pd.Series, op: ComparisonOperator, value: Any) -> pd.Series:
    comparators = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }
    return comparators[op](series, value)


def _conditional_counts(
    df: pd.DataFrame,
    specs: list[ConditionalCountSpec] | None,
) -> dict[str, dict[str, Any]]:
    if not specs:
        return {}

    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.name in results:
            raise ValueError(f"Duplicate conditional count name: {spec.name!r}")
        column = _resolve_condition_column(df, spec.column)

        series = df[column].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        value = spec.value
        if is_numeric and isinstance(value, str):
            try:
                value = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"Condition {spec.name!r} compares numeric column {spec.column!r} "
                    f"with non-numeric value {spec.value!r}"
                ) from exc
        if not is_numeric and spec.operator in {">", ">=", "<", "<="}:
            raise ValueError(
                f"Condition {spec.name!r} uses ordered comparison on non-numeric column {spec.column!r}"
            )

        mask = _compare_series(series, spec.operator, value)
        count = int(mask.sum())
        valid_count = int(series.count())
        results[spec.name] = {
            "column": column,
            "requested_column": spec.column,
            "operator": spec.operator,
            "value": _json_value(value),
            "count": count,
            "valid_count": valid_count,
            "missing_excluded": int(df[column].isna().sum()),
            "fraction": (count / valid_count) if valid_count else None,
        }
    return results


def csv_table_audit_core(
    csv_path: str,
    *,
    session_path: str,
    workspace_root: str | None = None,
    columns: list[str] | None = None,
    operations: list[CsvOperation] | None = None,
    include_audit: bool = True,
    group_by: list[str] | None = None,
    max_groups: int = 25,
    preview_rows: int = 0,
    std_ddof: int = 1,
    conditional_counts: Any = None,
) -> dict[str, Any]:
    operations = operations or list(DEFAULT_CSV_OPERATIONS)
    resolved = resolve_input_path(
        csv_path,
        session_path=session_path,
        workspace_root=workspace_root,
    )
    if not resolved.found or resolved.path is None:
        return {
            "success": False,
            "error": missing_input_path_message("CSV file", csv_path, resolved.searched_paths),
        }

    path = resolved.path
    if path.suffix.lower() != ".csv":
        return {
            "success": False,
            "error": f"Expected a .csv file, got: {path}",
        }

    try:
        df = _read_csv(path)
        selected_columns = _select_columns(df, columns)
        group_columns = _validate_group_by(df, group_by)
        normalized_counts = _normalize_conditional_counts(conditional_counts)
        count_specs = [
            spec if isinstance(spec, ConditionalCountSpec) else ConditionalCountSpec(**spec)
            for spec in (normalized_counts or [])
        ]
        result: dict[str, Any] = {
            "success": True,
            "row_count": int(len(df)),
            "selected_columns": selected_columns,
            "operations": list(operations),
        }
        if include_audit:
            result["audit"] = _audit(df, path)
        result["aggregates"] = _column_aggregates(df, selected_columns, operations, std_ddof)
        if group_columns:
            result["grouped_aggregates"] = _grouped_aggregates(
                df,
                selected_columns,
                operations,
                group_columns,
                max_groups,
                std_ddof,
            )
            result["grouped_aggregates_truncated"] = df.groupby(group_columns, dropna=False).ngroups > max_groups
        if count_specs:
            result["conditional_counts"] = _conditional_counts(df, count_specs)
        if preview_rows:
            result["preview"] = [
                {column: _json_value(value) for column, value in row.items()}
                for row in df.head(preview_rows).to_dict(orient="records")
            ]
        return result
    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


class CsvTableAuditTool(BaseTool):
    name: str = "csv_table_audit"
    description: str = (
        "Audit a CSV and compute concise aggregates. Returns shape, column names, dtypes, "
        "numeric columns, missing counts, and per-column count/mean/std/min/max/median/sum; "
        "optionally groups by categorical columns and computes safe declarative conditional "
        "row counts such as one column > a literal threshold. For conditional_counts, prefer "
        "string conditions like ['Volume [um3] > 50'] or object conditions with keys name, column, "
        "operator, value; operators should be >, >=, <, <=, ==, or !=, with gt/gte/lt/lte aliases "
        "accepted. Use for CSV outputs when summary "
        "statistics, threshold counts, fractions, or structural verification are needed. Use "
        "include_audit=true to discover exact CSV column names before requesting column "
        "aggregates, group_by, or conditional_counts; do not use it for arbitrary row-level "
        "code execution."
    )
    args_schema: type[BaseModel] = CsvTableAuditArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        args = CsvTableAuditArgs(**kwargs)
        return csv_table_audit_core(**args.model_dump())

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


ConditionalCountSpec.model_rebuild(
    _types_namespace={
        "Field": Field,
        "ComparisonOperator": ComparisonOperator,
    }
)
CsvTableAuditArgs.model_rebuild(
    _types_namespace={
        "Field": Field,
        "Literal": Literal,
        "CsvOperation": CsvOperation,
        "ComparisonOperator": ComparisonOperator,
        "ConditionalCountSpec": ConditionalCountSpec,
    }
)
CsvTableAuditTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "CsvTableAuditArgs": CsvTableAuditArgs,
        "ConditionalCountSpec": ConditionalCountSpec,
        "ComparisonOperator": ComparisonOperator,
    }
)


EXPORTED_TOOLS: dict[str, BaseTool] = {
    "csv_table_audit": CsvTableAuditTool(),
}
