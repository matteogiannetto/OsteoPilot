from __future__ import annotations

import ast
import math
import re
import statistics
from fractions import Fraction
from typing import Any

import sympy  # type: ignore[import-untyped]
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


VARIABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
IMPLICIT_MULTIPLICATION_RE = re.compile(
    r"(?:(?:\d|\))\s*[A-Za-z_])|(?:[A-Za-z_][A-Za-z0-9_]*\s*\()(?<!sqrt\()"
)
NUMERIC_STRING_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)
RATIONAL_STRING_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?/[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)
DEFAULT_DECIMAL_PRECISION = 12

FORBIDDEN_IDENTIFIERS = {
    "__builtins__",
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "help",
    "type",
    "object",
    "classmethod",
    "staticmethod",
    "property",
    "super",
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "pathlib",
    "shutil",
    "pickle",
    "marshal",
    "importlib",
}

SYMPY_FUNCTIONS = {
    "sqrt": sympy.sqrt,
    "log": sympy.log,
    "ln": sympy.log,
    "exp": sympy.exp,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "Abs": sympy.Abs,
    "abs": sympy.Abs,
    "floor": sympy.floor,
    "ceil": sympy.ceiling,
    "ceiling": sympy.ceiling,
}
CONSTANTS = {
    "pi": sympy.pi,
    "E": sympy.E,
    "e": sympy.E,
    "tau": 2 * sympy.pi,
}
STAT_FUNCTIONS = {"mean", "median", "stdev", "pstdev", "variance", "pvariance", "sum", "min", "max"}
ALLOWED_NAMES = set(SYMPY_FUNCTIONS) | set(CONSTANTS) | STAT_FUNCTIONS
ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        ...,
        description="Python/SymPy-style infix math expression. Use ** for powers and explicit multiplication, e.g. 2*x.",
    )
    variables: dict[str, int | float | str] | None = Field(
        default=None,
        description="Optional numeric variables. String values must represent finite numbers or rationals such as '1/3'.",
    )
    precision: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Optional decimal places to return for non-integer numeric results. "
            "For example, precision=4 returns four digits after the decimal point."
        ),
    )
    return_exact: bool = Field(
        default=False,
        description="Whether to include exact symbolic/rational output when explicitly needed.",
    )


def make_error(
    expression: str,
    variables: dict[str, Any],
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "expression": expression,
        "variables": variables,
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def make_success(
    expression: str,
    variables: dict[str, Any],
    result: dict[str, Any],
    precision: int,
) -> dict[str, Any]:
    return {
        "ok": True,
        "expression": expression,
        "variables": variables,
        "result": result,
        "precision": precision,
    }


def validate_precision(precision: int | None) -> int:
    if precision is None:
        return DEFAULT_DECIMAL_PRECISION
    if not isinstance(precision, int) or isinstance(precision, bool):
        raise ValueError("precision must be an integer.")
    if precision < 1 or precision > 50:
        raise ValueError("precision must be between 1 and 50.")
    return precision


def validate_expression_string(expression: str) -> str:
    if not isinstance(expression, str):
        raise ValueError("expression must be a string.")
    normalized = expression.strip()
    if not normalized:
        raise ValueError("expression must not be empty.")
    return normalized


def contains_unsupported_operator(expression: str) -> bool:
    return "^" in expression


def parse_numeric_variable(value: int | float | str) -> Any:
    if isinstance(value, bool):
        raise ValueError("Boolean variable values are not supported.")
    if isinstance(value, int):
        return sympy.Integer(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Variable values must be finite numbers.")
        return sympy.Rational(str(value))
    if not isinstance(value, str):
        raise ValueError("Variable values must be finite numbers or numeric strings.")

    text = value.strip()
    lower = text.lower()
    if lower in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        raise ValueError("Variable string values must be finite numbers.")
    if RATIONAL_STRING_RE.match(text) or NUMERIC_STRING_RE.match(text):
        parsed = sympy.Rational(text)
        if parsed.q == 0:
            raise ValueError("Rational variable denominators must not be zero.")
        return parsed
    raise ValueError("Variable string values must represent finite numbers or rationals.")


def validate_variables(variables: dict[str, int | float | str] | None) -> dict[str, Any]:
    if variables is None:
        return {}
    if not isinstance(variables, dict):
        raise ValueError("variables must be a dictionary.")

    normalized: dict[str, Any] = {}
    for name, value in variables.items():
        if not isinstance(name, str) or not VARIABLE_NAME_RE.match(name):
            raise ValueError(f"Invalid variable name: {name!r}.")
        if name.startswith("__"):
            raise ValueError(f"Invalid variable name: {name!r}.")
        if name in FORBIDDEN_IDENTIFIERS or name in ALLOWED_NAMES:
            raise ValueError(f"Variable name is reserved or forbidden: {name!r}.")
        normalized[name] = parse_numeric_variable(value)
    return normalized


def build_sympy_namespace(variables: dict[str, Any]) -> dict[str, Any]:
    return {
        **SYMPY_FUNCTIONS,
        **CONSTANTS,
        **variables,
    }


def _public_variables(variables: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in variables.items():
        if isinstance(value, sympy.Integer):
            public[key] = int(value)
        elif isinstance(value, sympy.Rational):
            public[key] = str(value) if value.q != 1 else int(value)
        else:
            public[key] = str(value)
    return public


def _contains_stat_call(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in STAT_FUNCTIONS
        for child in ast.walk(node)
    )


def _validate_ast(node: ast.AST, names: set[str]) -> None:
    for child in ast.walk(node):
        if not isinstance(child, ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported syntax: {type(child).__name__}.")
        if isinstance(child, ast.Name):
            if child.id in FORBIDDEN_IDENTIFIERS or child.id.startswith("__"):
                raise PermissionError(f"Forbidden identifier: {child.id}.")
            if child.id not in names:
                raise NameError(f"Unknown identifier: {child.id}.")
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name):
                raise ValueError("Only direct calls to supported math functions are allowed.")
            if child.func.id not in ALLOWED_NAMES:
                raise ValueError(f"Unsupported function: {child.func.id}.")
            if child.keywords:
                raise ValueError("Keyword arguments are not supported.")
        if isinstance(child, ast.Constant):
            if not isinstance(child.value, (int, float)):
                raise ValueError("Only numeric constants are supported.")
            if isinstance(child.value, float) and not math.isfinite(child.value):
                raise ValueError("Only finite numeric constants are supported.")


def _parse_and_validate_expression(expression: str, variable_names: set[str]) -> ast.Expression:
    if contains_unsupported_operator(expression):
        raise SyntaxError("Use ** for exponentiation. The ^ operator is not supported.")
    if _looks_like_implicit_multiplication(expression):
        raise ValueError("Implicit multiplication is not supported. Use 2*x instead of 2x.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        if "invalid decimal literal" in str(exc):
            raise ValueError("Implicit multiplication is not supported. Use 2*x instead of 2x.") from exc
        raise ValueError("Could not parse expression.") from exc
    _validate_ast(tree, ALLOWED_NAMES | variable_names)
    return tree


def _looks_like_implicit_multiplication(expression: str) -> bool:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return bool(re.search(r"\d\s*[A-Za-z_]", expression))
    tokens = list(ast.walk(tree))
    return any(isinstance(node, ast.Call) and not isinstance(node.func, ast.Name) for node in tokens)


def _constant_to_sympy(value: int | float) -> Any:
    if isinstance(value, int):
        return sympy.Integer(value)
    return sympy.Rational(str(value))


def _eval_sympy_node(node: ast.AST, namespace: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_sympy_node(node.body, namespace)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are supported.")
        return _constant_to_sympy(node.value)
    if isinstance(node, ast.Name):
        return namespace[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _eval_sympy_node(node.operand, namespace)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_sympy_node(node.left, namespace)
        right = _eval_sympy_node(node.right, namespace)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError
            return left % right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = namespace[node.func.id]
        return func(*[_eval_sympy_node(arg, namespace) for arg in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_sympy_node(item, namespace) for item in node.elts]
    raise ValueError("Unsupported expression.")


def _to_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, sympy.Integer):
        return Fraction(int(value), 1)
    if isinstance(value, sympy.Rational):
        return Fraction(int(value.p), int(value.q))
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Only finite statistics inputs are supported.")
        return Fraction(str(value))
    raise ValueError("Statistics inputs must be numeric.")


def _eval_statistics_node(node: ast.AST, namespace: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_statistics_node(node.body, namespace)
    if isinstance(node, ast.Constant):
        return _to_fraction(node.value)
    if isinstance(node, ast.Name):
        return _to_fraction(namespace[node.id])
    if isinstance(node, ast.List | ast.Tuple):
        return [_eval_statistics_node(item, namespace) for item in node.elts]
    if isinstance(node, ast.UnaryOp):
        value = _eval_statistics_node(node.operand, namespace)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_statistics_node(node.left, namespace)
        right = _eval_statistics_node(node.right, namespace)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError
            return left % right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [_eval_statistics_node(arg, namespace) for arg in node.args]
        if name in {"sum", "min", "max", "mean", "median", "stdev", "pstdev", "variance", "pvariance"}:
            if len(args) != 1 or not isinstance(args[0], list):
                raise ValueError(f"{name} requires one explicit list or tuple of numbers.")
            values = args[0]
            if not values:
                raise ValueError(f"{name} requires a non-empty list.")
            if name in {"stdev", "variance"} and len(values) < 2:
                raise ValueError(f"{name} requires at least 2 values.")
            if name == "sum":
                return sum(values, Fraction(0, 1))
            if name == "min":
                return min(values)
            if name == "max":
                return max(values)
            if name == "mean":
                return statistics.mean(values)
            if name == "median":
                return statistics.median(values)
            if name == "stdev":
                return sympy.Rational(str(statistics.stdev(float(v) for v in values)))
            if name == "pstdev":
                return sympy.Rational(str(statistics.pstdev(float(v) for v in values)))
            if name == "variance":
                return statistics.variance(values)
            if name == "pvariance":
                return statistics.pvariance(values)
        return _eval_sympy_node(node, {**build_sympy_namespace(namespace), **SYMPY_FUNCTIONS})
    raise ValueError("Unsupported expression.")


def evaluate_with_sympy(
    expression: str,
    namespace: dict[str, Any],
    precision: int,
    return_exact: bool,
    fixed_decimal_places: bool,
) -> dict[str, Any]:
    tree = _parse_and_validate_expression(expression, set(namespace) - ALLOWED_NAMES)
    value = _eval_sympy_node(tree, namespace)
    return format_result(value, value, precision, return_exact, fixed_decimal_places)


def evaluate_statistics_ast(
    expression: str,
    variables: dict[str, Any],
    precision: int,
    return_exact: bool,
    fixed_decimal_places: bool,
) -> dict[str, Any]:
    namespace = build_sympy_namespace(variables)
    tree = _parse_and_validate_expression(expression, set(variables))
    if not _contains_stat_call(tree):
        value = _eval_sympy_node(tree, namespace)
    else:
        value = _eval_statistics_node(tree, namespace)
    return format_result(value, value, precision, return_exact, fixed_decimal_places)


def _decimal_string(value: Any, precision: int, fixed_decimal_places: bool) -> str:
    sympy_value = sympy.sympify(value)
    if sympy_value.is_integer:
        return str(int(sympy_value))
    if fixed_decimal_places:
        return f"{float(sympy_value.evalf(precision + 5)):.{precision}f}"
    text = str(sympy.N(sympy_value, precision))
    if "e" in text.lower():
        mantissa, exponent = re.split("[eE]", text)
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent)}"
    return text.rstrip("0").rstrip(".")


def _result_type(value: Any) -> str:
    sympy_value = sympy.sympify(value)
    if sympy_value.is_integer:
        return "integer"
    if sympy_value.is_Rational:
        return "rational"
    if sympy_value.is_Boolean:
        return "boolean"
    if sympy_value.is_number:
        return "number"
    return "unknown"


def _validate_result_finite_real(value: Any) -> None:
    sympy_value = sympy.sympify(value)
    if sympy_value in {sympy.nan, sympy.oo, -sympy.oo, sympy.zoo}:
        raise ArithmeticError("non-finite")
    if sympy_value.has(sympy.nan, sympy.oo, -sympy.oo, sympy.zoo):
        raise ArithmeticError("non-finite")
    if sympy_value.is_real is False:
        raise TypeError("complex")
    if sympy_value.evalf().is_real is False:
        raise TypeError("complex")


def format_result(
    value: Any,
    exact: Any | None,
    precision: int,
    return_exact: bool,
    fixed_decimal_places: bool,
) -> dict[str, Any]:
    _validate_result_finite_real(value)
    exact_text = None
    if return_exact and exact is not None:
        exact_text = str(sympy.sympify(exact))
    result = {
        "decimal": _decimal_string(value, precision, fixed_decimal_places),
        "type": _result_type(value),
    }
    if exact_text is not None:
        result["exact"] = exact_text
    return result


def _error_from_exception(
    expression: str,
    variables: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    if isinstance(exc, SyntaxError) and "Use **" in str(exc):
        return make_error(
            expression,
            variables,
            "UnsupportedOperatorError",
            "Use ** for exponentiation. The ^ operator is not supported.",
        )
    if isinstance(exc, ZeroDivisionError):
        return make_error(expression, variables, "DivisionByZeroError", "Division by zero.")
    if isinstance(exc, ArithmeticError) and "non-finite" in str(exc):
        return make_error(
            expression,
            variables,
            "NonFiniteResultError",
            "The expression produced a non-finite result.",
        )
    if isinstance(exc, TypeError) and "complex" in str(exc):
        return make_error(
            expression,
            variables,
            "ComplexResultError",
            "Complex results are not supported by default.",
        )
    if isinstance(exc, PermissionError):
        return make_error(expression, variables, "SecurityError", str(exc))
    if isinstance(exc, NameError):
        return make_error(expression, variables, "NameError", str(exc))
    if isinstance(exc, ValueError):
        message = str(exc)
        error_type = "ParseError" if "Implicit multiplication" in message or "parse" in message.lower() else "ValidationError"
        return make_error(expression, variables, error_type, message)
    return make_error(expression, variables, "EvaluationError", str(exc))


def evaluate_math_expression_core(
    expression: str,
    variables: dict[str, int | float | str] | None = None,
    precision: int | None = None,
    return_exact: bool = True,
) -> dict[str, Any]:
    original_expression = expression if isinstance(expression, str) else str(expression)
    public_variables: dict[str, Any] = {}
    try:
        normalized_expression = validate_expression_string(expression)
        fixed_decimal_places = precision is not None
        normalized_precision = validate_precision(precision)
        normalized_variables = validate_variables(variables)
        public_variables = _public_variables(normalized_variables)
        namespace = build_sympy_namespace(normalized_variables)
        tree = _parse_and_validate_expression(normalized_expression, set(normalized_variables))
        result = (
            evaluate_statistics_ast(
                normalized_expression,
                normalized_variables,
                normalized_precision,
                return_exact,
                fixed_decimal_places,
            )
            if _contains_stat_call(tree)
            else evaluate_with_sympy(
                normalized_expression,
                namespace,
                normalized_precision,
                return_exact,
                fixed_decimal_places,
            )
        )
        return make_success(normalized_expression, public_variables, result, normalized_precision)
    except Exception as exc:
        return _error_from_exception(original_expression, public_variables, exc)


class MathCalculatorTool(BaseTool):
    name: str = "evaluate_math_expression"
    description: str = (
        "Safely evaluates deterministic mathematical expressions with real computation. "
        "Use this tool for all arithmetic, percentages, ratios, powers, logarithms, trigonometry, "
        "and simple statistics instead of calculating in your own reasoning. Expressions must use "
        "Python/SymPy infix syntax, explicit multiplication like 2*x, and ** for powers. Returns decimal "
        "string results or structured safety/parse errors. Exact symbolic/rational output is optional. "
        "To round a result, do NOT call round()/float() inside the expression (unsupported); instead pass "
        "the `precision` argument, e.g. expression='2/3', precision=3 -> 0.667."
    )
    args_schema: type[BaseModel] = CalculatorArgs

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        args = CalculatorArgs(**kwargs)
        return evaluate_math_expression_core(
            args.expression,
            variables=args.variables,
            precision=args.precision,
            return_exact=args.return_exact,
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("Async execution is not supported.")


CalculatorArgs.model_rebuild(
    _types_namespace={
        "Field": Field,
    }
)
MathCalculatorTool.model_rebuild(
    _types_namespace={
        "BaseModel": BaseModel,
        "CalculatorArgs": CalculatorArgs,
    }
)


EXPORTED_TOOLS: dict[str, BaseTool] = {
    "evaluate_math_expression": MathCalculatorTool(),
}
