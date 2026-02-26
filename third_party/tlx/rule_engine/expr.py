"""Expression evaluator for the rule engine.

Uses Python's eval() with a restricted namespace containing only safe
math builtins and a select() ternary function. The expression language
is simple enough that a C++ or Rust evaluator can be written as a
~200-line recursive descent parser when needed.

Supported operations:
  - Arithmetic: +, -, *, /, //, %
  - Comparisons: >, <, >=, <=, ==, !=
  - Boolean: and, or, not
  - Builtins: min, max, ceil, floor, abs
  - Ternary: select(cond, if_true, if_false)
"""

import ast
import math


def select(cond, if_true, if_false):
    """Ternary helper: returns if_true when cond is truthy, else if_false.

    NOTE: Both branches are eagerly evaluated (standard Python call semantics).
    Do not rely on short-circuit behaviour — e.g. ``select(N > 0, M / N, 0)``
    will raise ZeroDivisionError when N == 0.  Guard with ``max(N, 1)`` instead.
    """
    return if_true if cond else if_false


# The fixed set of names available to expressions.
_BUILTINS = {
    "min": min,
    "max": max,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "select": select,
    "True": True,
    "False": False,
}

_BUILTIN_NAMES = frozenset(_BUILTINS)

# AST node types that are allowed in expressions.  Anything not in this
# set is rejected at load time, ensuring the expression language stays
# portable to non-Python evaluators.
_ALLOWED_NODES = frozenset({
    # structural
    ast.Expression,
    ast.Module,
    # literals
    ast.Constant,
    # variables
    ast.Name,
    ast.Load,
    # arithmetic & comparisons
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.BoolOp,
    # operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    # function calls (only to whitelisted builtins — checked separately)
    ast.Call,
})


def validate_expr(expr, valid_names, context=""):
    """Validate an expression at load time.

    Checks:
    1. The expression parses as valid Python.
    2. Only whitelisted AST node types are used (no attribute access,
       subscripts, lambdas, comprehensions, imports, etc.).
    3. All referenced names are in *valid_names* or builtins.
    4. Function calls are only to whitelisted builtins.

    Args:
        expr: Expression string to validate.
        valid_names: Set of names that are allowed (inputs + features
                     defined before this expression).
        context: Human-readable location for error messages
                 (e.g. "feature 'mn_ratio'").

    Raises:
        ValueError: If validation fails.
    """
    prefix = f"{context}: " if context else ""

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"{prefix}invalid syntax in expression: {expr!r}: {e}") from None

    all_valid = valid_names | _BUILTIN_NAMES

    for node in ast.walk(tree):
        # Check node type whitelist.
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"{prefix}disallowed construct {type(node).__name__} "
                             f"in expression: {expr!r}")

        # Check name references.
        if isinstance(node, ast.Name) and node.id not in all_valid:
            raise ValueError(f"{prefix}unknown name {node.id!r} in expression: {expr!r}. "
                             f"Valid names: {sorted(all_valid)}")

        # Check function calls are only to builtins.
        if isinstance(node, ast.Call):
            func = node.func
            if not isinstance(func, ast.Name):
                raise ValueError(f"{prefix}only direct function calls are allowed "
                                 f"(no method calls or computed callables) in: {expr!r}")
            if func.id not in _BUILTIN_NAMES:
                raise ValueError(
                    f"{prefix}function {func.id!r} is not a builtin "
                    f"in expression: {expr!r}. "
                    f"Allowed functions: {sorted(n for n in _BUILTIN_NAMES if callable(_BUILTINS.get(n)))}")


def make_namespace(variables):
    """Build a restricted eval namespace from *variables*.

    Args:
        variables: dict mapping names to their current values
                   (inputs + already-computed features).

    Returns:
        A dict suitable for passing as *globals* to eval(), with
        __builtins__ suppressed.
    """
    ns = {"__builtins__": {}}
    ns.update(_BUILTINS)
    ns.update(variables)
    return ns


def eval_expr(expr, variables):
    """Evaluate a single expression string in a restricted namespace.

    Args:
        expr: A Python expression string (e.g. "M / max(N, 1)").
        variables: dict of names → values available to the expression.

    Returns:
        The result of evaluating the expression.

    Raises:
        NameError: if the expression references an undefined name.
        SyntaxError: if the expression is not valid Python.
    """
    ns = make_namespace(variables)
    return eval(expr, ns)  # noqa: S307
