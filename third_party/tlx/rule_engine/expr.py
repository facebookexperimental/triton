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

import math


def select(cond, if_true, if_false):
    """Ternary helper: returns if_true when cond is truthy, else if_false."""
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
