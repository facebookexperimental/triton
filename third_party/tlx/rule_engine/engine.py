"""RuleEngine: load a rules YAML file, evaluate features, match rules.

The rules file is an ordered list of rules, each with a ``when`` clause
(list of boolean expressions, implicitly ANDed) and a ``config`` dict.
The first rule whose ``when`` clause is satisfied wins.

Typical usage::

    engine = RuleEngine("blackwell_gemm_ws_rules.yaml")
    config = engine.evaluate(M=4096, N=4096, K=4096, num_sms=148)

    # Standalone validation (for use by other parsers):
    validate_rules_yaml("blackwell_gemm_ws_rules.yaml")
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .expr import _BUILTINS, validate_expr

_SUPPORTED_VERSION = 1


def _parse_features(raw_features):
    """Parse features from either list-of-pairs or mapping format.

    Version 1 supports two formats:
    - List of {name, expr} dicts (preferred, order-preserving in all parsers)
    - Ordered mapping of name → expr (legacy, order depends on YAML parser)

    Returns:
        List of (name, expr) tuples in evaluation order.
    """
    if raw_features is None:
        return []

    if isinstance(raw_features, list):
        result = []
        for i, item in enumerate(raw_features):
            if not isinstance(item, dict):
                raise ValueError(f"features[{i}]: expected a mapping with 'name' and 'expr', "
                                 f"got {type(item).__name__}")
            if "name" not in item or "expr" not in item:
                raise ValueError(f"features[{i}]: each feature must have 'name' and 'expr' keys, "
                                 f"got keys: {sorted(item.keys())}")
            result.append((item["name"], item["expr"]))
        return result

    if isinstance(raw_features, dict):
        return list(raw_features.items())

    raise ValueError(f"'features' must be a list of {{name, expr}} pairs or a mapping, "
                     f"got {type(raw_features).__name__}")


def validate_rules_yaml(path: str | Path) -> None:
    """Validate a rules YAML file without constructing a RuleEngine.

    Use this when you have a different parser (C++, Rust, PyTorch, etc.) that
    consumes the same YAML format and you want to validate it with
    the Python reference validator.

    Args:
        path: Path to the rules YAML file.

    Raises:
        ValueError: If the YAML is malformed or any expression is invalid.
    """
    path = Path(path)
    with open(path) as f:
        spec = yaml.safe_load(f)

    version = spec.get("version")
    if version != _SUPPORTED_VERSION:
        raise ValueError(f"Unsupported rules version {version!r} in {path} "
                         f"(expected {_SUPPORTED_VERSION})")

    for key in ("inputs", "rules"):
        if key not in spec:
            raise ValueError(f"Missing required key '{key}' in {path}")

    inputs = spec["inputs"]
    features = _parse_features(spec.get("features"))
    rules = spec["rules"]

    _validate_spec(inputs, features, rules, path)


def _validate_spec(inputs, features, rules, path):
    """Validate all expressions in a parsed rules spec.

    Args:
        inputs: List of input variable names.
        features: List of (name, expr) tuples in evaluation order.
        rules: List of rule dicts with 'when' and 'config' keys.
        path: File path for error messages.

    Raises:
        ValueError: If any expression is invalid or structure is wrong.
    """
    # Names accumulate as features are defined top-to-bottom.
    known_names = set(inputs)

    for name, expr in features:
        validate_expr(expr, known_names, context=f"{path}: feature '{name}'")
        known_names.add(name)

    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"{path}: rules[{idx}]: expected a mapping, "
                             f"got {type(rule).__name__}")

        rule_name = rule.get("name", f"<rules[{idx}]>")

        if "when" not in rule:
            raise ValueError(f"{path}: rule '{rule_name}': missing required key 'when'")
        if not isinstance(rule["when"], list):
            raise ValueError(f"{path}: rule '{rule_name}': 'when' must be a list of expressions, "
                             f"got {type(rule['when']).__name__}")

        if "config" not in rule:
            raise ValueError(f"{path}: rule '{rule_name}': missing required key 'config'")
        if not isinstance(rule["config"], dict):
            raise ValueError(f"{path}: rule '{rule_name}': 'config' must be a mapping, "
                             f"got {type(rule['config']).__name__}")

        for i, cond in enumerate(rule["when"]):
            validate_expr(cond, known_names, context=f"{path}: rule '{rule_name}' when[{i}]")

        # Validate $variable references in config values.
        for key, val in rule["config"].items():
            if isinstance(val, str) and val.startswith("$"):
                ref = val[1:]
                if ref not in known_names:
                    raise ValueError(f"{path}: rule '{rule_name}' config.{key}: "
                                     f"unknown variable reference '${ref}'. "
                                     f"Valid names: {sorted(known_names)}")


def _codegen_evaluate(inputs, features, rules, path):
    """Generate a Python function from features and rules.

    Produces a single ``_evaluate(input1, input2, ...)`` function whose body
    is equivalent to the original inline ``if/elif`` chain.  Compiled once
    at ``RuleEngine.__init__`` time so that ``evaluate()`` runs at the same
    speed as hand-written Python — no per-expression ``eval()`` calls, no
    namespace dict construction, just pre-compiled bytecode with local
    variable lookups.

    Args:
        inputs: List of input variable names.
        features: List of (name, expr) tuples in evaluation order.
        rules: List of rule dicts with 'when' and 'config' keys.
        path: File path (used as the code object's filename for tracebacks).

    Returns:
        A callable ``(input1, input2, ...) -> dict | None``.
    """
    lines = []

    # Function signature: inputs as positional args, builtins as default
    # args for fast local lookup (avoids global/closure overhead).
    callable_builtins = [name for name, val in _BUILTINS.items() if callable(val)]
    builtin_defaults = ", ".join(f"_{name}={name}" for name in callable_builtins)
    input_args = ", ".join(inputs)
    lines.append(f"def _evaluate({input_args}, {builtin_defaults}):")

    # Feature assignments — each becomes a local variable.
    for name, expr in features:
        # Replace bare builtin calls with their _prefixed local versions
        # (e.g. max -> _max) so they use the fast default-arg locals.
        local_expr = expr
        for builtin_name in callable_builtins:
            local_expr = local_expr.replace(f"{builtin_name}(", f"_{builtin_name}(")
        lines.append(f"    {name} = {local_expr}")

    # Rules as if/elif chain.
    first = True
    for rule in rules:
        keyword = "if" if first else "elif"
        first = False
        condition = " and ".join(f"({c})" for c in rule["when"])
        lines.append(f"    {keyword} {condition}:")

        # Build the return dict.
        config_parts = []
        for key, val in rule["config"].items():
            if isinstance(val, str) and val.startswith("$"):
                # $variable reference — use the local variable directly.
                config_parts.append(f"            {key!r}: {val[1:]},")
            else:
                config_parts.append(f"            {key!r}: {val!r},")
        lines.append("        return {")
        lines.extend(config_parts)
        lines.append("        }")

    lines.append("    return None")

    source = "\n".join(lines)
    code = compile(source, str(path), "exec")
    namespace = dict(_BUILTINS)
    exec(code, namespace)  # noqa: S102
    return namespace["_evaluate"]


class RuleEngine:
    """Declarative, first-match rule evaluator driven by a YAML file."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        with open(path) as f:
            spec = yaml.safe_load(f)

        version = spec.get("version")
        if version != _SUPPORTED_VERSION:
            raise ValueError(f"Unsupported rules version {version!r} in {path} "
                             f"(expected {_SUPPORTED_VERSION})")

        for key in ("inputs", "rules"):
            if key not in spec:
                raise ValueError(f"Missing required key '{key}' in {path}")

        self.inputs: list[str] = spec["inputs"]
        # Features are a list of (name, expr) tuples in evaluation order.
        self.features: list[tuple[str, str]] = _parse_features(spec.get("features"))
        self.rules: list[dict] = spec["rules"]

        _validate_spec(self.inputs, self.features, self.rules, path)

        # Generate and compile a single Python function from the YAML spec.
        # This runs at the same speed as hand-written inline Python.
        self._evaluate_fn = _codegen_evaluate(self.inputs, self.features, self.rules, path)

    def evaluate(self, **kwargs) -> dict | None:
        """Evaluate features, then return the first matching rule's config.

        Args:
            **kwargs: Input values (must cover every name in ``inputs``).

        Returns:
            A config dict (shallow copy) from the first matched rule,
            or ``None`` if no rule matches.

        Raises:
            ValueError: If a required input is missing.
        """
        missing = [k for k in self.inputs if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}. "
                             f"Expected: {self.inputs}")

        return self._evaluate_fn(**{k: kwargs[k] for k in self.inputs})
