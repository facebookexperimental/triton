"""RuleEngine: load a rules YAML file, evaluate features, match rules.

The rules file is an ordered list of rules, each with a ``when`` clause
(list of boolean expressions, implicitly ANDed) and a ``config`` dict.
The first rule whose ``when`` clause is satisfied wins.

Typical usage::

    engine = RuleEngine("blackwell_gemm_ws_rules.yaml")
    config = engine.evaluate(M=4096, N=4096, K=4096, num_sms=148)
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .expr import eval_expr, validate_expr


class RuleEngine:
    """Declarative, first-match rule evaluator driven by a YAML file."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        with open(path) as f:
            spec = yaml.safe_load(f)

        for key in ("inputs", "rules"):
            if key not in spec:
                raise ValueError(f"Missing required key '{key}' in {path}")

        self.inputs: list[str] = spec["inputs"]
        # Features are evaluated in insertion order (Python 3.7+ / PyYAML
        # guarantee), so later features can reference earlier ones.
        self.features: dict[str, str] = spec.get("features", {})
        self.rules: list[dict] = spec["rules"]

        self._validate(path)

    def _validate(self, path: Path) -> None:
        """Validate all expressions at load time."""
        # Names accumulate as features are defined top-to-bottom.
        known_names = set(self.inputs)

        for name, expr in self.features.items():
            validate_expr(expr, known_names, context=f"{path}: feature '{name}'")
            known_names.add(name)

        for rule in self.rules:
            rule_name = rule.get("name", "<unnamed>")
            for i, cond in enumerate(rule["when"]):
                validate_expr(cond, known_names, context=f"{path}: rule '{rule_name}' when[{i}]")

            # Validate $variable references in config values.
            for key, val in rule.get("config", {}).items():
                if isinstance(val, str) and val.startswith("$"):
                    ref = val[1:]
                    if ref not in known_names:
                        raise ValueError(f"{path}: rule '{rule_name}' config.{key}: "
                                         f"unknown variable reference '${ref}'. "
                                         f"Valid names: {sorted(known_names)}")

    def evaluate(self, **kwargs) -> dict | None:
        """Evaluate features, then return the first matching rule's config.

        Args:
            **kwargs: Input values (must cover every name in ``inputs``).

        Returns:
            A config dict (shallow copy) from the first matched rule,
            or ``None`` if no rule matches.
        """
        variables = {k: kwargs[k] for k in self.inputs}

        # Compute derived features top-to-bottom.
        for name, expr in self.features.items():
            variables[name] = eval_expr(expr, variables)

        # First-match rule evaluation.
        for rule in self.rules:
            conditions = rule["when"]
            if all(eval_expr(cond, variables) for cond in conditions):
                config = dict(rule["config"])
                # Resolve "$var_name" references in config values.
                for key, val in config.items():
                    if isinstance(val, str) and val.startswith("$"):
                        config[key] = variables[val[1:]]
                return config

        return None
