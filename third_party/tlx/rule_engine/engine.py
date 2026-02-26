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

from .expr import eval_expr


class RuleEngine:
    """Declarative, first-match rule evaluator driven by a YAML file."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        with open(path) as f:
            spec = yaml.safe_load(f)

        self.inputs: list[str] = spec["inputs"]
        self.features: dict[str, str] = spec.get("features", {})
        self.rules: list[dict] = spec["rules"]

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
