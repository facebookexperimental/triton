"""Configuration management for triton_lint."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import toml
except ImportError:
    toml = None


@dataclass
class Config:
    """
    Configuration for tritlint analysis.

    Attributes:
        enabled_rules: Set of rule IDs to enable (None = all enabled)
        disabled_rules: Set of rule IDs to disable
        rule_severities: Override severities for specific rules
        analysis_levels: Which analysis levels to run (ast, ttir, ttgir)
        target_backend: Target hardware backend (cuda, mtia, etc.)
        max_errors: Maximum number of errors before stopping (0 = unlimited)
        show_suggestions: Whether to show fix suggestions
        output_format: Output format (text, json, sarif)
    """
    enabled_rules: Optional[Set[str]] = None
    disabled_rules: Set[str] = field(default_factory=set)
    rule_severities: Dict[str, str] = field(default_factory=dict)
    analysis_levels: List[str] = field(default_factory=lambda: ["ast"])
    target_backend: str = "cuda"
    max_errors: int = 0
    show_suggestions: bool = True
    output_format: str = "text"

    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> "Config":
        """
        Load configuration from a TOML file.

        If config_path is None, searches for .triton_lint.toml in current directory
        and parent directories.
        """
        if config_path is None:
            config_path = cls._find_config_file()

        if config_path is None or not config_path.exists():
            return cls()

        if toml is None:
            raise RuntimeError("toml package required for config file support")

        with open(config_path, "r") as f:
            data = toml.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create Config from dictionary."""
        config = cls()

        # Parse rules section
        if "rules" in data:
            rules = data["rules"]
            # Rules can be specified as rule-name = "severity"
            for rule_id, severity in rules.items():
                if severity.lower() in ["off", "false", "disabled"]:
                    config.disabled_rules.add(rule_id)
                elif severity.lower() in ["error", "warning", "info", "hint"]:
                    config.rule_severities[rule_id] = severity

        # Parse analysis section
        if "analysis" in data:
            analysis = data["analysis"]
            if "levels" in analysis:
                config.analysis_levels = analysis["levels"]
            if "max_errors" in analysis:
                config.max_errors = analysis["max_errors"]

        # Parse target section
        if "target" in data:
            target = data["target"]
            if "backend" in target:
                config.target_backend = target["backend"]

        # Parse output section
        if "output" in data:
            output = data["output"]
            if "format" in output:
                config.output_format = output["format"]
            if "show_suggestions" in output:
                config.show_suggestions = output["show_suggestions"]

        return config

    @staticmethod
    def _find_config_file() -> Optional[Path]:
        """Search for .triton_lint.toml in current and parent directories."""
        current = Path.cwd()

        while True:
            config_path = current / ".triton_lint.toml"
            if config_path.exists():
                return config_path

            # Check if we've reached the root
            parent = current.parent
            if parent == current:
                break
            current = parent

        return None

    def is_rule_enabled(self, rule_id: str) -> bool:
        """Check if a rule is enabled."""
        if rule_id in self.disabled_rules:
            return False
        if self.enabled_rules is None:
            return True
        return rule_id in self.enabled_rules

    def get_rule_severity(self, rule_id: str, default: str = "warning") -> str:
        """Get the severity for a rule, with fallback to default."""
        return self.rule_severities.get(rule_id, default)
