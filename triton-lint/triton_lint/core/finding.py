"""Data structures for representing analysis findings."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    """Severity levels for findings."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


@dataclass
class Finding:
    """
    Represents a single finding from the analysis.

    Attributes:
        rule_id: Unique identifier for the rule that triggered this finding
        severity: Severity level of the finding
        filename: Path to the file where the finding was detected
        line: Line number (1-indexed)
        col: Column number (0-indexed)
        message: Human-readable description of the issue
        suggestion: Optional suggestion for how to fix the issue
        context: Optional additional context (e.g., code snippet)
    """
    rule_id: str
    severity: Severity
    filename: str
    line: int
    col: int
    message: str
    suggestion: Optional[str] = None
    context: Optional[str] = None

    def __str__(self) -> str:
        """Format finding as human-readable string."""
        location = f"{self.filename}:{self.line}:{self.col}"
        result = f"{location}: {self.severity.value}: {self.rule_id}\n"
        result += f"    {self.message}"
        if self.suggestion:
            result += f"\n    Suggestion: {self.suggestion}"
        return result

    def to_dict(self) -> dict:
        """Convert finding to dictionary for JSON serialization."""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "filename": self.filename,
            "line": self.line,
            "col": self.col,
            "message": self.message,
            "suggestion": self.suggestion,
            "context": self.context,
        }

    @property
    def is_error(self) -> bool:
        """Check if this finding is an error."""
        return self.severity == Severity.ERROR

    @property
    def is_warning(self) -> bool:
        """Check if this finding is a warning."""
        return self.severity == Severity.WARNING
