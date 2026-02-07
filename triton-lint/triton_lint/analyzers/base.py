"""Base class for all analyzers."""

from abc import ABC, abstractmethod
from typing import List

from triton_lint.core.config import Config
from triton_lint.core.finding import Finding


class Analyzer(ABC):
    """Base class for all analyzers."""

    def __init__(self, config: Config):
        self.config = config
        self.findings: List[Finding] = []

    @abstractmethod
    def analyze(self, source_code: str, filename: str) -> List[Finding]:
        """
        Analyze the given source code and return findings.

        Args:
            source_code: The source code to analyze
            filename: Path to the file being analyzed

        Returns:
            List of findings discovered during analysis
        """
        pass

    def reset(self):
        """Reset the analyzer state for a new file."""
        self.findings = []
