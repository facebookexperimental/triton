# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Main Spill Analyzer integrating all components.

Analyzes PTX files for register spills and maps them back to Python source.
"""

from pathlib import Path
from typing import List, Optional

from triton_lint.analyzers.base import Analyzer
from triton_lint.core.config import Config
from triton_lint.core.finding import Finding, Severity

from .aggregator import SpillAggregator
from .debug_mapper import DebugInfoMapper
from .ptx_parser import PTXParser


class SpillAnalyzer(Analyzer):
    """
    Post-compilation analyzer for register spills.

    Analyzes compiled PTX to identify register spills and traces them
    back to the original Python source code.

    Example usage:
        analyzer = SpillAnalyzer(config)
        findings = analyzer.analyze_ptx_file(Path("kernel.ptx"))
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the analyzer with optional configuration."""
        if config is None:
            config = Config()
        super().__init__(config)
        self.parser = PTXParser()
        self.mapper = DebugInfoMapper()
        self.aggregator = SpillAggregator()

    def analyze(self, source_code: str, filename: str) -> List[Finding]:
        """
        Analyze source code (implements base class method).

        For SpillAnalyzer, this expects PTX assembly as source_code.

        Args:
            source_code: PTX assembly code
            filename: Name of the PTX file (for reporting)

        Returns:
            List of findings
        """
        self.reset()
        return self._analyze_ptx_content(source_code, filename)

    def analyze_ptx_file(self, ptx_file: Path, llvm_ir_file: Optional[Path] = None) -> List[Finding]:
        """
        Analyze a PTX file for register spills.

        Args:
            ptx_file: Path to PTX file
            llvm_ir_file: Optional LLVM IR file for enhanced mapping

        Returns:
            List of findings
        """
        self.reset()

        if not ptx_file.exists():
            return [
                Finding(
                    rule_id="SPILL000",
                    severity=Severity.ERROR,
                    filename=str(ptx_file),
                    line=0,
                    col=0,
                    message=f"PTX file not found: {ptx_file}",
                )
            ]

        ptx_content = ptx_file.read_text()
        return self._analyze_ptx_content(ptx_content, str(ptx_file), llvm_ir_file)

    def _analyze_ptx_content(
        self,
        ptx_content: str,
        ptx_filename: str,
        llvm_ir_file: Optional[Path] = None,
    ) -> List[Finding]:
        """
        Internal method to analyze PTX content.

        Args:
            ptx_content: PTX assembly code
            ptx_filename: Filename for reporting
            llvm_ir_file: Optional LLVM IR file

        Returns:
            List of findings
        """
        # Step 1: Parse PTX
        ptx_info = self.parser.parse(ptx_content)

        # No spills or local memory? Great!
        if not ptx_info.spills and ptx_info.total_local_bytes == 0:
            return []

        # Step 2: Map spills to source
        mapped_spills = self.mapper.map_spills_to_source(ptx_info, llvm_ir_file)

        # Step 3: Aggregate and analyze
        report = self.aggregator.aggregate(mapped_spills)

        # Step 4: Generate findings
        findings = self._generate_findings(report, ptx_info, ptx_filename)

        self.findings.extend(findings)
        return findings

    def _generate_findings(self, report, ptx_info, ptx_filename: str) -> List[Finding]:
        """
        Convert spill report to Finding objects.

        Args:
            report: SpillReport with analysis
            ptx_info: PTXSpillInfo with raw data
            ptx_filename: PTX file name for reporting

        Returns:
            List of findings
        """
        findings = []

        # Overall summary finding
        if ptx_info.total_local_bytes > 0:
            severity = self._compute_severity(ptx_info.total_local_bytes)
            findings.append(
                Finding(
                    rule_id="SPILL001",
                    severity=severity,
                    filename=ptx_filename,
                    line=0,
                    col=0,
                    message=f"Register spills detected: {ptx_info.total_local_bytes} bytes local memory",
                    suggestion=f"Local memory is ~100x slower than registers. "
                    f"Detected {report.total_spills} spill instructions "
                    f"({report.stores} stores, {report.loads} loads)",
                    context=f"Pattern: {report.patterns['pattern']} - {report.patterns['reason']}",
                ))

        # Per-location findings for hotspots
        for loc, spill_list in report.hotspots:
            summary = self.aggregator.summarize_spills(spill_list)
            suggestions = self._suggest_fixes(report.patterns, len(spill_list))

            findings.append(
                Finding(
                    rule_id="SPILL002",
                    severity=Severity.WARNING,
                    filename=loc.file,
                    line=loc.line,
                    col=loc.column,
                    message=f"{len(spill_list)} register spills at this location",
                    suggestion=suggestions,
                    context=f"Spill details: {summary}",
                ))

        return findings

    def _compute_severity(self, total_bytes: int) -> Severity:
        """
        Compute severity based on spill size.

        Args:
            total_bytes: Total bytes of local memory

        Returns:
            Severity level
        """
        if total_bytes < 256:
            return Severity.INFO
        elif total_bytes < 1024:
            return Severity.WARNING
        else:
            return Severity.ERROR

    def _suggest_fixes(self, patterns: dict, spill_count: int) -> str:
        """
        Generate actionable fix suggestions.

        Args:
            patterns: Detected spill patterns
            spill_count: Number of spills at location

        Returns:
            Suggestion string
        """
        suggestions = []

        pattern = patterns.get("pattern", "unknown")

        if pattern == "store_heavy":
            suggestions.append("• Reduce tile sizes (BLOCK_M, BLOCK_N, BLOCK_K)")
            suggestions.append("• Recompute values instead of storing intermediate results")

        if pattern == "load_heavy":
            suggestions.append("• Consider restructuring to use more registers")
            suggestions.append("• Check if values can be computed instead of loaded")

        if spill_count > 10:
            suggestions.append("• Reduce num_stages to decrease register pressure from pipelining")
            suggestions.append("• Break kernel into smaller functions")

        if not suggestions:
            suggestions.append("• Reduce register pressure by simplifying computations")
            suggestions.append("• Consider using smaller data types if precision allows")

        return "\n".join(suggestions)
