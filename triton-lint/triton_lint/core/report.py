"""Report formatting and output for tritlint findings."""

import json
import sys
from collections import defaultdict
from typing import List, TextIO

from triton_lint.core.finding import Finding, Severity


class Reporter:
    """Formats and outputs findings in various formats."""

    def __init__(self, output_format: str = "text", show_suggestions: bool = True):
        self.output_format = output_format
        self.show_suggestions = show_suggestions

    def report(self, findings: List[Finding], output: TextIO = sys.stdout) -> int:
        """
        Output findings in the configured format.

        Returns:
            Exit code (0 if no errors, 1 if errors found)
        """
        if self.output_format == "json":
            self._report_json(findings, output)
        elif self.output_format == "sarif":
            self._report_sarif(findings, output)
        else:
            self._report_text(findings, output)

        # Return exit code based on errors
        return 1 if any(f.is_error for f in findings) else 0

    def _report_text(self, findings: List[Finding], output: TextIO):
        """Report findings in human-readable text format."""
        if not findings:
            output.write("No issues found.\n")
            return

        # Group findings by file
        by_file = defaultdict(list)
        for finding in findings:
            by_file[finding.filename].append(finding)

        # Sort files alphabetically
        for filename in sorted(by_file.keys()):
            file_findings = sorted(by_file[filename], key=lambda f: (f.line, f.col))

            for finding in file_findings:
                output.write(self._format_finding_text(finding))
                output.write("\n")

        # Summary
        output.write("\n")
        self._write_summary(findings, output)

    def _format_finding_text(self, finding: Finding) -> str:
        """Format a single finding as text."""
        location = f"{finding.filename}:{finding.line}:{finding.col}"
        lines = [f"{location}: {finding.severity.value}: {finding.rule_id}"]
        lines.append(f"    {finding.message}")

        if self.show_suggestions and finding.suggestion:
            lines.append(f"    Suggestion: {finding.suggestion}")

        if finding.context:
            lines.append(f"    Context: {finding.context}")

        return "\n".join(lines)

    def _write_summary(self, findings: List[Finding], output: TextIO):
        """Write summary of findings."""
        counts = defaultdict(int)
        for finding in findings:
            counts[finding.severity] += 1

        total = len(findings)
        parts = []

        if counts[Severity.ERROR] > 0:
            parts.append(f"{counts[Severity.ERROR]} error{'s' if counts[Severity.ERROR] > 1 else ''}")
        if counts[Severity.WARNING] > 0:
            parts.append(f"{counts[Severity.WARNING]} warning{'s' if counts[Severity.WARNING] > 1 else ''}")
        if counts[Severity.INFO] > 0:
            parts.append(f"{counts[Severity.INFO]} info")
        if counts[Severity.HINT] > 0:
            parts.append(f"{counts[Severity.HINT]} hint{'s' if counts[Severity.HINT] > 1 else ''}")

        summary = ", ".join(parts) if parts else "No issues"
        output.write(f"{total} issue{'s' if total != 1 else ''} found ({summary})\n")

    def _report_json(self, findings: List[Finding], output: TextIO):
        """Report findings in JSON format."""
        data = {
            "findings": [f.to_dict() for f in findings], "summary": {
                "total": len(findings),
                "errors": sum(1 for f in findings if f.is_error),
                "warnings": sum(1 for f in findings if f.is_warning),
            }
        }
        json.dump(data, output, indent=2)
        output.write("\n")

    def _report_sarif(self, findings: List[Finding], output: TextIO):
        """
        Report findings in SARIF format.

        SARIF (Static Analysis Results Interchange Format) is a standard format
        for static analysis tools, supported by GitHub, VS Code, and other IDEs.
        """
        # Group findings by file for locations
        results = []
        for finding in findings:
            result = {
                "ruleId":
                finding.rule_id, "level":
                self._severity_to_sarif_level(finding.severity), "message": {"text": finding.message}, "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": finding.filename}, "region": {
                            "startLine": finding.line, "startColumn": finding.col + 1,  # SARIF uses 1-indexed columns
                        }
                    }
                }]
            }

            if finding.suggestion:
                result["fixes"] = [{"description": {"text": finding.suggestion}}]

            results.append(result)

        sarif = {
            "$schema":
            "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json", "version":
            "2.1.0", "runs": [{
                "tool": {
                    "driver": {
                        "name": "triton_lint", "informationUri": "https://github.com/triton-lang/triton", "version":
                        "0.1.0"
                    }
                }, "results": results
            }]
        }

        json.dump(sarif, output, indent=2)
        output.write("\n")

    @staticmethod
    def _severity_to_sarif_level(severity: Severity) -> str:
        """Convert tritlint severity to SARIF level."""
        mapping = {
            Severity.ERROR: "error",
            Severity.WARNING: "warning",
            Severity.INFO: "note",
            Severity.HINT: "note",
        }
        return mapping.get(severity, "warning")
