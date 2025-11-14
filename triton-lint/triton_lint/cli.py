"""Command-line interface for triton_lint."""

import sys
from pathlib import Path
from typing import List, Optional

import click

from tritlint import __version__
from triton_lint.analyzers.ast_analyzer import ASTAnalyzer
from triton_lint.analyzers.ttir_analyzer import TTIRAnalyzer
from triton_lint.analyzers.mlir_analyzer import MLIRAnalyzer
from triton_lint.core.config import Config
from triton_lint.core.finding import Finding
from triton_lint.core.report import Reporter
from triton_lint.rules import list_rules


@click.command()
@click.version_option(version=__version__)
@click.argument("files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to config file (.triton_lint.toml)",
)
@click.option(
    "--format",
    type=click.Choice(["text", "json", "sarif"]),
    default="text",
    help="Output format",
)
@click.option(
    "--target",
    type=click.Choice(["cuda", "mtia", "rocm"]),
    default="cuda",
    help="Target hardware backend",
)
@click.option(
    "--enable-rule",
    multiple=True,
    help="Enable specific rule (can be used multiple times)",
)
@click.option(
    "--disable-rule",
    multiple=True,
    help="Disable specific rule (can be used multiple times)",
)
@click.option(
    "--level",
    type=click.Choice(["ast", "ttir", "ttgir"]),
    multiple=True,
    default=["ast"],
    help="Analysis levels to run (can be used multiple times)",
)
@click.option(
    "--list-rules",
    is_flag=True,
    help="List all available rules and exit",
)
@click.option(
    "--no-suggestions",
    is_flag=True,
    help="Don't show fix suggestions in output",
)
@click.option(
    "--max-errors",
    type=int,
    default=0,
    help="Maximum number of errors before stopping (0 = unlimited)",
)
def main(
    files: tuple,
    config: Optional[str],
    format: str,
    target: str,
    enable_rule: tuple,
    disable_rule: tuple,
    level: tuple,
    list_rules_flag: bool,
    no_suggestions: bool,
    max_errors: int,
):
    """
    Tritlint - Static analyzer for Triton kernel performance.

    Analyzes Triton kernels for performance anti-patterns and correctness issues.

    Examples:

        # Analyze a single file
        tritlint kernel.py

        # Analyze multiple files
        tritlint kernel1.py kernel2.py

        # Use specific rules
        tritlint --enable-rule=scalar-memory-access kernel.py

        # Output as JSON
        tritlint --format=json kernel.py > report.json
    """
    if list_rules_flag:
        _print_rules()
        sys.exit(0)

    # Load configuration
    if config:
        cfg = Config.from_file(Path(config))
    else:
        cfg = Config.from_file()

    # Override config with CLI options
    if format:
        cfg.output_format = format
    if target:
        cfg.target_backend = target
    if level:
        cfg.analysis_levels = list(level)
    if enable_rule:
        cfg.enabled_rules = set(enable_rule)
    if disable_rule:
        cfg.disabled_rules = set(disable_rule)
    if no_suggestions:
        cfg.show_suggestions = False
    if max_errors:
        cfg.max_errors = max_errors

    # Collect all findings
    all_findings: List[Finding] = []
    error_count = 0

    for file_path in files:
        path = Path(file_path)

        if not path.suffix == ".py":
            click.echo(f"Warning: Skipping non-Python file: {file_path}", err=True)
            continue

        try:
            with open(path, "r") as f:
                source_code = f.read()
        except Exception as e:
            click.echo(f"Error reading {file_path}: {e}", err=True)
            continue

        # Run analyzers based on configured levels
        findings = _analyze_file(source_code, str(path), cfg)
        all_findings.extend(findings)

        # Count errors
        error_count += sum(1 for f in findings if f.is_error)

        # Check if we've hit max errors
        if cfg.max_errors > 0 and error_count >= cfg.max_errors:
            click.echo(f"\nStopped after {error_count} errors (max_errors={cfg.max_errors})", err=True)
            break

    # Report findings
    reporter = Reporter(output_format=cfg.output_format, show_suggestions=cfg.show_suggestions)
    exit_code = reporter.report(all_findings)
    sys.exit(exit_code)


def _analyze_file(source_code: str, filename: str, config: Config) -> List[Finding]:
    """Analyze a single file with all configured analyzers."""
    findings: List[Finding] = []

    # Run AST analyzer if enabled
    if "ast" in config.analysis_levels:
        ast_analyzer = ASTAnalyzer(config)
        findings.extend(ast_analyzer.analyze(source_code, filename))

    # Run TTIR analyzer if enabled
    if "ttir" in config.analysis_levels:
        ttir_analyzer = TTIRAnalyzer(config)
        findings.extend(ttir_analyzer.analyze(source_code, filename))

    # Run MLIR/TTGIR analyzer if enabled
    if "ttgir" in config.analysis_levels:
        mlir_analyzer = MLIRAnalyzer(config)
        findings.extend(mlir_analyzer.analyze(source_code, filename))

    return findings


def _print_rules():
    """Print all available rules."""
    rules = list_rules()

    click.echo("Available Rules:\n")

    # Group by category
    by_category = {}
    for rule_id, rule_info in rules.items():
        category = rule_info["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((rule_id, rule_info))

    # Print by category
    for category, category_rules in sorted(by_category.items()):
        click.echo(f"{category.upper()}:")
        for rule_id, rule_info in sorted(category_rules):
            severity = rule_info["severity"]
            description = rule_info["description"]
            click.echo(f"  {rule_id:<30} [{severity:>7}]  {description}")
        click.echo()


if __name__ == "__main__":
    main()
