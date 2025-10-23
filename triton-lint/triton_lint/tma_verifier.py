"""
TMA Verification using C++ MLIR Pass

This module provides a Python interface to the C++ TMA verification pass.
"""

import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class TMAOpDetail:
    """Details about a TMA operation or missed opportunity."""
    index: Optional[int]
    operation: str
    critical: bool
    transfer_size_bytes: Optional[int]
    reason: Optional[str]
    location: Optional[str]
    function: Optional[str] = None


@dataclass
class TMAVerificationResult:
    """Result of TMA verification."""

    passed: bool
    errors: List[str]
    warnings: List[str]
    tma_ops_found: int
    regular_ops_found: int
    missed_opportunities: int
    critical_misses: int
    exit_code: int

    # Additional details
    compute_capability: Optional[int] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    tma_operations: List[TMAOpDetail] = None
    missed_details: List[TMAOpDetail] = None
    info_messages: List[Dict] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.tma_operations is None:
            self.tma_operations = []
        if self.missed_details is None:
            self.missed_details = []
        if self.info_messages is None:
            self.info_messages = []

    def print_tma_locations(self):
        """Print where TMA operations were found in kernel code (unique locations only)."""
        if not self.tma_operations:
            print("No TMA operations found.")
            return

        # Group by function
        by_function = {}
        for op in self.tma_operations:
            func = op.function or "unknown"
            if func not in by_function:
                by_function[func] = []
            by_function[func].append(op)

        print("\nTMA Operations Found in Kernel Code")
        print(f"{'' * 60}")

        total_unique_locations = 0

        for func_name, ops in sorted(by_function.items()):
            print(f"\n Function: {func_name}")

            # Group by operation type
            by_type = {}
            for op in ops:
                op_type = op.operation
                if op_type not in by_type:
                    by_type[op_type] = []
                by_type[op_type].append(op)

            for op_type, op_list in sorted(by_type.items()):
                # Get unique locations only
                unique_locations = set()
                for op in op_list:
                    if op.location:
                        unique_locations.add(op.location)

                total_unique_locations += len(unique_locations)

                print(f"\n    {op_type}")
                print(f"     Total: {len(op_list)} operation(s), {len(unique_locations)} unique location(s)")

                # Print unique locations in sorted order
                for location in sorted(unique_locations):
                    print(f"     - {location}")

        print(
            f"\nTotal: {len(self.tma_operations)} TMA operations, {total_unique_locations} unique locations, {len(by_function)} function(s)"
        )


class TMAVerifier:
    """
    Verifies TMA usage using standalone C++ MLIR pass.

    This runs the Triton-Lint TMA verification pass on MLIR IR to detect
    when Triton's TMA optimization passes failed to insert TMA operations.
    """

    def __init__(self, pass_lib_path: Optional[str] = None):
        """
        Initialize TMA verifier.

        Args:
            pass_lib_path: Path to libtriton_lint_passes.so
                          If None, looks in default locations
        """
        self.pass_lib = self._find_pass_library(pass_lib_path)
        self.triton_opt = self._find_triton_opt()

    def _find_pass_library(self, provided_path: Optional[str]) -> Path:
        """Find the tritlint pass library."""
        if provided_path:
            path = Path(provided_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Pass library not found: {provided_path}")

        # Search default locations
        search_paths = [
            Path(__file__).parent.parent / "install" / "lib" / "libtriton_lint_passes.so",
            Path(__file__).parent.parent / "lib" / "build" / "libtriton_lint_passes.so",
            Path.home() / ".local" / "lib" / "libtriton_lint_passes.so",
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            "libtriton_lint_passes.so not found. Please build it first:\n  cd triton-lint && ./build.sh")

    def _find_triton_opt(self) -> Optional[str]:
        """Find triton-opt executable."""
        import shutil
        import glob

        # Try to find triton-opt in PATH first
        triton_opt = shutil.which("triton-opt")

        if triton_opt:
            return triton_opt

        # Search in Triton's build directory
        # The structure is: ../build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt
        tritlint_dir = Path(__file__).parent.parent
        triton_build_dir = tritlint_dir.parent / "build"

        if triton_build_dir.exists():
            # Look for cmake.* directories
            cmake_dirs = list(triton_build_dir.glob("cmake.*"))
            for cmake_dir in cmake_dirs:
                triton_opt_path = cmake_dir / "bin" / "triton-opt"
                if triton_opt_path.exists():
                    return str(triton_opt_path)

        # Try other common locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "build" / "bin" / "triton-opt",
            Path("/usr/local/bin/triton-opt"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        return None

    def verify(self, ir_str: str, filename: str = "kernel.mlir", verbose: bool = False,
               source_code: Optional[str] = None) -> TMAVerificationResult:
        """
        Run TMA verification on MLIR IR.

        Args:
            ir_str: MLIR IR as string
            filename: Name for temp file (for error messages)
            verbose: If True, print diagnostic output

        Returns:
            TMAVerificationResult with details
        """
        if not self.triton_opt:
            return TMAVerificationResult(
                passed=False,
                errors=["triton-opt not found. Cannot run verification."],
                warnings=[],
                tma_ops_found=0,
                regular_ops_found=0,
                missed_opportunities=0,
                critical_misses=0,
                exit_code=-1,
            )

        # Write IR to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(ir_str)
            ir_file = f.name

        try:
            # Run triton-opt with verification pass
            ## /data/users/pka/triton/build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt
            ##  --load-pass-plugin=/data/users/pka/triton/triton-lint/install/lib/libtriton_lint_passes.so
            ##  --pass-pipeline='builtin.module(triton-lint-verify-tma)' /tmp/test.mlir
            cmd = [
                self.triton_opt,
                f"--load-pass-plugin={self.pass_lib}",
                "--pass-pipeline=builtin.module(triton-lint-verify-tma)",
                ir_file,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Optionally print diagnostic output
            if verbose:
                print(f"Running: {' '.join(cmd)}")
                print("\n=== triton-opt stdout ===")
                if result.stdout:
                    print(result.stdout)
                print("\n=== triton-opt stderr (diagnostics) ===")
                if result.stderr:
                    print(result.stderr)
                print(f"\n=== Exit code: {result.returncode} ===\n")

            # Parse output
            parsed_result = self._parse_output(result, verbose=verbose)

            # If source code was provided, check if TMA DSL usage matches TMA ops found
            if source_code and parsed_result.compute_capability and parsed_result.compute_capability >= 90:
                parsed_result = self._check_dsl_to_ir_consistency(source_code, parsed_result, verbose)

            return parsed_result

        except subprocess.TimeoutExpired:
            return TMAVerificationResult(
                passed=False,
                errors=["Verification timed out after 30 seconds"],
                warnings=[],
                tma_ops_found=0,
                regular_ops_found=0,
                missed_opportunities=0,
                critical_misses=0,
                exit_code=-1,
            )
        except Exception as e:
            return TMAVerificationResult(
                passed=False,
                errors=[f"Verification failed: {e}"],
                warnings=[],
                tma_ops_found=0,
                regular_ops_found=0,
                missed_opportunities=0,
                critical_misses=0,
                exit_code=-1,
            )
        finally:
            # Clean up temp file
            Path(ir_file).unlink(missing_ok=True)

    def _parse_output(self, result: subprocess.CompletedProcess, verbose: bool = False) -> TMAVerificationResult:
        """Parse triton-opt JSON output."""
        import json
        import re

        errors = []
        warnings = []
        tma_ops = 0
        regular_ops = 0
        missed = 0
        critical_misses = 0

        # Additional details
        compute_capability = None
        skipped = False
        skip_reason = None
        tma_operations = []
        missed_details = []
        info_messages = []

        # Parse stderr for JSON remarks
        for line in result.stderr.split("\n"):
            # Extract location for context
            location = None
            loc_match = re.match(r"^([^:]+:\d+:\d+):", line)
            if loc_match:
                location = loc_match.group(1)

            # Look for JSON summary
            if "[TRITON_LINT_JSON_SUMMARY]" in line:
                try:
                    json_start = line.find("{")
                    if json_start != -1:
                        json_str = line[json_start:]
                        data = json.loads(json_str)

                        tma_ops = data.get("tma_ops_found", 0)
                        regular_ops = data.get("regular_mem_ops", 0)
                        missed = data.get("tma_eligible_ops", 0)
                        critical_misses = data.get("critical_misses", 0)

                        # Check status
                        if data.get("status") == "failed":
                            errors.append(f"TMA verification failed: {missed} TMA-eligible operations found")
                except json.JSONDecodeError as e:
                    warnings.append(f"Failed to parse JSON summary: {e}")

            # Look for JSON info messages
            elif "[TRITON_LINT_JSON_INFO]" in line:
                try:
                    json_start = line.find("{")
                    if json_start != -1:
                        json_str = line[json_start:]
                        data = json.loads(json_str)
                        info_messages.append(data)

                        # Extract specific info
                        event = data.get("event")
                        if event == "skipped":
                            skipped = True
                            skip_reason = data.get("reason")
                        elif event == "started":
                            compute_capability = data.get("compute_capability")
                        elif event == "no_cc_attr":
                            compute_capability = data.get("default_cc")
                except json.JSONDecodeError:
                    pass

            # Look for TMA operations
            elif "[TRITON_LINT_JSON_TMA_OP]" in line:
                try:
                    json_start = line.find("{")
                    if json_start != -1:
                        json_str = line[json_start:]
                        data = json.loads(json_str)

                        tma_op = TMAOpDetail(
                            index=None,
                            operation=data.get("operation", "unknown"),
                            critical=False,
                            transfer_size_bytes=None,
                            reason=None,
                            location=location,
                            function=data.get("function"),
                        )
                        tma_operations.append(tma_op)
                except json.JSONDecodeError:
                    pass

            # Look for JSON regression details
            elif "[TRITON_LINT_JSON_REGRESSION]" in line:
                try:
                    json_start = line.find("{")
                    if json_start != -1:
                        json_str = line[json_start:]
                        data = json.loads(json_str)
                        errors.append(data.get("message", "TMA regression detected"))
                except json.JSONDecodeError:
                    pass

            # Look for JSON details of missed opportunities
            elif "[TRITON_LINT_JSON_DETAIL]" in line:
                try:
                    json_start = line.find("{")
                    if json_start != -1:
                        json_str = line[json_start:]
                        data = json.loads(json_str)

                        detail = TMAOpDetail(
                            index=data.get("index"),
                            operation=data.get("operation", "unknown"),
                            critical=data.get("critical", False),
                            transfer_size_bytes=data.get("transfer_size_bytes"),
                            reason=data.get("reason"),
                            location=location,
                        )
                        missed_details.append(detail)

                        # Add to errors list
                        critical_str = "CRITICAL" if detail.critical else "WARNING"
                        errors.append(f"{critical_str}: {detail.operation} - {detail.reason}")
                except json.JSONDecodeError:
                    pass

        passed = len(errors) == 0

        return TMAVerificationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            tma_ops_found=tma_ops,
            regular_ops_found=regular_ops,
            missed_opportunities=missed,
            critical_misses=critical_misses,
            exit_code=result.returncode,
            compute_capability=compute_capability,
            skipped=skipped,
            skip_reason=skip_reason,
            tma_operations=tma_operations,
            missed_details=missed_details,
            info_messages=info_messages,
        )

    def _check_dsl_to_ir_consistency(self, source_code: str, result: TMAVerificationResult,
                                     verbose: bool) -> TMAVerificationResult:
        """
        Check if TMA DSL usage in source code matches TMA operations in IR.

        This detects cases where the user wrote code expecting TMA (e.g., using
        tl.make_tensor_ptr) but the compiler didn't generate TMA operations.
        """
        tma_apis, dsl_locations = analyze_tma_dsl_usage(source_code)

        if not tma_apis:
            # No TMA DSL usage found - this is fine
            return result

        if verbose:
            print(f"\n=== TMA DSL Analysis ===")
            print(f"Found {len(tma_apis)} TMA API(s) in source: {tma_apis}")
            print(f"Found {result.tma_ops_found} TMA operations in IR")

        # If TMA DSL was used but no TMA ops in IR, this is a regression
        if result.tma_ops_found == 0:
            error_msg = (f"TMA DSL REGRESSION: Kernel uses TMA APIs ({', '.join(sorted(tma_apis))}) "
                         f"but compiled IR contains NO TMA operations. "
                         f"This indicates the Triton compiler failed to generate TMA operations.")
            result.errors.append(error_msg)
            result.passed = False

            # Add warnings about each DSL usage location
            for line_no, api in dsl_locations:
                result.warnings.append(f"Line {line_no}: {api} did not produce TMA operations in compiled IR")

        return result

    def is_available(self) -> bool:
        """Check if TMA verification is available."""
        return self.pass_lib.exists() and self.triton_opt is not None


def analyze_tma_dsl_usage(source_code: str) -> Tuple[Set[str], List[Tuple[int, str]]]:
    """
    Analyze Python source code for TMA DSL usage.

    Returns:
        Tuple of (tma_apis_used, locations)
        - tma_apis_used: Set of TMA API calls found
        - locations: List of (line_number, code_snippet) tuples
    """
    tma_apis = set()
    locations = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set(), []

    class TMAVisitor(ast.NodeVisitor):

        def visit_Call(self, node):
            # Check for tl.make_tensor_ptr()
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "make_tensor_ptr":
                    tma_apis.add("make_tensor_ptr")
                    locations.append((node.lineno, "make_tensor_ptr"))
                elif node.func.attr == "make_tensor_descriptor":
                    tma_apis.add("make_tensor_descriptor")
                    locations.append((node.lineno, "make_tensor_descriptor"))

            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            # Check for block pointer type annotations: ptr: tl.tensor[...]
            if isinstance(node.annotation, ast.Subscript):
                if isinstance(node.annotation.value, ast.Attribute):
                    if node.annotation.value.attr == "tensor":
                        # This might be a block pointer
                        tma_apis.add("block_pointer_type")
                        locations.append((node.lineno, "block_pointer_type_annotation"))

            self.generic_visit(node)

    visitor = TMAVisitor()
    visitor.visit(tree)

    return tma_apis, locations


def verify_tma(ir_str: str, pass_lib_path: Optional[str] = None) -> TMAVerificationResult:
    """
    Convenience function to verify TMA usage.

    Args:
        ir_str: MLIR IR as string
        pass_lib_path: Optional path to libtriton_lint_passes.so

    Returns:
        TMAVerificationResult
    """
    verifier = TMAVerifier(pass_lib_path)
    return verifier.verify(ir_str)
