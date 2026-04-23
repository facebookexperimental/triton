#!/usr/bin/env python3
"""
Simplified triton-lint runner without external dependencies.
Works directly on Meta's production system.

Usage:
    python3 triton_lint_simple.py examples/bad_kernel.py
    python3 triton_lint_simple.py --help
    python3 triton_lint_simple.py --verify-tma kernel.py  # TMA regression detection
"""

import sys
import os
from pathlib import Path

# Add tritlint to path
sys.path.insert(0, str(Path(__file__).parent))

from triton_lint.analyzers.ast_analyzer import ASTAnalyzer
from triton_lint.analyzers.spill_analyzer import SpillAnalyzer
from triton_lint.core.config import Config
from triton_lint.core.report import Reporter
import importlib.util

# ============================================================================
# Common Kernel Compilation Utilities
# ============================================================================


def load_module(file_path):
    """
    Load a Python file as a module.

    Args:
        file_path: Path to the Python file

    Returns:
        Loaded module or None on error
    """
    import_path = str(Path(file_path).parent.absolute())
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module from {file_path}", file=sys.stderr)
        return None

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing module: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def run_test_functions(module):
    """
    Find and run test_* functions in a module to compile kernels.

    Args:
        module: The loaded module

    Returns:
        List of test function names that were run
    """
    test_functions = []
    for name in dir(module):
        if name.startswith("test_") and callable(getattr(module, name)):
            test_functions.append(name)

    if test_functions:
        print(f"Found {len(test_functions)} test function(s) - running to compile kernels...")
        for test_name in test_functions:
            try:
                print(f"  Running {test_name}...", end=" ")
                test_fn = getattr(module, test_name)
                test_fn()
            except Exception as e:
                print(f" Error: {e}")
        print()

    return test_functions


def find_triton_kernels(module):
    """
    Find @triton.jit decorated kernels in a module.

    Args:
        module: The loaded module

    Returns:
        List of (kernel_name, kernel_function) tuples
    """
    kernels = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        # A JITFunction has run, src, and other attributes
        if callable(obj) and hasattr(obj, "run") and hasattr(obj, "src"):
            kernels.append((name, obj))

    return kernels


def get_kernel_cache(kernel_fn):
    """
    Extract compilation cache from a Triton kernel.

    Args:
        kernel_fn: The kernel function

    Returns:
        Cache dictionary or None if not found
    """
    cache_dict = None

    if hasattr(kernel_fn, "cache") and kernel_fn.cache:
        cache_dict = kernel_fn.cache
    elif hasattr(kernel_fn, "device_caches") and kernel_fn.device_caches:
        # device_caches is a dict of (device -> (cache_dict, target, backend, binder))
        for device_cache in kernel_fn.device_caches.values():
            if device_cache and len(device_cache) > 0 and isinstance(device_cache[0], dict):
                cache_dict = device_cache[0]
                break

    return cache_dict


def get_compiled_asm(cache_dict, asm_type):
    """
    Get compiled assembly from cache.

    Args:
        cache_dict: The kernel cache dictionary
        asm_type: Type of assembly to extract ('ttgir', 'ptx', etc.)

    Returns:
        Tuple of (success, asm_string, error_message)
    """
    # Get first cached compilation
    compiled = None
    for key, comp in cache_dict.items():
        if hasattr(comp, "asm"):
            compiled = comp
            break

    if compiled is None:
        return (False, None, "No compiled version with asm in cache")

    # Check if requested ASM type is available
    if asm_type not in compiled.asm:
        available = list(compiled.asm.keys()) if hasattr(compiled, 'asm') else 'none'
        return (False, None, f"{asm_type.upper()} not found. Available: {available}")

    asm_code = str(compiled.asm[asm_type])
    return (True, asm_code, None)


# ============================================================================
# Help and Main Functions
# ============================================================================


def print_help():
    """Print help message."""
    print("""
Tritlint - Static analyzer for Triton kernels

Usage:
    python3 triton_lint_simple.py <file.py|file.ptx> [options]

Options:
    --help              Show this help message
    --format=<fmt>      Output format: text, json, sarif (default: text)
    --no-suggestions    Don't show fix suggestions
    --verify-tma        Enable TMA regression detection (requires C++ pass)
    --analyze-spills    Analyze PTX file for register spills
    --target=<arch>     Target architecture: cuda:90 (Hopper), etc.

Analysis Modes:
    Default:          Fast AST-based linting (no compilation)
    --verify-tma:     Compile and verify TMA usage (requires C++ pass built)
    --analyze-spills: Analyze register spills in PTX files

Examples:
    # Basic source linting (fast)
    python3 triton_lint_simple.py examples/bad_kernel.py

    # JSON output
    python3 triton_lint_simple.py examples/bad_kernel.py --format=json

    # TMA regression detection (compiles kernel)
    python3 triton_lint_simple.py --verify-tma examples/good_tma_kernel.py

    # Analyze register spills (compiles kernel)
    python3 triton_lint_simple.py --analyze-spills kernel.py
    """)


def compile_and_analyze_spills(source_code, file_path, target="cuda:90"):
    """Compile kernel to PTX and analyze for register spills."""
    try:
        import triton
        from triton.compiler import compile as triton_compile
    except ImportError:
        print("Error: Triton not available for spill analysis", file=sys.stderr)
        print("Install Triton or use basic linting", file=sys.stderr)
        return False

    print(f"\n{'=' * 80}")
    print("Register spill analysis")
    print(f"{'=' * 80}\n")
    print(f"File: {file_path}")
    print(f"Target: {target}")
    print()

    # Load module
    module = load_module(file_path)
    if module is None:
        return False

    # Run test functions to compile kernels
    run_test_functions(module)

    # Find kernels
    kernels = find_triton_kernels(module)
    if not kernels:
        print(" No @triton.jit kernels found", file=sys.stderr)
        print("\nMake sure your kernels are decorated with @triton.jit", file=sys.stderr)
        return False

    print(f"Found {len(kernels)} kernel(s): {', '.join(k[0] for k in kernels)}\n")

    all_passed = True
    from triton_lint.core.finding import Severity

    for kernel_name, kernel_fn in kernels:
        print(f"Analyzing: {kernel_name}")

        try:
            # Get kernel cache
            cache_dict = get_kernel_cache(kernel_fn)
            if not cache_dict:
                print(" Kernel not compiled")
                print("    Add test_ functions to compile kernels automatically")
                print("    Or run the kernel before verification")
                all_passed = False
                continue

            # Extract PTX from cache
            print("  Extracting PTX from cache...", end=" ")
            success, ptx_code, error = get_compiled_asm(cache_dict, 'ptx')
            if not success:
                print(f" {error}")
                all_passed = False
                continue
            print("")

            # Analyze spills
            print("  Analyzing for spills...", end=" ")
            config = Config()
            analyzer = SpillAnalyzer(config)
            findings = analyzer.analyze(ptx_code, f"{kernel_name}.ptx")
            print("")

            if not findings:
                print("\n   No register spills detected! Kernel is well-optimized.\n")
                continue

            # Group findings by rule_id
            overall_findings = [f for f in findings if f.rule_id == "SPILL001"]
            location_findings = [f for f in findings if f.rule_id == "SPILL002"]

            # Print overall summary
            if overall_findings:
                print()
                for finding in overall_findings:
                    severity_symbol = {
                        Severity.ERROR: " ERROR",
                        Severity.WARNING: " WARNING",
                        Severity.INFO: " INFO",
                    }.get(finding.severity, "")

                    print(f"  {severity_symbol}: {finding.message}")
                    if finding.suggestion:
                        print(f"    {finding.suggestion}")
                    if finding.context:
                        print(f"    {finding.context}")

            # Print hotspots
            if location_findings:
                print(f"\n  HOTSPOTS (locations with most spills):")
                print(f"  {'' * 76}")

                for idx, finding in enumerate(location_findings[:3], 1):  # Show top 3
                    print(f"\n  {idx}. {finding.filename}:{finding.line}:{finding.col}")
                    print(f"     {finding.message}")
                    if finding.context:
                        print(f"     {finding.context}")
                    if finding.suggestion:
                        print("\n     Suggestions:")
                        for line in finding.suggestion.split("\n"):
                            if line.strip():
                                print(f"     {line}")

                if len(location_findings) > 3:
                    print(f"\n  ... and {len(location_findings) - 3} more location(s)")

            # Check if there are errors
            has_errors = any(f.severity == Severity.ERROR for f in findings)
            if has_errors:
                all_passed = False

        except Exception as e:
            print(f" Error: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

        print()

    if all_passed:
        print("All kernels passed spill analysis (no errors)")
    else:
        print("Some kernels have register spill issues")

    return all_passed


def compile_and_verify_tma(source_code, file_path, target="cuda:90"):
    """Compile kernel and verify TMA usage."""
    try:
        import triton
        from triton.compiler import compile as triton_compile
    except ImportError:
        print("Error: Triton not available for TMA verification", file=sys.stderr)
        print("Install Triton or use basic linting (without --verify-tma)", file=sys.stderr)
        return False

    # Try to import TMA verifier
    try:
        from triton_lint.tma_verifier import TMAVerifier
    except ImportError:
        print("Error: TMA verifier not available", file=sys.stderr)
        print("Build the C++ pass first: cd triton-lint && ./build.sh", file=sys.stderr)
        return False

    # Check if verifier is available
    verifier = TMAVerifier()
    if not verifier.is_available():
        print("Error: TMA verifier not fully available", file=sys.stderr)
        print("Ensure:", file=sys.stderr)
        print("  1. C++ pass is built: ./build.sh", file=sys.stderr)
        print("  2. triton-opt is in PATH or Triton is built", file=sys.stderr)
        return False

    print(f"\n{'=' * 60}")
    print(f"TMA Analysis")
    print(f"{'=' * 60}\n")
    print(f"File: {file_path}")
    print(f"Target: {target}")
    print()

    # Load module
    module = load_module(file_path)
    if module is None:
        return False

    # Run test functions to compile kernels
    run_test_functions(module)

    # Find kernels
    kernels = find_triton_kernels(module)
    if not kernels:
        print(" No @triton.jit kernels found", file=sys.stderr)
        print("\nMake sure your kernels are decorated with @triton.jit", file=sys.stderr)
        return False

    print(f"Found {len(kernels)} kernel(s): {', '.join(k[0] for k in kernels)}\n")

    all_passed = True

    for kernel_name, kernel_fn in kernels:
        print(f"Verifying: {kernel_name}")
        print(f"{'' * 60}")

        try:
            # Get kernel cache
            cache_dict = get_kernel_cache(kernel_fn)
            if not cache_dict:
                print(" Kernel not compiled")
                print("    Add test_ functions to compile kernels automatically")
                print("    Or run the kernel before verification")
                all_passed = False
                continue

            # Extract TTGIR from cache
            print("  Extracting IR from cache...", end=" ")
            success, mlir_ir, error = get_compiled_asm(cache_dict, 'ttgir')
            if not success:
                print(f" {error}")
                all_passed = False
                continue
            print("")

            # Verify TMA (pass source code for DSL consistency check)
            print("  Verifying TMA...", end=" ")
            result = verifier.verify(mlir_ir, f"{kernel_name}.mlir", source_code=source_code)

            if result.passed:
                print("")
                print(f"    TMA ops found: {result.tma_ops_found}")

                # Show where TMA operations were found
                if result.tma_operations:
                    result.print_tma_locations()
            else:
                print("")
                all_passed = False
                print(f"    TMA ops: {result.tma_ops_found}")
                print(f"    Regular ops: {result.regular_ops_found}")
                print(f"    Missed: {result.missed_opportunities}")

                if result.errors:
                    print(f"    Errors:")
                    for err in result.errors[:3]:
                        print(f"      - {err[:100]}")
                    if len(result.errors) > 3:
                        print(f"      ... and {len(result.errors) - 3} more")

                # Show TMA locations even on failure
                if result.tma_operations:
                    result.print_tma_locations()

        except Exception as e:
            print(f" Error: {e}")
            all_passed = False

        print()

    if all_passed:
        print("All kernels passed TMA verification")
    else:
        print("Some kernels failed TMA verification")
    print()

    return all_passed


def main():
    """Main entry point."""
    # Parse arguments
    if len(sys.argv) < 2 or "--help" in sys.argv:
        print_help()
        sys.exit(0)

    # Find the file to analyze
    file_path = None
    output_format = "text"
    show_suggestions = True
    verify_tma = False
    analyze_spills = False
    target = "cuda:90"

    for arg in sys.argv[1:]:
        if arg.startswith("--format="):
            output_format = arg.split("=")[1]
        elif arg == "--no-suggestions":
            show_suggestions = False
        elif arg == "--verify-tma":
            verify_tma = True
        elif arg == "--analyze-spills":
            analyze_spills = True
        elif arg.startswith("--target="):
            target = arg.split("=")[1]
        elif not arg.startswith("--"):
            file_path = arg

    if not file_path:
        print("Error: No file specified", file=sys.stderr)
        print_help()
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    # Read the file
    try:
        with open(file_path, "r") as f:
            source_code = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Spill analysis mode (compiles kernel to PTX and analyzes)
    if analyze_spills:
        compile_and_analyze_spills(source_code, file_path, target)

    # TMA verification mode (compiles kernel)
    if verify_tma:
        compile_and_verify_tma(source_code, file_path, target)

    print(f"\n{'=' * 60}")
    print("Fast AST linting (no compilation)")
    print(f"{'=' * 60}\n")

    # Create config and analyzer
    config = Config()
    config.output_format = output_format
    config.show_suggestions = show_suggestions

    # Analyze
    analyzer = ASTAnalyzer(config)
    findings = analyzer.analyze(source_code, file_path)

    # Report
    reporter = Reporter(output_format=config.output_format, show_suggestions=config.show_suggestions)
    exit_code = reporter.report(findings)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
