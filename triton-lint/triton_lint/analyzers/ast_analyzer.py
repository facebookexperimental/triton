"""Python AST-based analyzer for Triton kernels."""

import ast
from typing import Any, List, Optional, Set

from triton_lint.analyzers.base import Analyzer
from triton_lint.core.config import Config
from triton_lint.core.finding import Finding, Severity


class ASTAnalyzer(Analyzer):
    """
    Analyzes Triton kernel source code at the Python AST level.

    This is the fastest analysis stage and catches high-level anti-patterns
    that are visible in the source code without requiring compilation.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.filename = ""
        self.source_lines: List[str] = []
        self.current_function: Optional[ast.FunctionDef] = None
        self.triton_kernels: Set[str] = set()

    def analyze(self, source_code: str, filename: str) -> List[Finding]:
        """Analyze Python source code for Triton anti-patterns."""
        self.reset()
        self.filename = filename
        self.source_lines = source_code.splitlines()

        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError as e:
            self.findings.append(
                Finding(
                    rule_id="syntax-error",
                    severity=Severity.ERROR,
                    filename=filename,
                    line=e.lineno or 1,
                    col=e.offset or 0,
                    message=f"Syntax error: {e.msg}",
                ))
            return self.findings

        # First pass: identify all Triton kernels
        self._identify_kernels(tree)

        # Second pass: analyze each kernel
        self.visit(tree)

        return self.findings

    def _identify_kernels(self, tree: ast.AST):
        """Identify all functions decorated with @triton.jit."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_triton_kernel(node):
                    self.triton_kernels.add(node.name)

    def visit(self, node: ast.AST):
        """Visit AST nodes using the visitor pattern."""
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST):
        """Visit children of this node."""
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze function definitions."""
        if not self._is_triton_kernel(node):
            self.generic_visit(node)
            return

        old_function = self.current_function
        self.current_function = node

        # Check for autotuning
        self._check_autotuning(node)

        # Check for hardcoded block sizes
        self._check_hardcoded_block_sizes(node)

        # Analyze function body
        self.generic_visit(node)

        self.current_function = old_function

    def visit_Call(self, node: ast.Call):
        """Analyze function calls."""
        # Check for tl.load/tl.store calls
        if self._is_triton_op(node, ["load", "store"]):
            self._check_memory_operation(node)

        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        """Analyze subscript operations (e.g., ptr[i])."""
        # Check for scalar memory accesses
        if self.current_function and self._is_pointer_subscript(node):
            self._check_scalar_access(node)

        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """Analyze for loops."""
        # Check loop nesting depth
        self._check_loop_depth(node)

        self.generic_visit(node)

    # Helper methods for checking specific patterns

    def _is_triton_kernel(self, node: ast.FunctionDef) -> bool:
        """Check if function has @triton.jit decorator."""
        for decorator in node.decorator_list:
            if self._matches_triton_decorator(decorator, "jit"):
                return True
        return False

    def _has_autotune_decorator(self, node: ast.FunctionDef) -> bool:
        """Check if function has @triton.autotune decorator."""
        for decorator in node.decorator_list:
            if self._matches_triton_decorator(decorator, "autotune"):
                return True
        return False

    def _matches_triton_decorator(self, decorator: Any, name: str) -> bool:
        """Check if decorator matches triton.{name}."""
        if isinstance(decorator, ast.Call):
            func = decorator.func
        else:
            func = decorator

        if isinstance(func, ast.Attribute):
            if func.attr == name:
                if isinstance(func.value, ast.Name) and func.value.id == "triton":
                    return True
        elif isinstance(func, ast.Name):
            # Handle case where decorator is just imported: @jit
            if func.id == name:
                return True

        return False

    def _is_triton_op(self, node: ast.Call, op_names: List[str]) -> bool:
        """Check if call is a triton.language operation."""
        if not isinstance(node.func, ast.Attribute):
            return False

        if node.func.attr not in op_names:
            return False

        # Check if it's tl.* or triton.language.*
        if isinstance(node.func.value, ast.Name):
            return node.func.value.id in ["tl", "triton_language"]

        return False

    def _is_pointer_subscript(self, node: ast.Subscript) -> bool:
        """Check if this looks like a pointer subscript (e.g., ptr[i])."""
        # Heuristic: if it's a Name being subscripted, likely a pointer
        return isinstance(node.value, ast.Name)

    def _check_autotuning(self, node: ast.FunctionDef):
        """Check if kernel should have autotuning."""
        if not self._has_autotune_decorator(node):
            # Check if kernel has tunable parameters (BLOCK_*, num_warps, etc.)
            has_tunable = self._has_tunable_parameters(node)

            if has_tunable and self.config.is_rule_enabled("missing-autotune"):
                self.findings.append(
                    Finding(
                        rule_id="missing-autotune",
                        severity=Severity.WARNING,
                        filename=self.filename,
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"Kernel '{node.name}' lacks @triton.autotune decorator",
                        suggestion="Add @triton.autotune([triton.Config(...)]) to tune performance parameters",
                    ))

    def _has_tunable_parameters(self, node: ast.FunctionDef) -> bool:
        """Check if function has parameters that should be tuned."""
        # Look for BLOCK_* parameters or common tunable names
        tunable_patterns = ["BLOCK_", "NUM_WARPS", "NUM_STAGES"]

        for param in node.args.args:
            param_name = param.arg.upper()
            if any(pattern in param_name for pattern in tunable_patterns):
                return True

        return False

    def _check_hardcoded_block_sizes(self, node: ast.FunctionDef):
        """Check for hardcoded block size constants."""
        if not self.config.is_rule_enabled("hardcoded-block-size"):
            return

        # Look for assignments like BLOCK_M = 128
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        name = target.id.upper()
                        if "BLOCK" in name and isinstance(stmt.value, ast.Constant):
                            if not self._has_autotune_decorator(node):
                                self.findings.append(
                                    Finding(
                                        rule_id="hardcoded-block-size",
                                        severity=Severity.WARNING,
                                        filename=self.filename,
                                        line=stmt.lineno,
                                        col=stmt.col_offset,
                                        message=f"Hardcoded block size: {target.id} = {stmt.value.value}",
                                        suggestion="Use tl.constexpr and autotune this parameter",
                                    ))

    def _check_memory_operation(self, node: ast.Call):
        """Check tl.load/tl.store operations."""
        if not self.config.is_rule_enabled("missing-mask"):
            return

        # Check for missing mask parameter
        has_mask = any(kw.arg == "mask" for kw in node.keywords)

        if not has_mask:
            # Heuristic: likely needs mask if we're in a loop or near boundary
            if self._likely_needs_mask():
                op_name = node.func.attr  # type: ignore
                self.findings.append(
                    Finding(
                        rule_id="missing-mask",
                        severity=Severity.WARNING,
                        filename=self.filename,
                        line=node.lineno,
                        col=node.col_offset,
                        message=f"tl.{op_name}() may need masking for out-of-bounds safety",
                        suggestion=f"Add mask= parameter: tl.{op_name}(..., mask=your_mask)",
                    ))

    def _likely_needs_mask(self) -> bool:
        """Heuristic to determine if masking is likely needed."""
        # For now, always suggest masks as best practice
        # TODO: Improve heuristic based on surrounding code
        return True

    def _check_scalar_access(self, node: ast.Subscript):
        """Check for scalar pointer accesses."""
        if not self.config.is_rule_enabled("scalar-memory-access"):
            return

        # Check if this subscript is inside a tl.load/tl.store call
        # This is a simplified check - a full implementation would track parent nodes
        self.findings.append(
            Finding(
                rule_id="scalar-memory-access",
                severity=Severity.ERROR,
                filename=self.filename,
                line=node.lineno,
                col=node.col_offset,
                message="Potential scalar memory access detected",
                suggestion="Use block pointers with tl.make_block_ptr() or range-based indexing",
            ))

    def _check_loop_depth(self, node: ast.For):
        """Check for deeply nested loops."""
        if not self.config.is_rule_enabled("complex-loop-structure"):
            return

        depth = self._get_loop_depth(node)
        if depth > 3:  # More than 3 nested loops
            self.findings.append(
                Finding(
                    rule_id="complex-loop-structure",
                    severity=Severity.WARNING,
                    filename=self.filename,
                    line=node.lineno,
                    col=node.col_offset,
                    message=f"Deeply nested loop (depth {depth}) detected",
                    suggestion="Consider restructuring to reduce nesting or use tl.ravel",
                ))

    def _get_loop_depth(self, node: ast.For, current_depth: int = 1) -> int:
        """Calculate maximum loop nesting depth."""
        max_depth = current_depth

        for child in ast.walk(node):
            if isinstance(child, ast.For) and child != node:
                child_depth = self._get_loop_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)

        return max_depth
