from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable


def frozen_source_digest(source: str, editable_symbols: Iterable[str]) -> str:
    """Hash source after masking the top-level symbols an authoring phase owns."""
    editable = set(editable_symbols)
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    for node in tree.body:
        names: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target, )
            names.update(target.id for target in targets if isinstance(target, ast.Name))
        if not names.intersection(editable):
            continue
        owned = ",".join(sorted(names.intersection(editable)))
        replacements.append((node.lineno - 1, node.end_lineno or node.lineno, owned))
    for start, end, owned in reversed(replacements):
        lines[start:end] = [f"# TLX_AGENT_EDITABLE:{owned}\n"]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def validate_branch_comments(source: str, function_name: str) -> tuple[bool, str]:
    """Require a short explanatory comment immediately before each policy branch."""
    tree = ast.parse(source)
    function = next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name),
        None,
    )
    if function is None:
        return False, f"missing {function_name}()"
    lines = source.splitlines()
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        previous = node.lineno - 2
        if previous < 0 or not lines[previous].strip().startswith("#"):
            return False, f"{function_name}() branch at line {node.lineno} needs an explanatory comment"
    return True, ""
