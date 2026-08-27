from __future__ import annotations

import ast
import hashlib
import re

_CODE_BLOCK_RE = re.compile(r"```(?P<language>[A-Za-z0-9_+-]*)\s*\n(?P<code>.*?)```", re.DOTALL)


def extract_python_source(output: str) -> str:
    blocks = list(_CODE_BLOCK_RE.finditer(output))
    python_blocks = [
        match.group("code")
        for match in blocks
        if match.group("language").lower() in {"python", "py"}
    ]
    generic_blocks = [
        match.group("code") for match in blocks if not match.group("language")
    ]
    candidates = python_blocks or generic_blocks or [output]
    source = canonicalize_source(candidates[-1])
    validate_kernel_source(source)
    return source


def validate_kernel_source(source: str) -> None:
    """Raise ValueError if source is empty or not valid Python."""
    if not source.strip():
        raise ValueError("candidate source is empty")
    try:
        ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"candidate is not valid Python: {error}") from error


def canonicalize_source(source: str) -> str:
    return source.strip() + "\n"


def source_digest(source: str) -> str:
    return hashlib.sha256(canonicalize_source(source).encode()).hexdigest()
