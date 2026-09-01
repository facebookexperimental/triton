from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

_CODE_BLOCK_RE = re.compile(r"```(?P<language>[A-Za-z0-9_+-]*)\s*\n(?P<code>.*?)```", re.DOTALL)
_DIFF_HEADER_RE = re.compile(r"(?m)^--- candidate\.py\n\+\+\+ candidate\.py\n")


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
    valid_sources: list[str] = []
    last_error: ValueError | None = None
    for candidate in candidates:
        source = canonicalize_source(candidate)
        try:
            validate_kernel_source(source)
        except ValueError as error:
            last_error = error
            continue
        valid_sources.append(source)
    if valid_sources:
        return max(valid_sources, key=len)
    assert last_error is not None
    raise last_error


def apply_candidate_diff(output: str, current_source: str) -> str:
    """Apply a model-produced unified diff to an isolated source copy."""
    blocks = list(_CODE_BLOCK_RE.finditer(output))
    diff_blocks = [
        match.group("code")
        for match in blocks
        if match.group("language").lower() in {"diff", "patch"}
    ]
    candidates = diff_blocks or [output]
    patches = [candidate.strip() + "\n" for candidate in candidates if _DIFF_HEADER_RE.search(candidate)]
    if not patches:
        raise ValueError("candidate response does not contain a candidate.py unified diff")
    patch = max(patches, key=len)
    with tempfile.TemporaryDirectory(prefix="tlx-agent-patch-") as directory:
        root = Path(directory)
        candidate_path = root / "candidate.py"
        patch_path = root / "candidate.patch"
        candidate_path.write_text(current_source)
        patch_path.write_text(patch)
        completed = subprocess.run(
            ["patch", "--batch", "--forward", "--silent", "-p0", "-i", str(patch_path)],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            diagnostics = (completed.stderr or completed.stdout).strip()
            raise ValueError(f"candidate diff did not apply cleanly: {diagnostics}")
        source = canonicalize_source(candidate_path.read_text())
    validate_replacement_source(source, current_source)
    return source


def validate_kernel_source(source: str) -> None:
    """Raise ValueError if source is empty or not valid Python."""
    if not source.strip():
        raise ValueError("candidate source is empty")
    try:
        ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"candidate is not valid Python: {error}") from error


def validate_replacement_source(candidate: str, current: str) -> None:
    """Reject responses that are valid Python but not a complete replacement."""
    validate_kernel_source(candidate)
    current_tree = ast.parse(current)
    candidate_tree = ast.parse(candidate)
    node_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    required_names = {node.name for node in current_tree.body if isinstance(node, node_types)}
    candidate_names = {
        node.name for node in candidate_tree.body if isinstance(node, node_types)
    }
    missing_names = sorted(required_names - candidate_names)
    if missing_names:
        preview = ", ".join(missing_names[:8])
        suffix = "..." if len(missing_names) > 8 else ""
        raise ValueError(
            f"candidate is not a complete replacement; missing top-level symbols: "
            f"{preview}{suffix}"
        )
    if len(candidate) < len(current) * 0.8:
        raise ValueError(
            "candidate is not a complete replacement; source is unexpectedly short "
            f"({len(candidate)} bytes versus {len(current)} bytes)"
        )


def canonicalize_source(source: str) -> str:
    return source.strip() + "\n"


def source_digest(source: str) -> str:
    return hashlib.sha256(canonicalize_source(source).encode()).hexdigest()
