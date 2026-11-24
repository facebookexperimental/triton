"""Map the basename in a sanitizer report to the source path Triton compiled."""

from __future__ import annotations

import os
import re
from pathlib import Path

# Source paths in Triton's cached IR: loc("/abs/path/kernel.py":line:col).
_LOC_PATH_RE = re.compile(r'loc\("([^"]+)"')


def compiled_kernels(triton_cache: Path) -> list[str]:
    """Names of the kernels Triton compiled into ``triton_cache``."""
    return sorted({ir.stem for ir in triton_cache.glob("*/*.ttgir")})


def display_path(path: str) -> str:
    """The path below site-packages for an installed module, else ``path``."""
    _, sep, tail = path.partition("/site-packages/")
    return tail if sep else path


def resolve_paths(reports: list[dict], triton_cache: Path) -> None:
    """Set each report's ``path`` from the IR Triton cached for its kernel.

    The sanitizer prints only the basename (``sm100.py``), which many TLX
    kernels share; the kernel's cached TTGIR records the full source paths.
    """
    sources: dict[str, set[str]] = {}
    for report in reports:
        kernel, file = report["kernel"], report["file"]
        if not kernel or not file:
            continue
        if kernel not in sources:
            sources[kernel] = set()
            for ir in triton_cache.glob("*/*.ttgir"):
                if ir.stem == kernel:
                    sources[kernel].update(_LOC_PATH_RE.findall(ir.read_text(errors="replace")))
        matches = {path for path in sources[kernel] if os.path.basename(path) == file}
        if len(matches) == 1:
            report["path"] = display_path(matches.pop())
