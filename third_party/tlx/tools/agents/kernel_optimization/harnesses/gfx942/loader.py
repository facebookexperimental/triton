"""Candidate module loading, shared by the harness and its two child processes.

A candidate is written to a temp dir but may still use relative imports -- the
gfx942 mm op does (`from ._shapes import ...`). Loading it under a bare name
gives it no package, and the relative import fails. Setting
``TLX_CANDIDATE_PACKAGE`` in the target's ``environment`` names the package the
candidate belongs to; the module is then registered under it, so relative
imports resolve against the real installed package regardless of where the file
sits.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load_candidate(path: Path, suffix: str = "candidate") -> ModuleType:
    package = os.environ.get("TLX_CANDIDATE_PACKAGE", "").strip()
    if package:
        importlib.import_module(package)  # parent must be live for `from .x import y`
        name = f"{package}.{suffix}"
    else:
        name = f"tlx_agent_{suffix}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
