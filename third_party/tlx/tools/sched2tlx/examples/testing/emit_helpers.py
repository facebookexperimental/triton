"""Materialize before/after copies of the sched2tlx tool for any diff and
re-emit each example's ``generated.py`` with both emitter versions.

The regression driver compares the kernel produced by the emitter *before* a
change against the one produced *after* it. To stay robust for an arbitrary
diff (one that may touch more than ``emitter.py`` — including a case's
``schedule_graph.json`` or ``handwritten.py``) we materialize two
self-contained copies of the tool tree:

  1. Copy the working-copy ``sched2tlx/`` package and ``examples/`` dirs into a
     ``before/`` and an ``after/`` temp tree.
  2. For every file the diff touched (``sl status --change <rev>``) that lives
     under the tool subtree, overwrite the ``before`` copy with its pre-diff
     content (``sl cat -r <before>``) and the ``after`` copy with its post-diff
     content (``sl cat -r <after>``). Files the diff did not touch are identical
     across revisions, so the working-copy version already present is correct.

Each side is then a coherent snapshot of the tool, and ``emit_case`` runs
``python -m sched2tlx`` against it to produce that side's ``generated.py``.

Pure stdlib — no torch/triton/GPU needed for this module.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Repo-root-relative path of the sched2tlx tool (the dir holding ``sched2tlx/``
# and ``examples/``). Kept relative so ``sl`` paths and tree copies line up.
TOOL_RELPATH = "third-party/triton/beta/triton/third_party/tlx/tools/sched2tlx"

# Only files under these tool subdirectories affect emission / running.
_RELEVANT_SUBDIRS = ("sched2tlx/", "examples/")


def _sl(repo_root: Path, args: list[str], reason: str) -> str:
    """Run a read-only ``sl`` command and return stdout (text)."""
    return subprocess.run(
        ["sl", *args, "--reason", reason],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def find_repo_root(start: Path) -> Path:
    """Walk upward until we find the fbsource checkout root."""
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / TOOL_RELPATH).is_dir():
            return parent
    raise RuntimeError(f"could not locate repo root containing {TOOL_RELPATH} from {start}")


def changed_tool_files(repo_root: Path, diff_rev: str) -> list[tuple[str, str]]:
    """Return ``(status_code, relpath)`` for diff-touched files under the tool.

    ``status_code`` is ``M``/``A``/``R`` from ``sl status --change``.
    """
    out = _sl(
        repo_root,
        ["status", "--change", diff_rev],
        "list files changed by diff for sched2tlx before/after regression - sl help status",
    )
    changed: list[tuple[str, str]] = []
    for line in out.splitlines():
        if len(line) < 3 or line[1] != " ":
            continue
        code, relpath = line[0], line[2:].strip()
        if not relpath.startswith(TOOL_RELPATH + "/"):
            continue
        suffix = relpath[len(TOOL_RELPATH) + 1:]
        if suffix.startswith(_RELEVANT_SUBDIRS):
            changed.append((code, relpath))
    return changed


def _ignore_pycache(_dir: str, names: list[str]) -> list[str]:
    return [n for n in names if n == "__pycache__"]


def _place_file(side_dir: Path, suffix: str, repo_root: Path, relpath: str, rev: str, present: bool) -> None:
    """Overwrite ``side_dir/suffix`` with file content at ``rev`` (or remove it)."""
    dst = side_dir / suffix
    if not present:
        if dst.exists():
            dst.unlink()
        return
    content = _sl(
        repo_root,
        ["cat", "-r", rev, relpath],
        "fetch file at revision for sched2tlx before/after regression - sl help cat",
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content)


@dataclass
class SideTree:
    """A materialized snapshot of the tool at one revision."""

    root: Path  # contains ``sched2tlx/`` (package) and ``examples/``

    @property
    def examples(self) -> Path:
        return self.root / "examples"


def materialize(repo_root: Path, workdir: Path, before_rev: str, after_rev: str) -> tuple[SideTree, SideTree]:
    """Build coherent before/after copies of the tool tree under ``workdir``."""
    tool = repo_root / TOOL_RELPATH
    before = SideTree(workdir / "before")
    after = SideTree(workdir / "after")
    for side in (before, after):
        if side.root.exists():
            shutil.rmtree(side.root)
        side.root.mkdir(parents=True)
        shutil.copytree(tool / "sched2tlx", side.root / "sched2tlx", ignore=_ignore_pycache)
        shutil.copytree(tool / "examples", side.root / "examples", ignore=_ignore_pycache)

    for code, relpath in changed_tool_files(repo_root, after_rev):
        suffix = relpath[len(TOOL_RELPATH) + 1:]
        _place_file(before.root, suffix, repo_root, relpath, before_rev, present=(code != "A"))
        _place_file(after.root, suffix, repo_root, relpath, after_rev, present=(code != "R"))

    return before, after


def emit_case(side: SideTree, case_name: str) -> Path:
    """Run ``python -m sched2tlx`` for one case on this side; return the .py path.

    Writes ``generated.py`` into the side's case dir, overwriting the copied
    placeholder. Raises CalledProcessError if emission fails. Subprocess form —
    needs a plain interpreter with the ``sched2tlx`` package importable. For
    buck/par execution use :func:`emit_case_inproc`.
    """
    case_dir = side.examples / case_name
    graph = case_dir / "schedule_graph.json"
    out = case_dir / "generated.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(side.root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-m", "sched2tlx", str(graph), "-o", str(out)],
        cwd=str(side.root),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def _purge_sched2tlx_modules() -> None:
    for name in [m for m in sys.modules if m == "sched2tlx" or m.startswith("sched2tlx.")]:
        del sys.modules[name]


def emit_case_inproc(side: SideTree, case_name: str) -> Path:
    """Emit one case's ``generated.py`` in-process (works inside a buck par).

    Imports this side's ``sched2tlx`` package fresh (purging any other side's
    copy first so before/after emitters never collide) and calls ``emit``.
    """
    case_dir = side.examples / case_name
    graph = case_dir / "schedule_graph.json"
    out = case_dir / "generated.py"
    import importlib

    _purge_sched2tlx_modules()
    sys.path.insert(0, str(side.root))
    try:
        schedule_graph = importlib.import_module("sched2tlx.schedule_graph")
        emitter = importlib.import_module("sched2tlx.emitter")
        src = emitter.emit(schedule_graph.load_graph(str(graph)))
    finally:
        try:
            sys.path.remove(str(side.root))
        except ValueError:
            pass
        _purge_sched2tlx_modules()
    out.write_text(src)
    return out
