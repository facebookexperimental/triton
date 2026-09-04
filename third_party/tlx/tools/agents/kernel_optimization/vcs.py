from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal

from .models import AutoCommitResult

VcsKind = Literal["git", "hg"]
ATTRIBUTION = "TLX agent authored"


class AutoCommitError(RuntimeError):
    pass


@dataclass(frozen=True)
class AutoCommitSnapshot:
    vcs: VcsKind
    repo_root: Path
    target_path: Path
    target_relpath: str
    base_revision: str
    parent_source: str
    baseline_source: str
    dirty_target_at_start: bool
    file_mode: str | None = None
    index_state: str | None = None


@dataclass
class AutoCommitSession:
    initial_snapshot: AutoCommitSnapshot
    snapshot: AutoCommitSnapshot
    promotion_commits: list[AutoCommitResult] = field(default_factory=list)
    rollback_commit: AutoCommitResult | None = None

    @classmethod
    def create(cls, snapshot: AutoCommitSnapshot) -> AutoCommitSession:
        return cls(initial_snapshot=snapshot, snapshot=snapshot)


def _run(
    command: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    environment: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    if check and completed.returncode != 0:
        diagnostics = (completed.stderr or completed.stdout).strip()
        raise AutoCommitError(f"{' '.join(command[:2])} failed: {diagnostics}")
    return completed


def _discover_root(target: Path, vcs: VcsKind) -> Path | None:
    if vcs == "git":
        completed = _run(
            ["git", "-C", str(target.parent), "rev-parse", "--show-toplevel"],
            cwd=target.parent,
            check=False,
        )
    else:
        completed = _run(
            ["hg", "--cwd", str(target.parent), "root"],
            cwd=target.parent,
            check=False,
        )
    if completed.returncode != 0:
        return None
    root = Path(completed.stdout.strip()).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return root


def detect_vcs(target_path: Path, requested: str = "auto") -> tuple[VcsKind, Path]:
    target = target_path.resolve()
    kinds: tuple[VcsKind, ...] = ("git", "hg")
    if requested in kinds:
        root = _discover_root(target, requested)  # type: ignore[arg-type]
        if root is None:
            raise AutoCommitError(f"target is not inside the requested {requested} repository")
        return requested, root  # type: ignore[return-value]
    if requested != "auto":
        raise AutoCommitError(f"unsupported VCS selection: {requested}")
    candidates = [
        (kind, root)
        for kind in kinds
        if (root := _discover_root(target, kind)) is not None
    ]
    if not candidates:
        raise AutoCommitError("no Git or Mercurial repository contains the target kernel")
    candidates.sort(key=lambda item: len(item[1].parts), reverse=True)
    if len(candidates) > 1 and len(candidates[0][1].parts) == len(candidates[1][1].parts):
        raise AutoCommitError("ambiguous VCS repository; pass --vcs git or --vcs hg")
    return candidates[0]


def prepare_auto_commit(
    target_path: Path, baseline_source: str, requested_vcs: str = "auto"
) -> AutoCommitSnapshot:
    target = target_path.resolve()
    if not target.is_file():
        raise AutoCommitError(f"target kernel does not exist: {target}")
    if target.read_text() != baseline_source:
        raise AutoCommitError("target changed while preparing the optimization baseline")
    vcs, root = detect_vcs(target, requested_vcs)
    relpath = target.relative_to(root).as_posix()
    if vcs == "git":
        revision = _run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=root
        ).stdout.strip()
        unmerged = _run(
            ["git", "ls-files", "-u", "--", relpath], cwd=root
        ).stdout.strip()
        if unmerged:
            raise AutoCommitError("target kernel has unresolved Git conflicts")
        tree_entry = _run(
            ["git", "ls-tree", revision, "--", relpath], cwd=root
        ).stdout.strip()
        if not tree_entry:
            raise AutoCommitError("target kernel is not tracked in Git HEAD")
        mode = tree_entry.split(maxsplit=1)[0]
        parent_source = _run(
            ["git", "show", f"{revision}:{relpath}"], cwd=root
        ).stdout
        index_state = _run(
            ["git", "ls-files", "-s", "--", relpath], cwd=root
        ).stdout
    else:
        revision = _run(
            ["hg", "log", "-r", ".", "--template", "{node}"], cwd=root
        ).stdout.strip()
        files = _run(
            ["hg", "files", "--rev", ".", relpath], cwd=root, check=False
        )
        if files.returncode != 0 or not files.stdout.strip():
            raise AutoCommitError("target kernel is not tracked in the Mercurial parent")
        unresolved = _run(
            ["hg", "resolve", "--list", relpath], cwd=root, check=False
        ).stdout.splitlines()
        if any(line.startswith("U ") for line in unresolved):
            raise AutoCommitError("target kernel has unresolved Mercurial conflicts")
        parent_source = _run(
            ["hg", "cat", "-r", ".", relpath], cwd=root
        ).stdout
        mode = None
        index_state = None
    return AutoCommitSnapshot(
        vcs=vcs,
        repo_root=root,
        target_path=target,
        target_relpath=relpath,
        base_revision=revision,
        parent_source=parent_source,
        baseline_source=baseline_source,
        dirty_target_at_start=parent_source != baseline_source,
        file_mode=mode,
        index_state=index_state,
    )


def merge_winner_delta(snapshot: AutoCommitSnapshot, winner_source: str) -> str:
    with tempfile.TemporaryDirectory(prefix="tlx-agent-merge-") as directory:
        root = Path(directory)
        parent_path = root / "parent.py"
        baseline_path = root / "baseline.py"
        winner_path = root / "winner.py"
        parent_path.write_text(snapshot.parent_source)
        baseline_path.write_text(snapshot.baseline_source)
        winner_path.write_text(winner_source)
        completed = _run(
            [
                "git",
                "merge-file",
                "--stdout",
                str(parent_path),
                str(baseline_path),
                str(winner_path),
            ],
            cwd=root,
            check=False,
        )
    if completed.returncode == 1:
        raise AutoCommitError("winner overlaps pre-existing target edits")
    if completed.returncode != 0:
        diagnostics = (completed.stderr or completed.stdout).strip()
        raise AutoCommitError(f"three-way merge failed: {diagnostics}")
    if completed.stdout == snapshot.parent_source:
        raise AutoCommitError("winner has no commit-worthy delta")
    return completed.stdout


def _commit_message(
    subject: str, body: str = "", *, include_attribution: bool = True
) -> str:
    cleaned_body = "\n".join(
        line for line in body.replace("\x00", "").splitlines() if line.strip() != ATTRIBUTION
    ).strip()
    paragraphs = [subject.rstrip()]
    if cleaned_body:
        paragraphs.append(cleaned_body)
    if include_attribution:
        paragraphs.append(ATTRIBUTION)
    return "\n\n".join(paragraphs) + "\n"


def _verify_snapshot(snapshot: AutoCommitSnapshot) -> None:
    if snapshot.target_path.read_text() != snapshot.baseline_source:
        raise AutoCommitError("target kernel changed during optimization")
    if snapshot.vcs == "git":
        revision = _run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=snapshot.repo_root
        ).stdout.strip()
        index_state = _run(
            ["git", "ls-files", "-s", "--", snapshot.target_relpath],
            cwd=snapshot.repo_root,
        ).stdout
        if index_state != snapshot.index_state:
            raise AutoCommitError("target kernel index state changed during optimization")
    else:
        revision = _run(
            ["hg", "log", "-r", ".", "--template", "{node}"],
            cwd=snapshot.repo_root,
        ).stdout.strip()
    if revision != snapshot.base_revision:
        raise AutoCommitError("repository parent changed during optimization")


def _commit_git(
    snapshot: AutoCommitSnapshot,
    committed_source: str,
    winner_source: str,
    subject: str,
    body: str = "",
    *,
    include_attribution: bool = True,
) -> str:
    assert snapshot.file_mode is not None
    root = snapshot.repo_root
    blob = _run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=root,
        input_text=committed_source,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="tlx-agent-index-") as directory:
        index_path = Path(directory) / "index"
        environment = os.environ.copy()
        environment["GIT_INDEX_FILE"] = str(index_path)
        _run(["git", "read-tree", snapshot.base_revision], cwd=root, environment=environment)
        _run(
            [
                "git",
                "update-index",
                "--add",
                "--cacheinfo",
                f"{snapshot.file_mode},{blob},{snapshot.target_relpath}",
            ],
            cwd=root,
            environment=environment,
        )
        tree = _run(
            ["git", "write-tree"], cwd=root, environment=environment
        ).stdout.strip()
    commit = _run(
        ["git", "commit-tree", tree, "-p", snapshot.base_revision],
        cwd=root,
        input_text=_commit_message(
            subject, body, include_attribution=include_attribution
        ),
    ).stdout.strip()
    _run(
        [
            "git",
            "update-ref",
            "-m",
            "tlx-agent auto-commit winner",
            "HEAD",
            commit,
            snapshot.base_revision,
        ],
        cwd=root,
    )
    _run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            f"{snapshot.file_mode},{blob},{snapshot.target_relpath}",
        ],
        cwd=root,
    )
    snapshot.target_path.write_text(winner_source)
    committed = _run(
        ["git", "show", f"HEAD:{snapshot.target_relpath}"], cwd=root
    ).stdout
    if committed != committed_source:
        raise AutoCommitError("Git commit postcondition failed for target source")
    return commit


def _commit_hg(
    snapshot: AutoCommitSnapshot,
    committed_source: str,
    winner_source: str,
    subject: str,
    body: str = "",
    *,
    include_attribution: bool = True,
) -> str:
    root = snapshot.repo_root
    snapshot.target_path.write_text(committed_source)
    succeeded = False
    try:
        _run(
            [
                "hg",
                "commit",
                "-m",
                _commit_message(
                    subject, body, include_attribution=include_attribution
                ),
                snapshot.target_relpath,
            ],
            cwd=root,
        )
        succeeded = True
    finally:
        snapshot.target_path.write_text(
            winner_source if succeeded else snapshot.baseline_source
        )
    revision = _run(
        ["hg", "log", "-r", ".", "--template", "{node}"], cwd=root
    ).stdout.strip()
    committed = _run(
        ["hg", "cat", "-r", ".", snapshot.target_relpath], cwd=root
    ).stdout
    if committed != committed_source:
        raise AutoCommitError("Mercurial commit postcondition failed for target source")
    return revision


def commit_winner(
    snapshot: AutoCommitSnapshot,
    winner_source: str,
    subject: str,
    body: str = "",
    *,
    validate_committed_source: Callable[[str], None] | None = None,
) -> AutoCommitResult:
    _verify_snapshot(snapshot)
    committed_source = merge_winner_delta(snapshot, winner_source)
    if committed_source != winner_source:
        if validate_committed_source is None:
            raise AutoCommitError(
                "dirty-target winner requires validation of the merged commit source"
            )
        validate_committed_source(committed_source)
    _verify_snapshot(snapshot)
    if snapshot.vcs == "git":
        revision = _commit_git(snapshot, committed_source, winner_source, subject, body)
    else:
        revision = _commit_hg(snapshot, committed_source, winner_source, subject, body)
    return AutoCommitResult(
        requested=True,
        success=True,
        vcs=snapshot.vcs,
        repo_root=snapshot.repo_root,
        target_path=snapshot.target_path,
        target_relpath=snapshot.target_relpath,
        base_revision=snapshot.base_revision,
        commit_revision=revision,
        subject=subject,
        dirty_target_at_start=snapshot.dirty_target_at_start,
    )


def _advance_snapshot(
    snapshot: AutoCommitSnapshot,
    *,
    revision: str,
    parent_source: str,
    baseline_source: str,
) -> AutoCommitSnapshot:
    index_state = snapshot.index_state
    if snapshot.vcs == "git":
        index_state = _run(
            ["git", "ls-files", "-s", "--", snapshot.target_relpath],
            cwd=snapshot.repo_root,
        ).stdout
    return replace(
        snapshot,
        base_revision=revision,
        parent_source=parent_source,
        baseline_source=baseline_source,
        index_state=index_state,
    )


def commit_promotion(
    session: AutoCommitSession,
    candidate_source: str,
    subject: str,
    body: str = "",
    *,
    validate_committed_source: Callable[[str], None] | None = None,
) -> AutoCommitResult:
    snapshot = session.snapshot
    result = commit_winner(
        snapshot,
        candidate_source,
        subject,
        body,
        validate_committed_source=validate_committed_source,
    )
    assert result.commit_revision is not None
    committed_source = (
        _run(
            ["git", "show", f"{result.commit_revision}:{snapshot.target_relpath}"],
            cwd=snapshot.repo_root,
        ).stdout
        if snapshot.vcs == "git"
        else _run(
            ["hg", "cat", "-r", result.commit_revision, snapshot.target_relpath],
            cwd=snapshot.repo_root,
        ).stdout
    )
    session.snapshot = _advance_snapshot(
        snapshot,
        revision=result.commit_revision,
        parent_source=committed_source,
        baseline_source=candidate_source,
    )
    session.promotion_commits.append(result)
    return result


def commit_rollback(
    session: AutoCommitSession,
    subject: str,
    body: str = "",
) -> AutoCommitResult:
    if not session.promotion_commits:
        raise AutoCommitError("no promotion commit is available to roll back")
    current = session.snapshot
    initial = session.initial_snapshot
    rollback_source = initial.baseline_source
    committed_source = initial.parent_source
    _verify_snapshot(current)
    if current.vcs == "git":
        revision = _commit_git(
            current,
            committed_source,
            rollback_source,
            subject,
            body,
            include_attribution=False,
        )
    else:
        revision = _commit_hg(
            current,
            committed_source,
            rollback_source,
            subject,
            body,
            include_attribution=False,
        )
    result = AutoCommitResult(
        requested=True,
        success=True,
        vcs=current.vcs,
        repo_root=current.repo_root,
        target_path=current.target_path,
        target_relpath=current.target_relpath,
        base_revision=current.base_revision,
        commit_revision=revision,
        subject=subject,
        attribution="",
        dirty_target_at_start=initial.dirty_target_at_start,
    )
    session.snapshot = _advance_snapshot(
        current,
        revision=revision,
        parent_source=committed_source,
        baseline_source=rollback_source,
    )
    session.rollback_commit = result
    return result


def failed_auto_commit(
    snapshot: AutoCommitSnapshot | None, subject: str, error: Exception
) -> AutoCommitResult:
    return AutoCommitResult(
        requested=True,
        success=False,
        vcs=snapshot.vcs if snapshot else None,
        repo_root=snapshot.repo_root if snapshot else None,
        target_path=snapshot.target_path if snapshot else None,
        target_relpath=snapshot.target_relpath if snapshot else None,
        base_revision=snapshot.base_revision if snapshot else None,
        subject=subject,
        dirty_target_at_start=snapshot.dirty_target_at_start if snapshot else False,
        diagnostics=f"{type(error).__name__}: {error}",
    )
