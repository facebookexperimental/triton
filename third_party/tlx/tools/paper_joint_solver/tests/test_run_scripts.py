import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PKG_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPTS = tuple(
    PKG_ROOT / name
    for name in (
        "run_main_cases.sh",
        "run_ablations.sh",
        "run_emitter_cases.sh",
    )
)


def _extract_marked(path, begin, end):
    source = path.read_text()
    start = source.index(begin)
    finish = source.index(end, start) + len(end)
    return source[start:finish] + "\n"


def _write_command(directory, name, body):
    path = directory / name
    path.write_text(f"#!/bin/bash\nset -eu\n{body}\n")
    path.chmod(0o755)
    return path


def _run_vcs_block(tmp_path, *, git_body, sl_body=None, extra_env=None):
    command_dir = tmp_path / "bin"
    command_dir.mkdir()
    _write_command(command_dir, "git", git_body)
    if sl_body is not None:
        _write_command(command_dir, "sl", sl_body)
    block = _extract_marked(
        RUN_SCRIPTS[0],
        "# BEGIN SOURCE VCS DETECTION",
        "# END SOURCE VCS DETECTION",
    )
    script = block + (
        "printf '%s\\n%s\\n%s\\n' "
        '"$SOURCE_VCS" "$SOURCE_REVISION" "$SOURCE_DIRTY"\n'
    )
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["PATH"] = str(command_dir)
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        ["/bin/bash", "-c", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_source_vcs_blocks_are_byte_identical():
    blocks = [
        _extract_marked(
            path,
            "# BEGIN SOURCE VCS DETECTION",
            "# END SOURCE VCS DETECTION",
        )
        for path in RUN_SCRIPTS
    ]
    assert blocks[1:] == blocks[:-1]
    assert "git rev-parse --git-dir" in blocks[0]
    assert "sl root --reason" in blocks[0]
    assert "sl help root" in blocks[0]
    assert "sl log --reason" in blocks[0]
    assert "sl help log" in blocks[0]
    assert "sl status --reason" in blocks[0]
    assert "sl help status" in blocks[0]


def test_source_vcs_detection_prefers_git_over_sl(tmp_path):
    sl_called = tmp_path / "sl-called"
    result = _run_vcs_block(
        tmp_path,
        git_body="""
if [[ "$1" == rev-parse && "$2" == --git-dir ]]; then
  printf '.git\\n'
elif [[ "$1" == rev-parse && "$2" == HEAD ]]; then
  printf 'abc123\\n'
elif [[ "$1" == status ]]; then
  printf ' M tracked-file\\n'
else
  exit 1
fi
""",
        sl_body='printf called >"$SL_CALLED"; exit 99',
        extra_env={"SL_CALLED": str(sl_called)},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "git\nabc123\nyes\n"
    assert not sl_called.exists()


def test_source_vcs_detection_rejects_non_sapling_sl(tmp_path):
    result = _run_vcs_block(
        tmp_path,
        git_body="exit 1",
        sl_body="""
if [[ "$1" == root ]]; then
  printf '/not/a/real/sapling/root\\n'
  exit 0
fi
exit 99
""",
    )

    assert result.returncode != 0
    assert "not a git worktree and not a Sapling checkout" in result.stderr


def test_source_vcs_detection_accepts_sapling(tmp_path):
    sapling_root = tmp_path / "sapling-root"
    sapling_root.mkdir()
    result = _run_vcs_block(
        tmp_path,
        git_body="exit 1",
        sl_body="""
case "$1" in
  root) printf '%s\\n' "$FAKE_SAPLING_ROOT" ;;
  log) printf 'def456\\n' ;;
  status) printf 'M tracked-file\\n' ;;
  *) exit 1 ;;
esac
""",
        extra_env={"FAKE_SAPLING_ROOT": str(sapling_root)},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "sapling\ndef456\nyes\n"


def _run_host_policy(path, *, explicit, out, default_out):
    block = _extract_marked(
        path,
        "# BEGIN HOST PINNING POLICY",
        "# END HOST PINNING POLICY",
    )
    script = """
set -euo pipefail
PYTHON="$TEST_PYTHON"
UNPINNED_HOST=1
OUT_WAS_EXPLICIT="$TEST_OUT_WAS_EXPLICIT"
OUT="$TEST_OUT"
DEFAULT_OUT="$TEST_DEFAULT_OUT"
""" + block + 'printf "%s\\n" "$PAPER_COMPARABLE"\n'
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        {
            "TEST_PYTHON": sys.executable,
            "TEST_OUT_WAS_EXPLICIT": explicit,
            "TEST_OUT": str(out),
            "TEST_DEFAULT_OUT": str(default_out),
        }
    )
    return subprocess.run(
        ["/bin/bash", "-c", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize("script_path", RUN_SCRIPTS)
def test_unpinned_host_requires_explicit_nondefault_out(tmp_path, script_path):
    default_out = tmp_path / "canonical"
    missing = _run_host_policy(
        script_path,
        explicit="",
        out=tmp_path / "development",
        default_out=default_out,
    )
    assert missing.returncode != 0
    assert "requires an explicit non-default OUT" in missing.stderr

    canonical = _run_host_policy(
        script_path,
        explicit="x",
        out=default_out / ".",
        default_out=default_out,
    )
    assert canonical.returncode != 0
    assert "cannot write the canonical OUT" in canonical.stderr

    development = _run_host_policy(
        script_path,
        explicit="x",
        out=tmp_path / "development",
        default_out=default_out,
    )
    assert development.returncode == 0, development.stderr
    assert development.stdout == "no\n"


@pytest.mark.parametrize(
    ("attempts", "expected"),
    (
        ([{"ii": 2, "L": 5, "stage": "joint", "result": "sat"}], "yes"),
        (
            [
                {"ii": 2, "L": 5, "stage": "joint", "result": "unsat"},
                {"ii": 2, "L": 6, "stage": "joint", "result": "sat"},
            ],
            "yes",
        ),
        (
            [
                {"ii": 2, "L": 5, "stage": "joint", "result": "unsat"},
                {"ii": 2, "L": 6, "stage": "joint", "result": "unsat"},
                {"ii": 3, "L": 5, "stage": "joint", "result": "sat"},
            ],
            "no",
        ),
    ),
)
def test_joint_premise_uses_any_sat_at_minimum_ii(tmp_path, attempts, expected):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"attempts": attempts}))
    predicate = _extract_marked(
        PKG_ROOT / "run_ablations.sh",
        "# BEGIN MINIMUM-II JOINT PREMISE",
        "# END MINIMUM-II JOINT PREMISE",
    )

    completed = subprocess.run(
        [sys.executable, "-", str(result_path)],
        input=predicate,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected
