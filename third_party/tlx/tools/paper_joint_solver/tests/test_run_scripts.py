import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PKG_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PKG_ROOT / "run_main_cases.sh"
ABLATION_SCRIPT = PKG_ROOT / "run_ablations.sh"
RETIRED_SCRIPT = PKG_ROOT / "run_emitter_cases.sh"
# The emitter batch is retired, so the shared infrastructure blocks live in
# the two surviving canonical scripts only.
RUN_SCRIPTS = (MAIN_SCRIPT, ABLATION_SCRIPT)


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
PAPER_HOST=dgx003
PAPER_CPU_MODEL=8570
MACHINE_HOSTNAME=elsewhere
LSCPU_MODEL="a different cpu"
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


def test_host_pinning_blocks_are_byte_identical_and_parameterized():
    blocks = [
        _extract_marked(
            path,
            "# BEGIN HOST PINNING POLICY",
            "# END HOST PINNING POLICY",
        )
        for path in RUN_SCRIPTS
    ]
    assert blocks[1:] == blocks[:-1]
    # The pinned host and CPU model are inputs, not literals in the policy.
    assert "$PAPER_HOST" in blocks[0]
    assert "$PAPER_CPU_MODEL" in blocks[0]
    assert "dgx003" not in blocks[0]
    assert "8570" not in blocks[0]
    for path in RUN_SCRIPTS:
        source = path.read_text()
        assert "PAPER_HOST=${PAPER_HOST:-dgx003}" in source
        assert "PAPER_CPU_MODEL=${PAPER_CPU_MODEL:-8570}" in source


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
        ABLATION_SCRIPT,
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


def test_watchdog_blocks_are_byte_identical():
    blocks = [
        _extract_marked(
            path,
            "# BEGIN WATCHDOG RC CONTRACT",
            "# END WATCHDOG RC CONTRACT",
        )
        for path in RUN_SCRIPTS
    ]
    assert blocks[1:] == blocks[:-1]
    assert "run_solver_watchdogged()" in blocks[0]
    assert "timeout --kill-after=60" in blocks[0]
    # No branch may translate an exit code into a solver verdict.
    assert "sat" not in blocks[0].replace("sat/unsat", "")


def _run_watchdog(tmp_path, exit_code):
    block = _extract_marked(
        MAIN_SCRIPT,
        "# BEGIN WATCHDOG RC CONTRACT",
        "# END WATCHDOG RC CONTRACT",
    )
    obs_log = tmp_path / "observations.log"
    obs_log.write_text("")
    preamble = """
SOLVER_CPU=0
WATCHDOG_S=30
OBS_LOG="$TEST_OBS_LOG"
write_command() {
  local destination=$1
  shift
  printf '%q ' "$@" >"$destination"
}
"""
    epilogue = """
rc=0
outcome=$(run_solver_watchdogged probe-label "$TEST_CMD_LOG" "$TEST_OUT_LOG" \
  /bin/bash -c "exit $TEST_EXIT_CODE") || rc=$?
printf 'outcome=%s rc=%s\\n' "$outcome" "$rc"
"""
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.update(
        {
            "TEST_OBS_LOG": str(obs_log),
            "TEST_CMD_LOG": str(tmp_path / "cmd.log"),
            "TEST_OUT_LOG": str(tmp_path / "out.log"),
            "TEST_EXIT_CODE": str(exit_code),
        }
    )
    completed = subprocess.run(
        ["/bin/bash", "-c", preamble + block + epilogue],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    return completed, obs_log.read_text()


@pytest.mark.parametrize(
    ("exit_code", "outcome", "observation"),
    (
        (0, "completed", "outcome=completed"),
        (124, "did-not-terminate", "outcome=did-not-terminate-within-30s"),
        (137, "did-not-terminate", "outcome=did-not-terminate-within-30s"),
    ),
)
def test_watchdog_rc_contract(tmp_path, exit_code, outcome, observation):
    completed, observations = _run_watchdog(tmp_path, exit_code)

    assert completed.returncode == 0, completed.stderr
    assert f"outcome={outcome} rc=0" in completed.stdout
    assert observation in observations
    assert "label=probe-label" in observations


@pytest.mark.parametrize("exit_code", (2, 3))
def test_watchdog_treats_any_other_rc_as_infrastructure_failure(
    tmp_path, exit_code
):
    """rc=2 is argparse usage, not UNSAT: it must abort, never report."""
    completed, observations = _run_watchdog(tmp_path, exit_code)

    assert f"outcome= rc={exit_code}" not in completed.stdout
    assert "outcome= rc=1" in completed.stdout
    assert f"outcome=infrastructure-error rc={exit_code}" in observations


BANNED_FLAGS = (
    "--ilp-seconds",
    "--smt-seconds",
    "--max-wall-s",
    "--max-ii-span",
    "--no-structural-precheck",
    "--no-cross-warp-domain",
    "--paper-pure",
    "--no-spill-smem-footprint",
    "--no-reg-carried-same-lane",
    "--ignore-fixed-overheads",
    "--emitter-compatible-lane-masks",
    "--probe-first-window",
    "--profile",
    "--minimize-groups",
    "--max-groups",
    "--warp-fixed-overhead",
    "--fa4-refit-timeout-s",
    "--timeout-s",
    "paper-literal",
)


@pytest.mark.parametrize("script_path", RUN_SCRIPTS)
def test_canonical_scripts_have_no_banned_flags(script_path):
    source = script_path.read_text()
    for flag in BANNED_FLAGS:
        assert flag not in source, f"{script_path.name} still passes {flag}"


def test_emitter_batch_is_a_retired_stub():
    source = RETIRED_SCRIPT.read_text()
    assert "RETIRED" in source
    assert "exit 1" in source
    # It must not solve anything any more.
    assert "-m paper_joint_solver" not in source
    assert "taskset" not in source


def test_main_validation_has_no_shape_gates():
    source = MAIN_SCRIPT.read_text()
    for gate in (
        "no retry headroom",
        "did not charge accumulator TMEM",
        "has no TMEM storage objects",
        "failed fa4_optimal_set_member",
        "has no SAT FA4 template refit",
        "refit point does not match free optimum",
        "is not a passing report",
        "normalization_u_effective",
        "model_options",
        "model_profile",
        "PAPER_MODEL_PROFILE",
        "paper-literal",
    ):
        assert gate not in source, f"run_main_cases.sh still gates on {gate}"
    assert "OBSERVATION" in source
    assert "provenance.model" in source
    # Curation is a stage of the canonical run, and the chain is asserted.
    assert "curate_ddg" in source
    assert "curation_source" in source
    assert "paper-joint-strategy-v2" in source
    assert "paper-joint-pipelined-ir-v3" in source


def test_main_script_drops_the_unrenderable_figure():
    """27 distinct warp sets exceed the renderer's palette, and its documented
    policy is to refuse; the batch therefore emits no dot file."""
    source = MAIN_SCRIPT.read_text()
    assert "--viz" not in source
    assert "fig9" not in source


def test_ablation_warp_cuts_are_literal_constants():
    source = ABLATION_SCRIPT.read_text()
    assert "WARP_MINUS_ONE=31" in source
    assert "WARP_HALF=16" in source
    assert "budget_basis" not in source
    assert "scheduled_warps" not in source
    assert "FIXED_WARPS" not in source


def test_ablation_cases_use_probe_module():
    source = ABLATION_SCRIPT.read_text()
    assert "--probe-first-window" not in source
    # Only the probe helper reaches the probe verb; only baseline solves.
    assert "-m paper_joint_solver.probe" in source
    solve_lines = [
        line for line in source.splitlines() if line.startswith("run_solve_case ")
    ]
    probe_lines = [
        line for line in source.splitlines() if line.startswith("run_probe_case ")
    ]
    assert [line.split()[1] for line in solve_lines] == ["baseline"]
    assert [line.split()[1] for line in probe_lines] == [
        "warps_minus_one",
        "warps_half",
        "no_subtiling",
        "no_cross_warp",
    ]


def test_ablation_validation_writes_report_not_gates():
    source = ABLATION_SCRIPT.read_text()
    assert "ablation_report.md" in source
    assert "first-window-probe" in source
    for gate in (
        "used a structural UNSAT shortcut",
        "did not force search beyond",
        "is not joint-UNSAT at the baseline SAT point",
        "lacks joint-UNSAT at its first ZLP point",
        "see diagnostics/README.md",
        "resource_limited",
        "expected_rcs",
        "write_diagnostic_report",
    ):
        assert gate not in source, f"run_ablations.sh still gates on {gate}"


def test_ablation_validation_tolerates_a_did_not_terminate_case(tmp_path):
    """spec-7 WI-8: a watchdogged case may be missing its JSON.

    The observation line takes the artifact's place; the batch still reports.
    """
    import re

    source = ABLATION_SCRIPT.read_text()
    validation = re.findall(r"<<'PY'\n(.*?)\nPY\n", source, re.S)[-1]
    script = tmp_path / "validate.py"
    script.write_text(validation)

    out = tmp_path / "ablations_lit1"
    out.mkdir()
    (out / "environment.log").write_text("paper_comparable=yes\n")
    (out / "observations.log").write_text(
        "".join(
            f"observation label={label} "
            f"outcome=did-not-terminate-within-86400s rc=124\n"
            for label in (
                "baseline.solve",
                "warps_minus_one.probe",
                "warps_half.probe",
                "no_subtiling.probe",
                "no_cross_warp.probe",
            )
        )
    )
    for stem in ("sub", "fwd"):
        (out / f"{stem}.curated_ddg.json").write_text("{}")
        (out / f"{stem}.curation_manifest.json").write_text("{}")

    completed = subprocess.run(
        [sys.executable, str(script), str(out), "no"],
        capture_output=True,
        text=True,
        cwd=PKG_ROOT,
        env={**os.environ, "PYTHONPATH": str(PKG_ROOT)},
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = (out / "ablation_report.md").read_text()
    for name in ("baseline", "warps_half", "no_cross_warp"):
        assert f"| {name} |" in report
    assert "did-not-terminate" in report


def test_ablation_default_out_marks_the_lit1_batch():
    source = ABLATION_SCRIPT.read_text()
    assert "DEFAULT_OUT=ablations_lit1" in source


@pytest.mark.parametrize("script_path", RUN_SCRIPTS)
def test_scripts_declare_the_watchdog_in_their_environment_log(script_path):
    source = script_path.read_text()
    assert "WATCHDOG_S=${WATCHDOG_S:-86400}" in source
    assert "printf 'watchdog_s=%s\\n' \"$WATCHDOG_S\"" in source
    assert "timeout" in source
