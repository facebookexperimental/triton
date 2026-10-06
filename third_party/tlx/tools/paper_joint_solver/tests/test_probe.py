import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver import probe

_CURATED_STUB = {
    "schema_version": "curated-ddg-0.2",
    "curation_source": {
        "ddg_sha256": "0" * 64,
        "baseline_graph_sha256": "0" * 64,
        "curator_sources_sha256": "0" * 64,
    },
    "loops": [],
}


def _stub_solvers(monkeypatch, joint):
    def modulo(prob, ii):
        return None if ii < 2 else SimpleNamespace(ii=ii, length=3)

    monkeypatch.setattr(probe, "solve_modulo", modulo)
    monkeypatch.setattr(probe, "solve_joint", joint)


def _run_cli(monkeypatch, tmp_path, argv_extra=()):
    monkeypatch.setattr(
        probe, "load_problem", lambda *args, **kwargs: SimpleNamespace()
    )
    ddg = tmp_path / "curated_ddg.json"
    out = tmp_path / "probe.json"
    ddg.write_text(json.dumps(_CURATED_STUB))
    rc = probe.main([str(ddg), "-o", str(out), *argv_extra])
    return rc, json.loads(out.read_text())


def test_probe_reports_full_unsat_window(monkeypatch, tmp_path):
    _stub_solvers(
        monkeypatch, lambda prob, ii, length, **kwargs: (None, "unsat")
    )

    rc, payload = _run_cli(monkeypatch, tmp_path)

    assert rc == 0
    assert payload["ii"] == 2
    # I=2, LEN(M)=3 -> the ceiling window is L in {3, 4}.
    assert [point["L"] for point in payload["window"]] == [3, 4]
    assert payload["window_size"] == payload["unsat_count"] == 2
    assert payload["sat_found"] is False


def test_probe_stamps_its_own_source_digest(monkeypatch, tmp_path):
    """Two hashes, two scopes: the shared core, and this module.

    The core digest cannot cover ``probe.py`` -- widening it would invalidate
    every solve artifact on a probe-only edit -- so the probe carries its own.
    """
    _stub_solvers(
        monkeypatch, lambda prob, ii, length, **kwargs: (None, "unsat")
    )

    _, payload = _run_cli(monkeypatch, tmp_path)

    provenance = payload["provenance"]
    assert provenance["probe_sources_sha256"] == probe.probe_sources_sha256()
    assert len(provenance["probe_sources_sha256"]) == 64
    # Distinct scopes, so distinct digests.
    assert provenance["probe_sources_sha256"] != provenance["solver_sources_sha256"]


def test_probe_source_digest_tracks_the_module_bytes():
    import hashlib

    expected = hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()

    assert probe.probe_sources_sha256() == expected


def test_probe_records_sat_but_finishes_window(monkeypatch, tmp_path):
    def joint(prob, ii, length, **kwargs):
        if length == 3:
            return SimpleNamespace(), "sat"
        return None, "unsat"

    _stub_solvers(monkeypatch, joint)

    rc, payload = _run_cli(monkeypatch, tmp_path)

    assert rc == 0
    assert payload["sat_found"] is True
    assert payload["window_size"] == 2
    assert payload["unsat_count"] == 1
    assert [point["verdict"] for point in payload["window"]] == [
        "sat",
        "unsat",
    ]


@pytest.mark.parametrize("verdict", ["sat", "unsat"])
def test_probe_rc_is_zero_regardless_of_verdicts(
    monkeypatch, tmp_path, verdict
):
    _stub_solvers(
        monkeypatch,
        lambda prob, ii, length, **kwargs: (
            (SimpleNamespace(), "sat") if verdict == "sat" else (None, "unsat")
        ),
    )

    rc, payload = _run_cli(monkeypatch, tmp_path)

    assert rc == 0
    assert payload["schema_version"] == "paper-joint-probe-v1"
    assert payload["execution_mode"] == "first-window-probe"


def test_probe_forwards_experiment_inputs(monkeypatch, tmp_path):
    seen = []

    def joint(prob, ii, length, **kwargs):
        seen.append(kwargs)
        return None, "unsat"

    _stub_solvers(monkeypatch, joint)

    rc, payload = _run_cli(
        monkeypatch,
        tmp_path,
        ("--no-cross-warp", "--num-warps-override", "16"),
    )

    assert rc == 0
    provenance = payload["provenance"]
    assert provenance["model"] == "paper"
    assert "baseline_graph_sha256" not in provenance
    assert provenance["experiment_inputs"] == {
        "allow_cross_warp": False,
        "num_warps_override": 16,
        "reg_budget": None,
    }
    assert seen[0] == {"num_warps_override": 16, "allow_cross_warp": False}


def test_probe_cli_has_no_baseline_graph_or_model_switches(monkeypatch, tmp_path):
    monkeypatch.setattr(
        probe, "load_problem", lambda *args, **kwargs: SimpleNamespace()
    )
    ddg = tmp_path / "curated_ddg.json"
    ddg.write_text(json.dumps(_CURATED_STUB))

    with pytest.raises(SystemExit) as error:
        probe.main(
            [
                str(ddg),
                "-o",
                str(tmp_path / "probe.json"),
                "--baseline-graph",
                str(tmp_path / "schedule_graph.json"),
            ]
        )

    assert error.value.code == 2
