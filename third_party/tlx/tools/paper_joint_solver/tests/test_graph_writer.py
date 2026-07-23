import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_joint_solver.graph_writer import rewrite_schedule_graph, validate

TOOLS = Path(__file__).resolve().parents[2]
SCHED2TLX = TOOLS / "sched2tlx"
BASELINE = SCHED2TLX / "examples" / "case3_FA_fp16" / "schedule_graph.json"
VENV_PY = Path(__file__).resolve().parents[5] / ".venv" / "bin" / "python"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable


def _emit(graph_path, out_py):
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    subprocess.run(
        [PYTHON, "-m", "sched2tlx", str(graph_path), "-o", str(out_py)],
        cwd=str(SCHED2TLX), env=env, check=True, capture_output=True)
    return out_py.read_text()


def _non_comment(src):
    return "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def _identity_solution():
    data = json.loads(BASELINE.read_text())
    sl = data["loops"][0]["schedule_loop"]
    nodes = sl["graph"]["nodes"]
    cycles = {n["id"]: n["schedule"]["cycle"] for n in nodes}
    warp = {n["id"]: n["warp_group"] for n in nodes if n["warp_group"] >= 0}
    return sl["II"], cycles, warp, max(cycles.values()) + 1


def test_identity_roundtrip_emits_identically(tmp_path):
    ii, cycles, warp, length = _identity_solution()
    out_json = tmp_path / "rewritten.json"
    rewrite_schedule_graph(BASELINE, out_json,
                           ii=ii, cycles=cycles, warp=warp, length=length)
    base_py = _emit(BASELINE, tmp_path / "base.py")
    rew_py = _emit(out_json, tmp_path / "rewritten.py")
    assert _non_comment(rew_py) == _non_comment(base_py)


def test_identity_roundtrip_validates(tmp_path):
    ii, cycles, warp, length = _identity_solution()
    out_json = tmp_path / "rewritten.json"
    rewrite_schedule_graph(BASELINE, out_json,
                           ii=ii, cycles=cycles, warp=warp, length=length)
    assert validate(out_json) == []


def test_permuted_warp_assignment_validates(tmp_path):
    ii, cycles, warp, length = _identity_solution()
    ids = sorted(set(warp.values()))
    rot = {w: ids[(i + 1) % len(ids)] for i, w in enumerate(ids)}
    permuted = {nid: rot[w] for nid, w in warp.items()}
    out_json = tmp_path / "permuted.json"
    rewrite_schedule_graph(BASELINE, out_json,
                           ii=ii, cycles=cycles, warp=permuted, length=length)
    assert validate(out_json) == []
    data = json.loads(out_json.read_text())
    loop = data["loops"][0]
    used = {n["warp_group"] for n in loop["schedule_loop"]["graph"]["nodes"]
            if n["warp_group"] >= 0}
    assert used == {w["id"] for w in loop["warp_groups"]}
    assert sorted(used) == list(range(len(used)))
