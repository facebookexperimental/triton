import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from paper_joint_solver.ddg import load_problem
from paper_joint_solver.viz import render, to_dot

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "sched2tlx"
    / "examples"
    / "case3_FA_fp16_subtiled"
)
DDG = (
    Path(__file__).resolve().parent
    / "golden_curated"
    / "fwd_subtiled"
    / "fwd_subtiled.curated_ddg.json"
)


@pytest.fixture(scope="module")
def prob():
    return load_problem(DDG)


@pytest.fixture(scope="module")
def warp(prob):
    """Fig-9-shaped fabricated assignment: TMA loads on the variable-latency
    warp 0, MMAs on the TC warp 1, the two softmax exp chains alternating
    on warps 2/3, everything else on warp 4.  The paper assigns warp *sets*,
    so every entry here is a one-element set."""
    out, exp_seen = {}, 0
    for v in sorted(prob.nodes.values(), key=lambda v: v.id):
        if v.pipeline == "TMA":
            out[v.id] = (0,)
        elif "mma" in v.op_kind:
            out[v.id] = (1,)
        elif "exp2" in v.op_kind:
            out[v.id] = (2 + exp_seen % 2,)
            exp_seen += 1
        else:
            out[v.id] = (4,)
    return out


def test_to_dot_structure(prob, warp):
    assert prob.streaming == {2, 3}

    dot = to_dot(prob, warp)
    assert dot.startswith("digraph")
    assert dot.rstrip().endswith("}")
    assert "fillcolor" in dot

    node_ids = {int(m) for m in re.findall(r"^  n(\d+) \[", dot, re.M)}
    assert len(node_ids) > 10
    # Zero-cost helpers are contracted away, tile ops survive.
    assert "expand_dims" not in dot and "broadcast" not in dot
    assert "TC GEMM" in dot and "TMA load" in dot and "row_red" in dot
    for vid in node_ids:
        v = prob.nodes[vid]
        assert v.latency > 0 or v.occupancy > 0
    # Edges only reference emitted nodes; loop-carried ones are dashed.
    for src, dst in re.findall(r"^  n(\d+) -> n(\d+)", dot, re.M):
        assert int(src) in node_ids and int(dst) in node_ids
    assert "style=dashed" in dot


def test_to_dot_warp_colors_and_cycles(prob, warp):
    from paper_joint_solver.viz import PALETTE, UNASSIGNED_FILL
    dot = to_dot(prob, warp, cycles={v: 7 for v in prob.nodes})
    for warps in sorted(set(warp.values())):
        assert PALETTE[warps[0]] in dot
        assert f'label="warp {warps[0]}"' in dot  # legend names each set
    assert "\\nt=7" in dot
    # No warp map: neutral fill, no legend, no cycle line.
    plain = to_dot(prob, None)
    assert UNASSIGNED_FILL in plain
    assert "cluster_legend" not in plain and "t=" not in plain


def test_to_dot_reranks_sparse_warp_ids_without_color_collisions(prob):
    from paper_joint_solver.viz import PALETTE

    visible = [
        node.id
        for node in sorted(prob.nodes.values(), key=lambda node: node.id)
        if node.latency > 0 or node.occupancy > 0
    ]
    sparse_warps = (0, 8, 19, 31, 47, 63, 80, 101)
    sparse_warp = {
        node_id: (warp,) for node_id, warp in zip(visible[:8], sparse_warps)
    }
    dot = to_dot(prob, sparse_warp)

    for node_id, group, color in zip(visible, sparse_warps, PALETTE):
        node_line = re.search(rf"^  n{node_id} \[.*$", dot, re.M)
        assert node_line is not None
        assert f'fillcolor="{color}"' in node_line.group()
        assert f'label="warp {group}"' in dot
    assert len(set(PALETTE)) == len(sparse_warps)


def test_to_dot_rejects_more_warp_sets_than_the_palette(prob):
    from paper_joint_solver.viz import PALETTE

    visible = [
        node.id
        for node in sorted(prob.nodes.values(), key=lambda node: node.id)
        if node.latency > 0 or node.occupancy > 0
    ]
    warp = {
        node_id: (100 + index,)
        for index, node_id in enumerate(visible[: len(PALETTE) + 1])
    }
    with pytest.raises(ValueError, match=r"cannot render 9 distinct warp sets"):
        to_dot(prob, warp)


def test_cli_round_trip(tmp_path, warp):
    sol = tmp_path / "solution.json"
    # Ids as JSON strings on purpose — the CLI must coerce them.
    sol.write_text(json.dumps({
        "warp_sets": {str(k): list(v) for k, v in warp.items()},
        "cycles": {str(k): k * 3 for k in warp},
    }))
    out = tmp_path / "out.dot"
    env = dict(os.environ, PYTHONPATH=str(PKG_ROOT))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_joint_solver.viz",
            str(DDG),
            str(sol),
            "-o",
            str(out),
        ],
        check=True, env=env, cwd=tmp_path)
    dot = out.read_text()
    assert dot.startswith("digraph")
    assert "fillcolor" in dot and "t=" in dot
    assert dot == to_dot(
        load_problem(DDG),
        warp,
        cycles={k: k * 3 for k in warp},
    )


def test_render_writes_dot(tmp_path, prob, warp):
    out = render(prob, warp, tmp_path / "fig9.dot")
    assert out.read_text().startswith("digraph")
