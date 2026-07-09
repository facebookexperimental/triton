"""Multi-WG OUTER loop bodies lower correctly (guard-3 retirement).

The fixture is case2 (persistent GEMM) with its outer epilogue store split
into its own warp group — the exact shape that historically made the emitter
silently drop the store (the partitioner-side guard 3 existed to prevent it).
The split WG must get its own async task, and the cvt→store hand-off must
travel through a synthesized SMEM channel with outer-iteration phases:
producer waits empty / stores / fences / arrives full; the store task waits
full, TMA-stores straight from the channel buffer, drains, arrives empty.

GPU-validated 2026-07-09 on B200: correctness rel=0.0 on four shapes, perf
identical to the single-WG emission (1.00/0.88/0.91/0.97x of handwritten).
This test pins the emission protocol without needing a GPU.
"""

from pathlib import Path

from sched2tlx import emitter, schedule_graph

FIXTURE = Path(__file__).parent / "test_outer_multiwg_split.schedule_graph.json"


def test_outer_multiwg_split_lowers():
    src = emitter.emit(schedule_graph.load_graph(FIXTURE))

    # The split outer WG gets its own task.
    assert "# Async task: role=TMA ← outer wg3" in src

    # Producer side (default task): stage the tile into the channel with the
    # full/empty handshake, fenced for the TMA consumer.
    assert "tlx.barrier_wait(sem2_b3_empty[0], ((_it & 1) ^ 1))" in src
    assert "tlx.local_store(L1_smem_3[0], trunc_" in src
    assert "tlx.fence_async_shared()" in src
    assert "tlx.barrier_arrive(sem2_b3_full[0], 1)" in src

    # Consumer side (store task): wait full, TMA straight from the channel
    # (no register round-trip), drain, recycle.
    assert "tlx.barrier_wait(sem2_b3_full[0], (_it & 1))" in src
    assert "tlx.async_descriptor_store(c_desc, L1_smem_3[0]," in src
    assert "tlx.barrier_arrive(sem2_b3_empty[0], 1)" in src

    # Outer-iteration counter drives the phases in both tasks.
    assert src.count("_oit = 0") == 2
    assert src.count("_oit += 1") == 2

    # No op may be silently dropped: the store appears exactly once.
    assert src.count("tlx.async_descriptor_store(c_desc,") == 1


def test_outer_multiwg_split_deterministic():
    g1 = schedule_graph.load_graph(FIXTURE)
    g2 = schedule_graph.load_graph(FIXTURE)
    assert emitter.emit(g1) == emitter.emit(g2)
