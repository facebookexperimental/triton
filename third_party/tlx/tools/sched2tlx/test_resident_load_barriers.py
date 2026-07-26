from pathlib import Path

from sched2tlx import emitter, schedule_graph

FIXTURE = (
    Path(__file__).parent / "examples/case11_wait_order/schedule_graph.json"
)


def test_transposed_resident_mma_operands_wait_for_tma():
    src = emitter.emit(schedule_graph.load_graph(FIXTURE))

    consumers = {
        "q_smem_6": "tlx.async_dot(L0_acc_tmem_2[0], q_smem_6[0]",
        "q_smem_7": "tlx.local_trans(q_smem_7[0])",
        "q_smem_8": "tlx.local_trans(q_smem_8[0])",
    }
    for alloc_var, consumer in consumers.items():
        wait = f"tlx.barrier_wait({alloc_var}_full[0], 0)"
        assert src.count(wait) == 1
        assert src.index(wait) < src.index(consumer)
