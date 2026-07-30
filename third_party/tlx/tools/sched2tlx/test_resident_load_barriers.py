from pathlib import Path

from sched2tlx import emitter, schedule_graph, semaphore_ir

FIXTURE = Path(__file__).parent / "examples/case11_wait_order/schedule_graph.json"
BWD_FIXTURE = Path(__file__).parent / "examples/case4_FA_bwd/schedule_graph_hd128.json"


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


def _semaphore_for(semaphores, producer, consumer):
    return next(
        semaphore
        for semaphore in semaphores
        if semaphore.producers[0].node.node_id == producer
        and semaphore.consumers[0].node.node_id == consumer
    )


def test_legacy_barrier_distance_comes_from_dependence_edge():
    graph = schedule_graph.load_graph(BWD_FIXTURE)
    loop = next(loop for loop in graph.loops if not loop.is_outer)
    distances = {
        (barrier.producer_node, barrier.consumer_node): barrier.distance
        for barrier in loop.schedule.cross_wg_barriers
    }

    assert distances[1, 10] == 0
    assert distances[12, 11] == 1


def test_semaphore_direction_uses_distance_not_cycle_order():
    graph = schedule_graph.load_graph(BWD_FIXTURE)
    loop = next(loop for loop in graph.loops if not loop.is_outer)
    nodes = {node.id: node for node in loop.schedule.nodes}
    nodes[1].schedule_cycle = 1000
    nodes[10].schedule_cycle = 0
    nodes[12].schedule_cycle = 0
    nodes[11].schedule_cycle = 1000

    semaphores = semaphore_ir.derive_semaphores(loop, graph)
    forward = _semaphore_for(semaphores, 1, 10)
    carried = _semaphore_for(semaphores, 12, 11)

    assert not forward.is_released
    assert forward.buffer is not None
    assert carried.is_released
    assert carried.buffer is None
