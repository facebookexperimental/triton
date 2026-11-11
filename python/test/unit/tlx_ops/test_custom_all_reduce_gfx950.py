"""Public ``tlx.ops.custom_all_reduce`` coverage for gfx950.

Multi-rank cases spawn one process per GPU, so they need 2, 4 or 8 gfx950 GPUs in one xGMI domain.
"""

import datetime
import os
import socket

import pytest
import torch

from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedBackward
from triton.tlx.ops import custom_all_reduce, prepare_custom_all_reduce
from triton.tlx.ops._catalog import CATALOG

requires_gfx950 = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")


def _gpus():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_catalog_entries():
    entries = {(spec.op, spec.arch): spec for spec in CATALOG if spec.op.endswith("custom_all_reduce")}
    assert set(entries) == {("prepare_custom_all_reduce", "gfx950"), ("custom_all_reduce", "gfx950")}
    assert not any(spec.supports_backward for spec in entries.values())
    assert entries[("custom_all_reduce", "gfx950")].dtypes == {"float16", "bfloat16", "float32"}
    accepts = entries[("prepare_custom_all_reduce", "gfx950")].accepts
    assert [w for w in range(1, 10) if accepts({"WORLD": w})] == [2, 4, 8]


@requires_gfx950
def test_prepare_needs_process_group():
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    with pytest.raises(InvalidInput, match="process group"):
        prepare_custom_all_reduce()


@requires_gfx950
def test_rejects_bad_inputs_before_dispatch():
    x = torch.ones(16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(InvalidInput, match="method"):
        custom_all_reduce(x, plan=None, method="3stage")
    with pytest.raises(InvalidInput, match="contiguous"):
        custom_all_reduce(torch.ones(4, 4, device="cuda", dtype=torch.bfloat16).T, plan=None)
    with pytest.raises(InvalidInput, match="alias"):
        custom_all_reduce(x, plan=None, out=x)
    with pytest.raises(InvalidInput, match="does not support"):
        custom_all_reduce(torch.ones(16, device="cuda", dtype=torch.int32), plan=None)
    with pytest.raises(InvalidInput, match="plan"):
        custom_all_reduce(x, plan=object())
    with pytest.raises(UnsupportedBackward):
        custom_all_reduce(x.clone().requires_grad_(), plan=None)


def _check_case(dist, plan, rank, world, numel, method, seed):
    gen = torch.Generator(device="cuda").manual_seed(1000 * rank + seed)
    x = torch.randn(numel, device="cuda", dtype=torch.bfloat16, generator=gen)
    ref = x.float()
    dist.all_reduce(ref)
    got = custom_all_reduce(x, plan, method=method)
    torch.testing.assert_close(got.float(), ref, atol=0.05 * world**0.5, rtol=2e-2)
    gathered = [torch.empty_like(got) for _ in range(world)]
    dist.all_gather(gathered, got)
    assert all(torch.equal(gathered[0].view(torch.int16), t.view(torch.int16)) for t in gathered), \
        f"ranks disagree bitwise (numel={numel}, method={method})"


def _check_graph_replay(plan, rank, world, tokens):
    from triton.tlx.ops.kernels.custom_all_reduce._shapes import MUSE_HIDDEN

    x = torch.zeros(MUSE_HIDDEN * tokens, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        custom_all_reduce(x, plan, out=out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        custom_all_reduce(x, plan, out=out)
    for rep in range(5):
        x.fill_(float(rank + 1 + rep))
        graph.replay()
        torch.cuda.synchronize()
        expect = float(sum(r + 1 + rep for r in range(world)))
        assert torch.all(out == expect), f"graph replay {rep} (tokens={tokens}) got {out[:4].tolist()}"


def _worker(rank, world, port):
    import torch.distributed as dist

    from triton.tlx.ops.kernels.custom_all_reduce._shapes import CORRECTNESS_NUMELS, GRAPH_TOKENS

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world, device_id=device,
                            timeout=datetime.timedelta(minutes=5))
    try:
        plan = prepare_custom_all_reduce(max_bytes=16 * 1024 * 1024)
        for method in (None, "1stage", "2stage"):
            for numel in CORRECTNESS_NUMELS:
                for seed in range(3):  # repeated calls advance the device-side barrier flags
                    _check_case(dist, plan, rank, world, numel, method, seed)
        for tokens in GRAPH_TOKENS:
            _check_graph_replay(plan, rank, world, tokens)
        too_big = torch.zeros(plan.max_bytes // 2 + 1, device=device, dtype=torch.bfloat16)
        with pytest.raises(InvalidInput, match="max_bytes"):
            custom_all_reduce(too_big, plan)
        plan.check()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@requires_gfx950
@pytest.mark.parametrize("world", [2, 4, 8])
def test_custom_all_reduce_multi_rank(world):
    if _gpus() < world:
        pytest.skip(f"needs {world} GPUs, found {_gpus()}")
    import torch.multiprocessing as mp

    # Spawn, not fork: the parent may already hold a GPU context from earlier tests.
    mp.spawn(_worker, args=(world, _free_port()), nprocs=world, join=True, start_method="spawn")
