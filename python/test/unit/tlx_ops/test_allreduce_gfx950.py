"""Correctness of the gfx950 all-reduce kernels across every local GPU (one process per GPU)."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from triton._internal_testing import is_hip_cdna4

# Multi-GPU gate: this test only runs where at least two gfx950 GPUs are visible. Single-GPU CI workers skip it;
# coverage comes from manual runs on a multi-GPU host (see the test plan).
pytestmark = pytest.mark.skipif(not is_hip_cdna4() or torch.cuda.device_count() < 2,
                                reason="needs at least two gfx950 GPUs (skipped on single-GPU CI workers)")

# numel values: tiny, not a multiple of the per-program block, and large enough for a grid-stride loop.
SIZES = [8, 4096, 8 * 1000, 9 * 4096, 14 * 8192, 64 * 4096 + 8 * 24]


def _check(out, ref, world, group, what):
    err = (out.float() - ref).abs().max().item()
    tol = 2**-7 * ref.abs().max().item() + 1e-2
    assert err <= tol, f"{what}: max abs error {err} > {tol}"
    gathered = [None] * world
    dist.all_gather_object(gathered, out.view(torch.int16).cpu(), group=group)
    assert all(torch.equal(gathered[0], o) for o in gathered), f"{what}: ranks disagree"


def _inputs(world, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(world, n, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)


def _worker(rank, world, port, errors):
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        torch.cuda.set_device(rank)
        dist.init_process_group("gloo", rank=rank, world_size=world)
        group = dist.group.WORLD
        from triton.tlx.ops.kernels.allreduce.gfx950 import CustomAllReduce

        comm = CustomAllReduce(group, max_bytes=4 * 1024 * 1024)
        algos = ["one_stage"]
        for n in SIZES:
            xs = _inputs(world, n, seed=n)
            ref = xs.float().sum(0)
            for algo in algos:
                out = torch.full((n, ), float("nan"), dtype=torch.bfloat16, device="cuda")
                comm.input_buffer(n).copy_(xs[rank])
                getattr(comm, algo)(n, out)
                torch.cuda.synchronize()
                _check(out, ref, world, group, f"{algo} n={n}")
            out = comm.all_reduce(xs[rank].clone())
            _check(out, ref, world, group, f"all_reduce n={n}")

        # Back-to-back calls with fresh inputs, alternating algorithms and sizes, without host synchronization.
        outs, refs = [], []
        for i in range(40):
            n = SIZES[i % len(SIZES)]
            xs = _inputs(world, n, seed=1000 + i)
            comm.input_buffer(n).copy_(xs[rank])
            out = torch.empty(n, dtype=torch.bfloat16, device="cuda")
            getattr(comm, algos[i % len(algos)])(n, out)
            outs.append(out)
            refs.append(xs.float().sum(0))
        torch.cuda.synchronize()
        for i, (out, ref) in enumerate(zip(outs, refs)):
            _check(out, ref, world, group, f"back-to-back call {i}")

        # Graph replay: every replay must redo the barriers, not reuse the captured flag values.
        n = 14 * 8192
        xs = _inputs(world, n, seed=7)
        ref = xs.float().sum(0)
        comm.input_buffer(n).copy_(xs[rank])
        for algo in algos:
            out = torch.empty(n, dtype=torch.bfloat16, device="cuda")
            getattr(comm, algo)(n, out)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(5):
                    getattr(comm, algo)(n, out)
            for _ in range(3):
                out.fill_(float("nan"))
                dist.barrier(group=group)
                graph.replay()
                torch.cuda.synchronize()
                _check(out, ref, world, group, f"{algo} graph replay")
            del graph
        dist.barrier(group=group)
        comm.close()
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001 - reported to the parent, which fails the test
        errors.put(f"rank {rank}: {e!r}")
        raise


def test_allreduce_gfx950():
    world = min(torch.cuda.device_count(), 8)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    errors = ctx.SimpleQueue()
    mp.start_processes(_worker, args=(world, port, errors), nprocs=world, join=True, start_method="spawn")
    assert errors.empty(), errors.get()
