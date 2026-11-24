"""Public ``tlx.ops.all_reduce`` and ``tlx.ops.all_reduce_norm_residual_norm`` coverage for gfx950."""

from __future__ import annotations

import functools
import importlib
import os
import pathlib
import socket
import sys
import time
from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp

from triton._internal_testing import is_hip_cdna4
from triton.tlx import ops
from triton.tlx.ops import InvalidInput, UnsupportedBackward
from triton.tlx.ops._catalog import CATALOG
from triton.tlx.ops.kernels.collectives.all_reduce import gfx950 as impl
from triton.tlx.ops.kernels.collectives.all_reduce._shapes import CORRECTNESS_SHAPES, AllReduceShape, partial, reference
from triton.tlx.ops.kernels.collectives.fused.all_reduce_norm_residual_norm import gfx950 as fused_impl
from triton.tlx.ops.kernels.fused.norm_residual_norm._shapes import NormResidualNormShape, inputs

_SPAWN_TIMEOUT_SECONDS = 900


def _count_gfx950(queue):
    queue.put(torch.cuda.device_count() if is_hip_cdna4() else 0)


@functools.lru_cache(maxsize=None)
def _gpus():
    """gfx950 device count, probed in a child so this process can still fork GPU workers."""
    if torch.cuda.is_initialized():
        return torch.cuda.device_count() if is_hip_cdna4() else 0
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    probe = ctx.Process(target=_count_gfx950, args=(queue, ))
    probe.start()
    count = queue.get(timeout=120)
    probe.join()
    return count


def test_catalog_entries():
    specs = {spec.op: spec for spec in CATALOG if spec.arch == "gfx950"}
    plain = specs["all_reduce"]
    fused = specs["all_reduce_norm_residual_norm"]
    assert plain.dtypes == frozenset({"bfloat16"}) and not plain.supports_backward
    assert all(plain.accepts(dict(world=w)) for w in (2, 4, 8))
    assert not plain.accepts(dict(world=3)) and not plain.accepts(dict(world=16))
    assert fused.accepts(dict(world=8, H=8192)) and fused.accepts(dict(world=8, H=16384))
    assert not fused.accepts(dict(world=8, H=12288)) and not fused.accepts(dict(world=6, H=8192))
    assert "all_reduce_workspace" in specs


def test_ops_are_exported():
    for name in ("all_reduce", "all_reduce_workspace", "all_reduce_norm_residual_norm"):
        assert name in ops.__all__


def test_algorithm_choice():
    assert impl.pick_all_reduce(8192, 8) == "one_shot"
    assert impl.pick_all_reduce(5 * 8192, 8) == "one_shot"
    assert impl.pick_all_reduce(6 * 8192, 8) == "two_shot"
    assert impl.pick_all_reduce(10 * 4096, 8) == "one_shot"
    assert impl.pick_all_reduce(11 * 4096, 8) == "two_shot"
    assert impl.two_shot_config(18 * 4096, 8) == (128, 72)
    assert impl.two_shot_config(64 * 4096, 8) == (256, 128)
    assert impl.two_shot_config(100 * 4096, 8) == (512, 100)
    assert impl.two_shot_config(151 * 4096, 8) == (512, 151)
    assert impl.two_shot_config(256 * 4096, 8) == (128, impl.MAX_PROGRAMS)
    assert impl.two_shot_config(1024 * 4096, 8) == (512, impl.PROGRAMS)
    # Two-shot partitions must start on 16-byte boundaries.
    assert impl.pick_all_reduce(16 * 8192 + 8, 8) == "one_shot"
    assert fused_impl.pick_kernel(1, 8192) == "coop" and fused_impl.pick_kernel(8, 8192) == "coop"
    assert fused_impl.pick_kernel(9, 8192) is None
    assert fused_impl.pick_kernel(1, 4096) == "row" and fused_impl.pick_kernel(24, 4096) == "row"
    assert fused_impl.pick_kernel(25, 4096) is None and fused_impl.pick_kernel(1, 16384) is None
    # Every program of a cooperative call must be resident at once, and each has barrier slots.
    assert fused_impl.COOP_ROWS[8192][1] * fused_impl.COOP_PARTS <= impl.MAX_PROGRAMS
    assert fused_impl.COOP_ROWS[8192][1] <= impl.ROW_SYNC_ROWS and fused_impl.COOP_PARTS <= impl.ROW_SYNC_PARTS


class _OtherWorkspace:
    world = 8

    def holds(self, x):
        return False


def test_rejects_input_outside_the_workspace():
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    with pytest.raises(InvalidInput, match="workspace.input"):
        ops.all_reduce(x, _OtherWorkspace())
    with pytest.raises(InvalidInput, match="workspace.input"):
        ops.all_reduce_norm_residual_norm(torch.zeros(4, 8), x, _OtherWorkspace())
    with pytest.raises(InvalidInput, match="BF16"):
        ops.all_reduce(x.float(), _OtherWorkspace())


class _HoldingWorkspace(_OtherWorkspace):

    def holds(self, x):
        return True


@pytest.mark.skipif(_gpus() < 1, reason="Requires a gfx950 GPU")
def test_rejects_grad_tracked_outputs():
    workspace = _HoldingWorkspace()
    x = torch.zeros(4, 4096, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(UnsupportedBackward):
        ops.all_reduce(x, workspace, out=torch.empty_like(x, requires_grad=True))
    kw = inputs(NormResidualNormShape(4, 4096, "ffn"))
    for name, like in (("out", kw["branch"]), ("new_stream", kw["stream"])):
        with pytest.raises(UnsupportedBackward):
            ops.all_reduce_norm_residual_norm(workspace=workspace, **kw,
                                              **{name: torch.empty_like(like, requires_grad=True)})


@pytest.mark.skipif(_gpus() < 1, reason="Requires a gfx950 GPU")
def test_rejects_misaligned_out():
    x = torch.zeros(4, 4096, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(x.numel() + 1, dtype=x.dtype, device="cuda")[1:].view(x.shape)
    with pytest.raises(InvalidInput, match="16-byte aligned"):
        ops.all_reduce(x, _HoldingWorkspace(), out=out)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn(worker, world, *args):
    if torch.cuda.is_initialized():
        # A forkserver child unpickles the worker by module name, which must not depend on pytest's import mode.
        here = str(pathlib.Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.append(here)
        worker = getattr(importlib.import_module(pathlib.Path(__file__).stem), worker.__name__)
        mp.set_forkserver_preload(["torch", "triton"])
        method = "forkserver"
    else:
        method = "fork"
    context = mp.start_processes(worker, args=(world, _free_port(), *args), nprocs=world, join=False,
                                 start_method=method)
    deadline = time.monotonic() + _SPAWN_TIMEOUT_SECONDS
    try:
        while not context.join(timeout=max(deadline - time.monotonic(), 1), grace_period=5):
            if time.monotonic() > deadline:
                raise TimeoutError(f"{worker.__name__} exceeded {_SPAWN_TIMEOUT_SECONDS} s")
    except BaseException:
        for process in context.processes:
            if process.is_alive():
                process.kill()
        raise


def _init(rank, world, port):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank),
                            timeout=timedelta(minutes=5))


def _all_reduce_worker(rank, world, port):
    import torch.distributed as dist

    _init(rank, world, port)
    try:
        largest = max(s.tokens * s.H for s in CORRECTNESS_SHAPES)
        workspace = ops.all_reduce_workspace(largest)
        for shape in CORRECTNESS_SHAPES:
            x = workspace.input(shape.tokens, shape.H)
            x.copy_(partial(shape, rank))
            got = ops.all_reduce(x, workspace)
            want = reference([partial(shape, r) for r in range(world)])
            assert torch.equal(got, want), f"rank {rank} {shape}"
        # Both algorithms on shapes the heuristic sends to the other one, and an odd length only one-shot takes.
        for kind, shape in (("two_shot", AllReduceShape(1, 4096)), ("one_shot", AllReduceShape(300, 4096)),
                            ("one_shot", AllReduceShape(1, 4104))):
            impl.pick_all_reduce = lambda numel, w, kind=kind: kind
            x = workspace.input(shape.tokens, shape.H)
            x.copy_(partial(shape, rank, seed=1))
            assert torch.equal(ops.all_reduce(x, workspace),
                               reference([partial(shape, r, seed=1) for r in range(world)])), (kind, shape)
        workspace.close()
        with pytest.raises(InvalidInput, match="closed"):
            ops.all_reduce(x, workspace)
        assert not workspace.holds(x)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _setup_failure_worker(rank, world, port, phase, failing):
    import torch.distributed as dist

    _init(rank, world, port)
    try:
        real = impl._hip()
        freed = []

        class _Spy:

            def __getattr__(self, name):
                return getattr(real, name)

            def hipFree(self, ptr):
                freed.append(ptr.value)
                return real.hipFree(ptr)

            def hipIpcOpenMemHandle(self, *args):
                return 1 if phase == "open" and rank in failing else real.hipIpcOpenMemHandle(*args)

        impl._hip = _Spy
        allocate = impl.Workspace._uncached_alloc
        owned = []

        def fail_second(self, nbytes):
            if phase == "alloc" and rank in failing and owned:
                raise RuntimeError("injected allocation failure")
            owned.append(allocate(self, nbytes))
            return owned[-1]

        impl.Workspace._uncached_alloc = fail_second
        match = ("injected" if phase == "alloc" else "hipIpcOpenMemHandle") if rank in failing else "setup failed"
        with pytest.raises(RuntimeError, match=match):
            ops.all_reduce_workspace(1024)
        assert sorted(freed) == sorted(owned), (freed, owned)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _graph_worker(rank, world, port):
    import torch.distributed as dist

    _init(rank, world, port)
    try:
        workspace = ops.all_reduce_workspace(64 * 8192)
        for shape in (AllReduceShape(1, 8192), AllReduceShape(64, 8192)):
            x = workspace.input(shape.tokens, shape.H)
            out = torch.empty_like(x)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                ops.all_reduce(x, workspace, out=out)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(3):
                    ops.all_reduce(x, workspace, out=out)
            for seed in range(3):
                x.copy_(partial(shape, rank, seed=seed))
                torch.cuda.synchronize()
                dist.barrier()
                graph.replay()
                assert torch.equal(out, reference([partial(shape, r, seed=seed) for r in range(world)])), seed
        workspace.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _fused_worker(rank, world, port):
    import torch.distributed as dist

    _init(rank, world, port)
    try:
        workspace = ops.all_reduce_workspace(300 * 8192)
        # Several seeds in the cooperative range: a row sum that is off by one ulp flips only a few BF16 outputs.
        cases = [(T, seed) for T in (1, 2, 4, 6, 7, 8) for seed in range(8)]
        cases += [(T, T) for T in (9, 13, 24, 27, 64, 300)]
        for T, seed in cases:
            for H in (4096, 8192):
                for boundary in ("attn", "ffn", "final"):
                    kw = inputs(NormResidualNormShape(T, H, boundary), seed=seed)
                    shape = AllReduceShape(T, H)
                    branch = workspace.input(T, H)
                    branch.copy_(partial(shape, rank, seed=seed))
                    kw["branch"] = reference([partial(shape, r, seed=seed) for r in range(world)])
                    want_out, want_stream = ops.norm_residual_norm(**kw)
                    kw["branch"] = branch
                    got_out, got_stream = ops.all_reduce_norm_residual_norm(workspace=workspace, **kw)
                    assert torch.equal(got_out, want_out), (rank, T, H, boundary)
                    assert torch.equal(got_stream, want_stream), (rank, T, H, boundary)
        workspace.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _fused_graph_worker(rank, world, port):
    import torch.distributed as dist

    _init(rank, world, port)
    try:
        workspace = ops.all_reduce_workspace(32 * 8192)
        # Two cooperative sizes captured back to back: the per-row counters must be clear at the start of every
        # call, also on replay.
        calls = []
        for T in (8, 3):
            kw = inputs(NormResidualNormShape(T, 8192, "attn"), seed=T)
            kw["out"] = torch.empty_like(kw["branch"])
            kw["new_stream"] = torch.empty_like(kw["stream"])
            kw["branch"] = workspace.input(T, 8192)
            calls.append((T, kw))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _, kw in calls:
                ops.all_reduce_norm_residual_norm(workspace=workspace, **kw)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _, kw in calls:
                ops.all_reduce_norm_residual_norm(workspace=workspace, **kw)
        for seed in range(3):
            shape = AllReduceShape(8, 8192)
            # Both calls read the same workspace rows; the second (3 rows) reads the first 3 rows of the input.
            workspace.input(8, 8192).copy_(partial(shape, rank, seed=seed))
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            reduced = reference([partial(shape, r, seed=seed) for r in range(world)])
            for T, kw in calls:
                want = dict(kw, branch=reduced[:T].contiguous(), out=None, new_stream=None)
                want_out, want_stream = ops.norm_residual_norm(**want)
                assert torch.equal(kw["out"], want_out), (rank, seed, T)
                assert torch.equal(kw["new_stream"], want_stream), (rank, seed, T)
        workspace.close()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(_gpus() < 8, reason="Requires 8 gfx950 GPUs")
def test_all_reduce_is_the_rank_order_sum_on_8_gpus():
    _spawn(_all_reduce_worker, 8)


@pytest.mark.skipif(_gpus() < 4, reason="Requires 4 gfx950 GPUs")
@pytest.mark.parametrize("world", [2, 4])
def test_all_reduce_smaller_groups(world):
    _spawn(_all_reduce_worker, world)


@pytest.mark.skipif(_gpus() < 2, reason="Requires 2 gfx950 GPUs")
@pytest.mark.parametrize("phase", ["alloc", "open"])
@pytest.mark.parametrize("failing", [(0, 1), (1, )], ids=["all_ranks", "one_rank"])
def test_workspace_setup_failure_raises_on_every_rank_and_frees(phase, failing):
    _spawn(_setup_failure_worker, 2, phase, failing)


@pytest.mark.skipif(_gpus() < 8, reason="Requires 8 gfx950 GPUs")
def test_all_reduce_replays_in_a_graph():
    _spawn(_graph_worker, 8)


@pytest.mark.skipif(_gpus() < 8, reason="Requires 8 gfx950 GPUs")
def test_fused_equals_all_reduce_then_boundary():
    _spawn(_fused_worker, 8)


@pytest.mark.skipif(_gpus() < 8, reason="Requires 8 gfx950 GPUs")
def test_fused_replays_in_a_graph():
    _spawn(_fused_graph_worker, 8)
