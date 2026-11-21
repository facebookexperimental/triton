"""All-reduce latency on every local GPU: the TLX kernels against aiter's.

Spawns one process per GPU. Each algorithm is checked against an fp32 reference (and for bitwise agreement across
ranks), then timed as a HIP graph of ``--iters`` back-to-back calls; the reported time is the median over ``--trials``
replays of the slowest rank, in microseconds per call. The output of a final replay is checked again.

aiter's kernels come from a small shim library (``--aiter-lib``, built by
``fbcode/scripts/njriasan/tlx_allreduce/build_shim.sh``) that launches them on the same registered input buffers.
``--stress N`` instead runs N calls per algorithm with fresh inputs and random shapes and no host synchronization.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import random
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEFAULT_SHAPES = "1x4096,2x4096,4x4096,9x4096,1x8192,8x8192,14x4096,14x8192,36x4096,64x4096,128x4096"


class _Aiter:
    """aiter's kernels, launched through the shim on this group's registered input buffers."""

    def __init__(self, lib_path, comm, max_bytes):
        from triton.tlx.ops.kernels.allreduce._ipc import HIP_DEVICE_MALLOC_UNCACHED, IpcBuffer

        self.lib = ctypes.CDLL(lib_path)
        self.lib.aiter_signal_bytes.restype = ctypes.c_size_t
        self.lib.aiter_allreduce.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p
        ]
        # aiter's layout: a Signal block followed by the two-stage scratch buffer.
        self.meta = IpcBuffer(self.lib.aiter_signal_bytes() + max_bytes, rank=comm.rank, world_size=comm.world_size,
                              group=comm.group, flags=HIP_DEVICE_MALLOC_UNCACHED)
        self.sig_ptrs = (ctypes.c_void_p * comm.world_size)(*self.meta.ptrs)
        self.rank_data = torch.tensor(comm._input.ptrs, dtype=torch.int64, device="cuda")
        self.comm = comm

    def run(self, stages, n, out):
        err = self.lib.aiter_allreduce(stages, self.rank_data.data_ptr(), self.sig_ptrs, out.data_ptr(), self.comm.rank,
                                       self.comm.world_size, n,
                                       torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"aiter {stages}-stage launch failed: hipError {err}")

    def close(self):
        self.meta.close()


def _runners(comm, aiter, tlx_blocks):
    if len(tlx_blocks) == 1:
        runners = {"tlx_one_stage": lambda n, out: comm.one_stage(n, out, max_blocks=tlx_blocks[0])}
    else:
        runners = {f"tlx_one_stage_{b}": (lambda n, out, b=b: comm.one_stage(n, out, max_blocks=b)) for b in tlx_blocks}
    if aiter is not None:
        runners["aiter_1stage"] = lambda n, out: aiter.run(1, n, out)
    return runners


def _time(fn, iters, trials, group, out):
    torch.cuda.synchronize()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        dist.barrier(group=group)
        # Park the early ranks on the GPU so host-side barrier-return skew is absorbed before the aligning call; the
        # timed replay below is then enqueued with no host synchronization in between, keeping the skew out of the
        # slowest-rank number. torch.cuda._sleep spins the given number of GPU cycles (~13 us at 1.5 GHz).
        if hasattr(torch.cuda, "_sleep"):
            torch.cuda._sleep(20000)
        fn()  # aligning call
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    out.fill_(float("nan"))
    dist.barrier(group=group)
    graph.replay()
    torch.cuda.synchronize()
    del graph
    return statistics.median(times)


def _tolerance(ref):
    return 2**-7 * ref.abs().max().item() + 1e-2


def _bench(rank, world, comm, runners, algos, shapes, args, group):
    rows = []
    for tokens, hidden in shapes:
        n = tokens * hidden
        g = torch.Generator(device="cuda").manual_seed(n)
        xs = torch.randn(world, n, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        ref = xs.float().sum(0)
        comm.input_buffer(n).copy_(xs[rank])
        row = {}
        for algo in algos:
            out = torch.full((n, ), float("nan"), dtype=torch.bfloat16, device="cuda")
            runners[algo](n, out)
            torch.cuda.synchronize()
            err = (out.float() - ref).abs().max().item()
            gathered = [None] * world
            dist.all_gather_object(gathered, out.view(torch.int16).cpu(), group=group)
            ok = err <= _tolerance(ref) and all(torch.equal(gathered[0], o) for o in gathered)
            us = _time(lambda: runners[algo](n, out), args.iters, args.trials, group, out)
            ok = ok and (out.float() - ref).abs().max().item() <= _tolerance(ref)
            results = [None] * world
            dist.all_gather_object(results, (us, ok), group=group)
            row[algo] = (max(r[0] for r in results), all(r[1] for r in results))
        rows.append((f"{tokens}x{hidden}", n * 2 / 1024, row))
    if rank == 0:
        print(f"world={world} iters={args.iters} trials={args.trials} (us per all-reduce, slowest rank, "
              "X = wrong result)")
        print(f"{'shape':>10} {'KiB':>7} " + " ".join(f"{a:>14}" for a in algos))
        for shape, kib, row in rows:
            cells = " ".join(f"{row[a][0]:>13.2f}{' ' if row[a][1] else 'X'}" for a in algos)
            print(f"{shape:>10} {kib:>7.1f} {cells}")


def _stress(rank, world, comm, runners, algos, shapes, iters, group):
    for algo in algos:
        rng = random.Random(1234)
        errs = []
        for i in range(iters):
            tokens, hidden = rng.choice(shapes)
            n = tokens * hidden
            g = torch.Generator(device="cuda").manual_seed(i)
            xs = torch.randn(world, n, generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
            comm.input_buffer(n).copy_(xs[rank])
            out = torch.empty(n, dtype=torch.bfloat16, device="cuda")
            runners[algo](n, out)
            ref = xs.float().sum(0)
            errs.append(((out.float() - ref).abs().max() / (ref.abs().max() * 2**-7 + 1e-2)).reshape(1))
        worst = [None] * world
        dist.all_gather_object(worst, torch.cat(errs).max().item(), group=group)
        if rank == 0:
            print(
                f"stress {algo}: {iters} calls, worst err/tol {max(worst):.3f} "
                f"({'OK' if max(worst) <= 1 else 'FAIL'})", flush=True)


def _worker(rank, world, port, args):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.group.WORLD
    from triton.tlx.ops.kernels.allreduce.gfx950 import CustomAllReduce

    max_bytes = 8 * 1024 * 1024
    comm = CustomAllReduce(group, max_bytes=max_bytes)
    aiter = _Aiter(args.aiter_lib, comm, max_bytes) if args.aiter_lib else None
    runners = _runners(comm, aiter, args.tlx_blocks)
    algos = [a for a in args.algos.split(",") if a in runners] if args.algos else list(runners)
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    shapes = [(t, h) for t, h in shapes if t * h * 2 <= max_bytes]
    if args.stress:
        _stress(rank, world, comm, runners, algos, shapes, args.stress, group)
    else:
        _bench(rank, world, comm, runners, algos, shapes, args, group)
    dist.barrier(group=group)
    if aiter is not None:
        aiter.close()
    comm.close()
    dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aiter-lib", default=os.environ.get("AITER_AR_SHIM"), help="path to libaiter_ar_shim.so")
    p.add_argument("--tlx-blocks", type=lambda s: tuple(int(v) for v in s.split(",")),
                   default=(16, ),  # == gfx950.ONE_STAGE_BLOCKS (kept literal: no triton import in the parent process)
                   help="comma-separated TLX one-stage program counts to time")
    p.add_argument("--algos", default="", help="comma-separated subset of the available algorithms")
    p.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma-separated TOKENSxHIDDEN list")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--world-size", type=int, default=0, help="number of GPUs (default: all)")
    p.add_argument("--stress", type=int, default=0, help="run N random-shape calls per algorithm and check each")
    args = p.parse_args()
    world = args.world_size or torch.cuda.device_count()
    mp.spawn(_worker, args=(world, _free_port(), args), nprocs=world, join=True)


if __name__ == "__main__":
    main()
