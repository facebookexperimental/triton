"""Eight-rank timing of ``tlx.ops.all_reduce``.

The single-process harness cannot drive a collective: ``supported()`` is false there, and running this file forks
one process per GPU instead. Every side is the
replay of a graph of 20 calls, timed per rank and reduced by max over ranks, median over rounds. Before each replay
the ranks queue a 2 ms sleep, so the graph launch is off the clock, then run one aligning call, so they start
together: the times exclude rank skew. Columns, in us per call:

- ``tlx``: ``tlx.ops.all_reduce``; ``rccl``: ``torch.distributed.all_reduce`` on the same tensor.

    python python/test/tlx_benchmark/bench_all_reduce.py [--suite gfx950_1] [--world 8]
"""

from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import pathlib
import statistics
import sys
import tempfile
import traceback

from triton.tlx.ops.kernels.collectives.all_reduce._shapes import FOCUS as SHAPE_SUITES
from triton.tlx.ops.kernels.collectives.all_reduce._shapes import SYNTHETIC, AllReduceShape, label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import Case, driver  # noqa: E402

OP = "all_reduce"
REF_NAME = "torch.distributed.all_reduce (RCCL), timed by this file's multi-rank main"
EXTRA_COLUMNS = ()


def cases(synthetic: bool = False, suites=None) -> list[Case]:
    shapes = SYNTHETIC if synthetic else SHAPE_SUITES.shapes(driver.arch(), suites)
    return [Case(op=OP, arch=driver.arch(), dtype="bfloat16", shape=tuple(s), label=label(s)) for s in shapes]


def prepare(case: Case, space: str):
    raise NotImplementedError(f"{OP} is a collective; run {pathlib.Path(__file__).name} directly "
                              f"({label(AllReduceShape(*case.shape))})")


def _time(fn, iters=20, rounds=15):
    import torch
    import torch.distributed as dist

    for _ in range(3):
        fn()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(4_000_000)
        fn()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        t = torch.tensor(start.elapsed_time(end) * 1e3 / iters, device="cuda", dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        samples.append(t.item())
    return statistics.median(samples)


def _worker(rank, world, init_file, suites, queue):
    try:
        import torch
        import torch.distributed as dist
        from triton.tlx import ops
        from triton.tlx.ops.kernels.collectives.all_reduce._shapes import partial

        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world,
                                device_id=torch.device("cuda", rank), timeout=datetime.timedelta(minutes=10))
        shapes = SHAPE_SUITES.shapes("gfx950", suites)
        workspace = ops.all_reduce_workspace(max(s.tokens * s.H for s in shapes))
        if rank == 0:
            print(f"{'shape':<34}{'tlx':>9}{'rccl':>9}", flush=True)
        for shape in shapes:
            x = workspace.input(shape.tokens, shape.H)
            x.copy_(partial(shape, rank))
            out = torch.empty_like(x)
            rccl_in = x.clone()
            row = (
                _time(lambda: ops.all_reduce(x, workspace, out=out)),
                _time(lambda: dist.all_reduce(rccl_in)),
            )
            if rank == 0:
                print(f"{label(shape):<34}" + "".join(f"{t:9.2f}" for t in row), flush=True)
        workspace.close()
        dist.barrier()
        dist.destroy_process_group()
        queue.put((rank, None))
    except Exception:
        queue.put((rank, traceback.format_exc()))


def supported() -> bool:
    return False


_, default_json, run, _ = driver.bind(sys.modules[__name__])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", action="append", help="focus suite; repeat to combine (default: gfx950_all)")
    parser.add_argument("--world", type=int, default=8)
    args = parser.parse_args()
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    init_file = tempfile.mktemp(prefix="tlx_all_reduce_")
    procs = [ctx.Process(target=_worker, args=(r, args.world, init_file, args.suite, queue)) for r in range(args.world)]
    for proc in procs:
        proc.start()
    failed = False
    for _ in procs:
        rank, error = queue.get()
        if error:
            failed = True
            print(f"rank {rank} failed:\n{error}", flush=True)
    for proc in procs:
        proc.join()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
