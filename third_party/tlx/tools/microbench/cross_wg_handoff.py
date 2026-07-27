"""B200 microbenchmark: cross-warp-group hand-off costs for the modulo
scheduler's partitioner cost model.

Calibrates the second-order terms in ModuloSchedulePass.cpp's
scoreCandidate (see docs/SolverMigrationNotes.md "Suggested sequencing"):
  - kCrossWGRoundTripLatency: fixed cost of a reg->SMEM->reg hand-off
    between two warp groups (store visibility + mbarrier arrive + waiter
    wake-up), previously back-fitted from a single FA A/B, never measured.
  - smemMoveCost: the size-dependent part of the same hand-off.
  - kBarrierOverhead sanity check: mbarrier arrive issue cost.
  - named-barrier round-trip (TC->CUDA cross-WG edges use NAMED kind).

All kernels run ONE CTA with two 4-warp warp groups (the default task and
one explicit async_task) and count SM cycles with tlx.clock64, so results
are clock-frequency independent. Each ping-pong iteration contains two
one-way hand-offs; one-way = cycles/iter/2.

The store->arrive sequence is byte-identical to what the sched2tlx emitter
produces for synthesized register channels (e.g. case3_FA generated.py:
local_store(L0_smem_4[0], exp2_26); barrier_arrive(sem2_b4_full[0], 1)),
so this measures the same mechanism the cost model prices.

Usage (B200 node, sanitized env — see repo memory/uv setup):
  env -u LD_LIBRARY_PATH python third_party/tlx/tools/microbench/cross_wg_handoff.py
"""

import argparse
import statistics

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


# ── kernels ─────────────────────────────────────────────────────────────────


@triton.jit
def k_signal_pingpong(out_ptr, NITER: tl.constexpr):
    """Pure mbarrier ping-pong, no data: 2 x (arrive + wake) per iter."""
    bar_a = tlx.alloc_barriers(num_barriers=1, arrive_count=1)  # A -> B
    bar_b = tlx.alloc_barriers(num_barriers=1, arrive_count=1)  # B -> A
    with tlx.async_tasks():
        with tlx.async_task("default"):
            tid = tlx.thread_id(0)
            t0 = tlx.clock64()
            for i in range(NITER):
                tlx.barrier_arrive(bar_a[0], 1)
                tlx.barrier_wait(bar_b[0], i & 1)
            t1 = tlx.clock64()
            if tid == 0:
                tl.store(out_ptr, t1 - t0)
        with tlx.async_task(num_warps=4):
            for i in range(NITER):
                tlx.barrier_wait(bar_a[0], i & 1)
                tlx.barrier_arrive(bar_b[0], 1)


@triton.jit
def k_data_pingpong(out_ptr, res_ptr, NITER: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    """reg->SMEM->reg ping-pong: 2 x (store + arrive + wake + load) per iter.

    The A->B->A value dependence (v -> buf_a -> w -> buf_b -> v) makes the
    chain serial; nothing overlaps across iterations.
    """
    buf_a = tlx.local_alloc((ROWS, COLS), tl.float32, tl.constexpr(1))
    buf_b = tlx.local_alloc((ROWS, COLS), tl.float32, tl.constexpr(1))
    full_a = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    full_b = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            tid = tlx.thread_id(0)
            v = tl.full((ROWS, COLS), 1.0, tl.float32)
            t0 = tlx.clock64()
            for i in range(NITER):
                tlx.local_store(buf_a[0], v)
                tlx.barrier_arrive(full_a[0], 1)
                tlx.barrier_wait(full_b[0], i & 1)
                v = tlx.local_load(buf_b[0])
            t1 = tlx.clock64()
            if tid == 0:
                tl.store(out_ptr, t1 - t0)
            # Consume v so the load/store chain is not dead. B increments by 1
            # per hand-off, so v ends at exactly 1 + NITER in every element —
            # a stale/skipped hand-off shows up as a lower value (the +1 add
            # costs ~1 cyc on B's side, negligible vs the handshake).
            s = tl.sum(tl.sum(v, axis=1), axis=0)
            if tid == 0:
                tl.store(res_ptr, s)
        with tlx.async_task(num_warps=4):
            for i in range(NITER):
                tlx.barrier_wait(full_a[0], i & 1)
                w = tlx.local_load(buf_a[0])
                tlx.local_store(buf_b[0], w + 1.0)
                tlx.barrier_arrive(full_b[0], 1)


@triton.jit
def k_named_pingpong(out_ptr, NITER: tl.constexpr):
    """Named-barrier ping-pong across two 4-warp groups (256 threads).

    bar.arrive counts arrivals only; bar.sync (wait) arrives and blocks, so
    each barrier completes at 128 (arrive side) + 128 (sync side) = 256.
    IDs 9/10 match the ping-pong scheduling convention in the TLX tutorials.
    """
    with tlx.async_tasks():
        with tlx.async_task("default"):
            tid = tlx.thread_id(0)
            t0 = tlx.clock64()
            for _ in range(NITER):
                tlx.named_barrier_arrive(9, 256)
                tlx.named_barrier_wait(10, 256)
            t1 = tlx.clock64()
            if tid == 0:
                tl.store(out_ptr, t1 - t0)
        with tlx.async_task(num_warps=4):
            for _ in range(NITER):
                tlx.named_barrier_wait(9, 256)
                tlx.named_barrier_arrive(10, 256)


@triton.jit
def k_arrive_issue(out_ptr, NITER: tl.constexpr):
    """mbarrier arrive issue cost. arrive_count = NITER + 1 so the loop's
    arrivals never flip the phase; the final arrive + wait after the loop
    completes phase 0, which orders the closing clock64 read after all
    arrivals (without the wait, SASS reorders %clock64 across the
    dependency-free loop and measures 0)."""
    bar = tlx.alloc_barriers(num_barriers=1, arrive_count=NITER + 1)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            tid = tlx.thread_id(0)
            t0 = tlx.clock64()
            for _ in range(NITER):
                tlx.barrier_arrive(bar[0], 1)
            tlx.barrier_arrive(bar[0], 1)
            tlx.barrier_wait(bar[0], 0)
            t1 = tlx.clock64()
            if tid == 0:
                tl.store(out_ptr, t1 - t0)
        with tlx.async_task(num_warps=4):
            tid = tlx.thread_id(0)
            if tid == 512:  # never true; keeps the task non-empty
                tl.store(out_ptr, 0)


# ── driver ──────────────────────────────────────────────────────────────────

NITER = 2048
REPS = 7
WARMUP = 3


def _run(kernel, pre_args, niter=NITER, post_args=()):
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    full_args = [out] + list(pre_args) + [niter] + list(post_args)
    for _ in range(WARMUP):
        kernel[(1,)](*full_args, num_warps=4)
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPS):
        out.zero_()
        kernel[(1,)](*full_args, num_warps=4)
        torch.cuda.synchronize()
        samples.append(out.item() / niter)
    return statistics.median(samples), min(samples), max(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=NITER)
    args = parser.parse_args()
    niter = args.niter

    assert torch.cuda.get_device_capability()[0] >= 10, "B200 (sm_100) required"
    print(f"device: {torch.cuda.get_device_name()}, niter={niter}, reps={REPS} (median [min..max])")

    med, lo, hi = _run(k_signal_pingpong, [], niter)
    signal_oneway = med / 2
    print(f"\nsignal-only mbarrier ping-pong : {med:8.1f} cyc/iter [{lo:.1f}..{hi:.1f}]"
          f" -> one-way handshake = {signal_oneway:.1f} cyc")

    med, lo, hi = _run(k_named_pingpong, [], niter)
    print(f"named-barrier ping-pong (256t) : {med:8.1f} cyc/iter [{lo:.1f}..{hi:.1f}]"
          f" -> one-way = {med / 2:.1f} cyc")

    med, lo, hi = _run(k_arrive_issue, [], niter)
    print(f"mbarrier arrive issue loop     : {med:8.1f} cyc/arrive [{lo:.1f}..{hi:.1f}]")

    print(f"\n{'shape':>12} {'bytes':>8} {'elems':>7} {'cyc/iter':>9} {'one-way':>8} {'-handshake':>10}")
    sizes = [(128, 4), (128, 16), (128, 32), (128, 64), (128, 128)]
    points = []
    for rows, cols in sizes:
        res = torch.zeros(1, dtype=torch.float32, device="cuda")
        med, lo, hi = _run(k_data_pingpong, [res], niter, post_args=(rows, cols))
        # v ends at exactly 1 + niter per element. Tolerance 0.5*elems: less
        # than one whole-tile missed hand-off, safely above fp32 tree-sum
        # rounding at these magnitudes.
        elems = rows * cols
        expected = float(elems * (niter + 1))
        ok = abs(res.item() - expected) < 0.5 * elems
        oneway = med / 2
        nbytes = elems * 4
        points.append((elems, nbytes, oneway))
        print(f"{rows:>5}x{cols:<6} {nbytes:>8} {elems:>7} {med:>9.1f} {oneway:>8.1f}"
              f" {oneway - signal_oneway:>10.1f}  {'OK' if ok else 'HANDOFF BROKEN: res=%s' % res.item()}")

    # Least-squares fit: oneway(bytes) = H + k * bytes
    n = len(points)
    sx = sum(p[1] for p in points)
    sy = sum(p[2] for p in points)
    sxx = sum(p[1] * p[1] for p in points)
    sxy = sum(p[1] * p[2] for p in points)
    k = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    h = (sy - k * sx) / n

    print("\nfit: one-way data hand-off = H + k*bytes")
    print(f"  H (fixed)                  = {h:.0f} cyc")
    print(f"  k (slope)                  = {k * 1024:.1f} cyc/KB")
    print("\nmodel mapping (per-iteration charge for a depth-1 register channel,")
    print("consumer side charge = kCrossWGRoundTripLatency + smemMoveCost(bytes)):")
    print(f"  suggested kCrossWGRoundTripLatency ~= H + signal one-way = {h + signal_oneway:.0f} cyc")
    print("  (H = data-direction handshake; + signal one-way for the empty-barrier return)")
    print(f"  suggested smemMoveCost(bytes)      ~= {k * 1024:.1f} * bytes/1024 cyc")
    print("note: every cyc/iter above includes ~6 cyc of loop overhead (induction,")
    print("phase compute, branch), so the fixed terms are slight upper bounds.")


if __name__ == "__main__":
    main()
