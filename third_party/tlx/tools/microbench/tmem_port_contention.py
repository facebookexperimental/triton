"""B200 microbenchmark: TMEM ld/st round-trip latency and cross-warp-group
port contention, for the modulo scheduler's shared-engine model.

Motivated by the Route A sub-tiling post-mortem (see
ModuloScheduling/docs/SubTilingDesign.md "Experiment log"): the generated
sub-tiled FA kernel runs its two softmax chains in phase, each doing a full
128x128 f32 TMEM round-trip (acc rescale) per iteration, and measures 11.3K
cycles/iteration against a modeled II of 2487. The model prices tmem_load/
tmem_store as per-WG fixed-latency ops with no shared engine. This bench
answers, with numbers:

  1. What does a serial TMEM round-trip (tcgen05.ld -> fma -> tcgen05.st)
     cost from one 4-warp group, as a function of tile columns?
  2. Do TWO warp groups doing concurrent round-trips on DISJOINT TMEM
     buffers serialize on the per-SM TMEM port (dual ~= 2x solo), or
     overlap (dual ~= solo)?

One CTA, two 4-warp groups (default task + one async_task), cycles via
tlx.clock64. The load->fma->store->load chain is serial within a WG by
value dependence; across WGs the buffers are disjoint so any slowdown in
dual mode is engine contention, not synchronization. In dual mode both
groups rendezvous on a named barrier immediately before the timed loop so
the measurement windows coincide; both groups' cycles are reported and
should agree within noise.

Usage (B200 node, sanitized env — see repo memory/uv setup):
  env -u LD_LIBRARY_PATH python third_party/tlx/tools/microbench/tmem_port_contention.py
"""

import argparse
import statistics

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def k_tmem_roundtrip(out_ptr, res_ptr, NITER: tl.constexpr, COLS: tl.constexpr, DUAL: tl.constexpr):
    """Serial TMEM round-trip loop in WG0; WG1 runs the same loop on its own
    buffer iff DUAL. v = v + 1 keeps the chain serial and makes the final
    value exact (elems * (init + NITER)) for a hand-off check."""
    acc0 = tlx.local_alloc((128, COLS), tl.float32, 1, tlx.storage_kind.tmem)
    acc1 = tlx.local_alloc((128, COLS), tl.float32, 1, tlx.storage_kind.tmem)
    with tlx.async_tasks():
        with tlx.async_task("default"):
            tid = tlx.thread_id(0)
            v = tl.full((128, COLS), 1.0, tl.float32)
            tlx.local_store(acc0[0], v)
            if DUAL:
                tlx.named_barrier_wait(11, 256)
            t0 = tlx.clock64()
            for _ in range(NITER):
                x = tlx.local_load(acc0[0])
                tlx.local_store(acc0[0], x + 1.0)
            t1 = tlx.clock64()
            if tid == 0:
                tl.store(out_ptr, t1 - t0)
            fin = tlx.local_load(acc0[0])
            row = tl.sum(fin, axis=1)
            tl.store(res_ptr + tl.arange(0, 128), row)
        with tlx.async_task(num_warps=4):
            tid = tlx.thread_id(0)
            if DUAL:
                w = tl.full((128, COLS), 1.0, tl.float32)
                tlx.local_store(acc1[0], w)
                tlx.named_barrier_wait(11, 256)
                t0 = tlx.clock64()
                for _ in range(NITER):
                    y = tlx.local_load(acc1[0])
                    tlx.local_store(acc1[0], y + 1.0)
                t1 = tlx.clock64()
                if tid == 128:
                    tl.store(out_ptr + 1, t1 - t0)
                fin = tlx.local_load(acc1[0])
                row = tl.sum(fin, axis=1)
                tl.store(res_ptr + 128 + tl.arange(0, 128), row)
            else:
                if tid == 4096:  # never true; keeps the task non-empty
                    tl.store(out_ptr + 1, 0)


NITER = 2048
REPS = 7
WARMUP = 3


def _run(cols, dual, niter):
    out = torch.zeros(2, dtype=torch.int64, device="cuda")
    res = torch.zeros(256, dtype=torch.float32, device="cuda")
    args = [out, res, niter, cols, dual]
    for _ in range(WARMUP):
        k_tmem_roundtrip[(1, )](*args, num_warps=4)
    torch.cuda.synchronize()
    s0, s1 = [], []
    for _ in range(REPS):
        out.zero_()
        k_tmem_roundtrip[(1, )](*args, num_warps=4)
        torch.cuda.synchronize()
        s0.append(out[0].item() / niter)
        if dual:
            s1.append(out[1].item() / niter)
    elems = 128 * cols
    expected = float(elems * (1 + niter))
    ok = abs(res[:128].sum().item() - expected) < 0.5 * elems
    if dual:
        ok = ok and abs(res[128:].sum().item() - expected) < 0.5 * elems
    med0 = statistics.median(s0)
    med1 = statistics.median(s1) if dual else 0.0
    return med0, med1, min(s0), max(s0), ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=NITER)
    args = parser.parse_args()
    niter = args.niter

    assert torch.cuda.get_device_capability()[0] >= 10, "B200 (sm_100) required"
    print(f"device: {torch.cuda.get_device_name()}, niter={niter}, reps={REPS} (median [min..max])")
    print(f"\n{'shape':>12} {'bytes':>8} {'solo cyc/rt':>12} {'dual wg0':>10} {'dual wg1':>10} {'dual/solo':>10}")

    for cols in (32, 64, 128):
        solo, _, lo, hi, ok_s = _run(cols, 0, niter)
        d0, d1, dlo, dhi, ok_d = _run(cols, 1, niter)
        nbytes = 128 * cols * 4
        ratio = d0 / solo
        flag = "" if (ok_s and ok_d) else "  CHAIN BROKEN"
        print(f"{'128x%d' % cols:>12} {nbytes:>8} {solo:>12.1f} {d0:>10.1f} {d1:>10.1f} {ratio:>10.2f}{flag}")

    print("\ninterpretation: dual/solo ~= 2.0 means the TMEM ld/st path is a")
    print("per-SM shared engine (model must reserve it across WGs); ~= 1.0")
    print("means per-WG throughput and the sub-tiling slowdown lies elsewhere.")
    print("solo cyc/rt at 128x128 is the per-iteration charge the model should")
    print("put INSIDE any recurrence that contains an unsliced acc rescale.")


if __name__ == "__main__":
    main()
