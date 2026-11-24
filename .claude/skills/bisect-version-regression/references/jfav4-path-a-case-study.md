# Case Study (Path A): jfav4 Triton 3.5 to 3.8 Regression

Condensed from the 2026-08-27 investigation of the `tlx_jfav4_beta`
jagged flash-attention kernel (`_attn_fwd_ws`, forward, load balancing
enabled, representative input `event_9000_pma`) on a B200. This is the
example where tritonparse's built-in bisect was the right tool.

## Why Path A applied

The decision gate passed on every condition:

1. Both endpoints, and the whole range between them, were commits in the
   facebookexperimental/triton GitHub history.
2. A mini-repro extracted from a training-path trace reproduced the
   regression deterministically (the bad endpoint was roughly a
   quarter slower than the good one), so it worked directly as a
   bisect oracle.
3. A 221-commit subrange was found in which the LLVM pin was completely
   constant (`f6ded0be897e2878612dd903f7e8bb85448269e5`), so the bisect
   was one-dimensional: Triton commit only, its own LLVM pin untouched.
   Fixed ptxas (CUDA 12.8.93) across all measurements.

## What happened

After stage localization put the difference below TTGIR, `git bisect
run` over the 221-commit same-LLVM interval converged in 7 measurement
points to the first bad commit:

| Role | Triton commit | Oracle result |
| --- | --- | --- |
| Known good endpoint | `517ce79ccf28` | good |
| First bad commit's parent | `44ce51850af7` | good |
| First bad commit | `a4eafa3c6de3` | bad |
| Known bad endpoint | `e00f766380da` | bad |

First bad commit: `a4eafa3c6` ("[AutoWS] Fix thread ID and barrier
handling for warp specialization (#861)", D94327565). It carried two
logical hunks: extending a warp-specialized barrier's shared-memory
allocation lifetime, and computing the thread-ID mask in `getThreadId()`
from `ttg.total-num-warps`.

## Hunk isolation (still required after a Path A hit)

| Variant | Verdict |
| --- | --- |
| Parent | good |
| Barrier lifetime only | good |
| Thread-ID mask only | bad |
| Full culprit commit | bad |

Parent and barrier-only produced byte-identical LLIR/PTX/cubin hashes;
full and thread-ID-only were also byte-identical to each other. The
regression was the `getThreadId()` hunk alone: the mask upper bound grew
from 128 threads (4 warps) to 511 (16 total warps), and under the same
ptxas the stack frame and spill store/load traffic grew several-fold.

## Lessons this case contributes to the skill

- Shrink to a same-LLVM subrange before bisecting; do not pin an
  arbitrary Triton commit against an arbitrary LLVM. Build/API
  incompatibility inside the range is a skip (125), never good or bad.
- A pass-level cliff found earlier (NVPTX MachineCSE at opt-bisect-limit
  51) was a downstream trigger, not the commit-level culprit. Both were
  true; they answer different questions, and the commit-level answer
  still needed hunk isolation inside the culprit commit.
- Tooling friction is normal (a missing packaged bisect script, an
  offline-build misconfiguration). Fix it, reset any half-run bisect,
  and only trust a clean `git bisect run` to completion.
