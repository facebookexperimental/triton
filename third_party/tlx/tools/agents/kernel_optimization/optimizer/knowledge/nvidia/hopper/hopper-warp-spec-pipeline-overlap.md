# Hopper Warp-Spec Pipeline Overlap

Use this guidance only for NVIDIA Hopper (H100, sm_90) warp-spec kernels issuing two or more `tlx.async_dot` ops per iteration on distinct accumulators with elementwise work between them. It chains WGMMA execution with intermediate compute and staggers paired-load releases.

## Recognize The Chainable Shape

All must hold: two or more async dots per iteration writing DISTINCT accumulators; elementwise or reduction work between them that depends on the FIRST dot's result but not the second; both dots issued unconditionally every iteration. Skip single-accumulator GEMM (a single `wait(1)` suffices) and single-iteration loops (no steady state; take the synchronous fallback below).

## Chain The Dots

`async_dot_wait(N, t)` blocks until at most N dots remain in flight and returns the just-completed op's tensor. With dot1→`acc_a` (older) and dot2→`acc_b` (newer), the chain is a prologue/steady-state/epilogue skeleton — the peel is mandatory, not optional, because dot2 consumes the previous iteration's intermediate:

Prologue (the first iteration has no previous intermediate): issue dot1 alone, `acc_a = async_dot_wait(0, acc_a)`, run intermediate compute to seed. Never predicate the first dot2 to "skip" the prologue; the pending count must be iteration-invariant.

Steady state (remaining iterations):

1. Issue dot1, then dot2 (dot2 consumes the PREVIOUS iteration's intermediate).
2. `acc_a = async_dot_wait(1, acc_a)` — drains ONLY dot1; dot2 stays in flight.
3. Run intermediate compute on `acc_a` (overlaps dot2's WGMMA).
4. `acc_b = async_dot_wait(0, acc_b)` only where `acc_b` is consumed.
5. Arrive each input buffer's empties AFTER its matching wait.

Epilogue: issue the final dot2 on the last intermediate, drain every accumulator to `wait(0)`, then store.

Invariants: issue order is drain order — always wait on the OLDER tensor. `wait(K)` keeps K dots in flight. No conditional dots: mask values instead of predicating issues. SSA-rebind at every issue and wait. Re-apply the chain independently to every consumer-loop stage, with per-stage transforms as pure elementwise ops on `acc_a` between `wait(1)` and downstream compute. Single-iteration paths take a `tl.constexpr`-selected synchronous fallback — never a runtime-int branch, since the pending count cannot vary across paths.

## Release Paired Loads Asymmetrically

When the producer issues paired A/B loads per iteration but the consumer's matmuls have different lifetimes (A dies with the early matmul, B lives into the chained later matmul), release A immediately after `wait(1)` and hold B until `wait(0)`, so the next A load overlaps the current second matmul. Requires the chain above AND late-load latency comparable to or larger than early-load latency; TMA latency is hundreds of cycles, so the overlap is load-bearing.

Rules: separate A and B barrier sets (never share one barrier here); derive each empties barrier's `arrive_count` from the releasers of that specific slot — `NUM_MMA_GROUPS` for a shared operand read by every replica, `1` for a private per-replica ring or a pingpong-owned slot, where waiting for all groups blocks the producer (see the shared/private distinction in `hopper-warp-spec-consumer-patterns.md`); steady-state mm2 consumes B from iteration i−1; independent `a_phase`/`b_phase` counters (their flip points diverge); `NUM_BUFFERS >= 2`; peel the first iteration (no mm2 yet) and drain the final mm2 in the epilogue; replicate the asymmetric placement in EVERY stage — collapsing any stage to symmetric release silently restores baseline timing there.

Failure modes: symmetric release exposes A-load at every boundary (~10–20%); current-iteration B index hangs or corrupts (mm2 must match the iteration that produced its intermediate); shared A/B phase corrupts flips; missing prologue or epilogue hangs at the ends.

## Avoid The Chain Failure Modes

- `wait(0)` after each dot: serialized WGMMA, no overlap.
- `wait(1)` on the newer tensor: wrong op returned, silent corruption.
- Arrive before the matching wait: the producer overwrites SMEM under a live WGMMA read.
- Conditional dots: pending-count divergence, hangs or stale results.
- Missing epilogue `wait(0)`: in-flight dot at store time, garbage output.
- Reordered dots (dot2 first): data deadlock — dot2 needs this iteration's intermediate.

## Validate The Overlap

Verify correctness across stages, boundary and single-iteration paths, and repeated launches. Confirm via warp-granularity Proton attribution or NCU that intermediate compute overlaps the second WGMMA and that A-load latency hides behind mm2. Promotion requires a stable end-to-end win.
