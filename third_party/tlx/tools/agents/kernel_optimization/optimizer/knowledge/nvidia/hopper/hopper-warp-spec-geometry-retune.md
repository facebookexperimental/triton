# Hopper Warp-Spec Geometry Retune

Use this guidance only for NVIDIA Hopper (H100, sm_90) warp-spec kernels. It relieves single-wave SMEM starvation by shrinking geometry, and it runs BEFORE any pipeline-relief change.

## Recognize Single-Wave Starvation

All must hold: warp-spec producer/consumer kernel; NCU `Block_Limit_Shared_Mem == 1` AND Waves Per SM == 1.0; long-scoreboard the dominant (or near-dominant) CPI contributor, or high LG-throttle; low fraction of SOL. Skip when `Block_Limit_Shared_Mem >= 2` or the signature mismatches. Pipeline relief cannot help at BLSM=1 — every such change needs SMEM headroom or ≥2 CTAs/SM to interleave.

## Halve BLOCK_N Once

`BLOCK_N //= 2` at the launch site only; the kernel body does not change. Recompute `per_cta_smem = NUM_STAGES * (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * elem_bytes` plus ~16 KB barriers: at or below the 116 KB H100 pair-cap, `Block_Limit_Shared_Mem` lifts 1→2; above it, the win still lands via L2-thrash relief at BLSM=1. Either way expect Compute SM%, IPC, and Memory Pipes Busy to jump and CPI to drop — the win is per-CTA throughput, not occupancy (registers may still pin Theoretical Occ).

Floors (never violate): `BLOCK_N >= 64`; `BLOCK_M * BLOCK_N >= 8192` (one full WGMMA tile); `NUM_STAGES >= 2`. Prefer BN-shrink over stage-shrink: BN halves staging while keeping depth; stage-shrink kills overlap. When BN cannot shrink further without losing tensor-pipe efficiency, drop one stage only as a second lever (often a wash alone).

A second halving is not the default: attempt it only when post-first-halving Compute SM% is still well below the SOL band (residual SMEM-bandwidth pressure) AND the post-halving tile stays at or above 8192. Never loop "halve until BLSM ≥ 2". When Compute SM% reaches the band, stop shrinking and switch lever class (cooperative consumer with a BLOCK_M split, or a non-persistent grid).

Apply this retune BEFORE sizing TMA descriptors, which bake in BLOCK_N. Pingpong needs no BLSM ≥ 2: it alternates two consumer groups within one CTA, so BLSM=1 persistent pingpong is valid when the CTA-total SMEM/register allocation fits and profiling shows the targeted unit contention — gate on per-CTA fit plus measured contention instead, so canonical single-resident-block pingpong kernels are not excluded.

## Avoid The Retune Failure Modes

- Below-floor BN: tensor pipe collapses, wall regresses — floor BN at 64 / tile at 8192.
- `NUM_STAGES=1` to free SMEM: all async overlap gone — never below 2.
- Chasing BLSM via stage-shrink on persistent + register-bound kernels: the lift is occupancy-irrelevant and the lost stage hurts the producer margin — keep or bump depth instead.
- Retune applied after TMA adoption: descriptors sized for stale BLOCK_N — always retune first.
- Ungated double halving on persistent + register-bound kernels: occupancy frozen, WGMMA floor collapse, regression.

## Validate The Retune

Verify correctness at the kernel's standard dtype tolerance. Re-run NCU: Compute SM% must rise substantially whether or not BLSM lifts; confirm the win with a multi-trial median benchmark. Promotion requires a stable end-to-end win.
