# TLX Unified Persistent Flash-Attention (causal + non-causal)

A single TLX Flash-Attention forward kernel for **gfx950 (MI350)** that runs both
causal and non-causal attention and beats the per-mode baselines in every config.
Lives in `third_party/tlx/tutorials/amd-fa-pipelined_test.py` as the `persistent`
mode (`flash_attn_persistent` → `_attn_fwd_persistent`).

## How to run

```bash
source ~/nod/tlx/venv/tlx_venv/bin/activate

# Benchmark (TFLOPS) — causal and non-causal
python third_party/tlx/tutorials/amd-fa-pipelined_test.py \
    -b 1 -hq 64 -sq 8192 16384 -d 64 128 -causal true false \
    --kernel persistent async_prefetch

# Correctness (all N, both modes)
pytest third_party/tlx/tutorials/amd-fa-pipelined_test.py::test_fa_correctness \
    -k persistent -s --tb=short
```

## Kernels

| Registry key | Host wrapper | JIT kernel | Description |
|---|---|---|---|
| `async_simple` | `flash_attn_async_simple` | `_attn_fwd_async_simple` | Simple async-DMA reference |
| `async_prefetch` | `flash_attn_async_prefetch` | `_attn_fwd_async_prefetch` | Double-buffered prefetch reference (causal + non-causal) |
| `persistent` | `flash_attn_persistent` | `_attn_fwd_persistent` | **Unified persistent fold-bundling kernel (default champion)** |

## Performance (gfx950, B=1, H=64, bf16)

TFLOPS, higher is better. `persistent` beats the prefetch baseline in **both**
modes and every config:

| Config | mode | `persistent` | `async_prefetch` | Torch SDPA |
|---|---|---|---|---|
| D=64,  N=8192  | causal | **694.5** | 351.0 | 324.3 |
| D=64,  N=16384 | causal | **737.0** | 434.6 | 392.9 |
| D=128, N=8192  | causal | **801.8** | 478.1 | 415.8 |
| D=128, N=16384 | causal | **855.2** | 548.6 | 464.1 |
| D=64,  N=8192  | non-c. | **735.7** | 713.7 | 559.4 |
| D=64,  N=16384 | non-c. | **753.6** | 737.8 | 602.2 |
| D=128, N=8192  | non-c. | **867.2** | 838.9 | 601.3 |
| D=128, N=16384 | non-c. | **888.9** | 865.4 | 657.6 |

(The baseline `async_prefetch` applies the causal mask in every inner iteration
and uses a plain 2-D grid; `persistent` peels the mask, balances the causal
triangle, and groups heads per XCD — hence the large causal gap and the +2–3.5%
non-causal edge.)

## Design

`_attn_fwd_persistent` synthesizes four ideas into one clean scheduler.

1. **Peeled mask (one tile fn for both modes).** `_attn_tile(IS_CAUSAL)` splits a
   tile's K range into an unmasked steady-state loop (FMA-friendly softmax, no
   `tl.where`) and a short masked tail. Causal masks only the `BLOCK_M//BLOCK_N`
   diagonal blocks (blocks fully below the diagonal need no mask); non-causal
   sets `hi = N_CTX` so the steady loop covers everything and the tail handles
   only the ragged boundary. The async double-buffered K/V prefetch chain is
   continuous across both loops (slot = `block_idx % 2`), so there is no bubble.
   K is transposed at the memdesc level (`local_trans`) to land directly in
   dot-operand layout.

2. **Constant-cost fold bundling (the load balancer).** A causal tile `pid_m`
   costs `pid_m+1` K-blocks, so the last wave of a naive schedule is all-heavy
   tiles. We walk a head's m-tiles in **ping-pong (fold) order**
   `0, N-1, 1, N-2, …`, interleaving the lightest and heaviest remaining tile.
   Because adjacent tiles in this order sum to a constant cost (`≈ N_M+1`
   K-blocks), *every* scheduling unit does equal work — eliminating the causal
   tail-wave imbalance. This is the principle behind classic "mirror pairing",
   stated generally.

3. **`TILES_PER_UNIT` knob (not hardcoded to 2).** Each unit is `TILES_PER_UNIT`
   consecutive ping-pong tiles, run back-to-back so cost-per-iteration stays
   constant (fixed prologue/epilogue overhead stays amortized — no overhead-bound
   "all-light" tail). Causal default = **2** (the minimal constant-cost bundle ⟺
   mirror pairing); non-causal default = **1** (tiles are already equal-cost).
   The knob is a true generalization — see the sweep below.

4. **Persistent + XCD L2 remap (occupancy + locality).** Launches `NUM_SMS`
   resident programs. gfx950 has 8 XCDs and the HW pins program `pid` to XCD
   `pid % NUM_XCDS`, so heads are pinned to XCDs (`hz % NUM_XCDS`) — each XCD's
   K/V stays L2-resident. Units are flattened `(head_on_xcd, bundle)` and
   round-robin strided across the XCD's `NUM_LOCAL = NUM_SMS/NUM_XCDS` programs;
   constant-cost units make plain striding balance. A program runs a *variable*
   number of tiles (`units/NUM_LOCAL × TILES_PER_UNIT`), not a fixed 2 — which is
   why this is "persistent general", not static mirror.

### Why 2 (causal) is optimal, not arbitrary

A program's runtime ≈ (K-block compute) + (per-tile fixed overhead × tile count)
— two independent cost dimensions. Bundling one heavy tile with one light tile is
the minimal construction that makes every unit identical in **both** dimensions
(constant block sum *and* constant tile count). The `TILES_PER_UNIT` sweep
confirms it (D=128, N=16384, causal):

| TILES_PER_UNIT | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| TFLOPS | 566 | **855** | 783 | 784 |

`=1` is the no-bundle regime (imbalanced + overhead-bound tail); `=2` is the
optimum; `≥4` over-coarsens (fewer schedulable units than programs → idle CUs).

## Notes

- **Block sizes**: `BLOCK_M=256`; `BLOCK_N=128` for `D<=64` (more compute per LDS
  barrier now that the mask is peeled), `BLOCK_N=64` for `D=128` (LDS budget for
  double-buffered K+V).
- **Scope**: `N % BLOCK_N == 0`; partial-block N falls back to the baseline
  `async_prefetch` (correct, not perf-critical) to avoid a modulo-decode
  `iota_range` compiler crash.
- **Profile** (AMDGCN, `~/.triton/cache`): vgpr_count=317, 0 vgpr/sgpr spills,
  num_warps=4.

## Future work

- Extend to GQA, ragged batch, and single-token decode (`BLOCK_M=1`); for that
  low-parallelism regime a StreamK-style K-block split + partial-softmax
  reduction beats whole-tile bundling.
- Greedy-bundle (accumulate tiles to a target cost) for non-linear cost profiles
  (e.g. sliding-window) — the `TILES_PER_UNIT` bundle generalizes to that.
- Fix the `iota_range` modulo-decode crash upstream so partial-N needs no fallback.
