# Debugging AutoWS Accuracy Issues

This skill describes a systematic workflow for debugging accuracy issues in
Triton autoWS-compiled kernels, using the BWD persistent flash attention
kernel (`_attn_bwd_persist`) as the running example. The kernel has a known
TLX reference implementation (`_attn_bwd_ws`) that produces correct results,
providing a ground truth for comparison.

## Prerequisites

- A test harness that runs both the autoWS and TLX kernels with identical
  inputs and compares outputs (e.g., `test_tlx_bwd_from_fused_attention.py`)
- Pre-generated TTGIR and PTX files in directories organized by configuration:
  - `b64/` — autoWS kernel compiled with BLOCK_M=64 (failing)
  - `b128/` — autoWS kernel compiled with BLOCK_M=128 (working reference)
  - `tlx-b64/` — TLX kernel compiled with BLOCK_M=64 (working reference)
- The conda environment with torch and triton (`conda run -n metamain2`)

## Option 1: Characterize the Error Pattern via Test Harness

**When to use**: First step for any accuracy issue. Determines whether the
error is zeros (race/zeroing bug), systematic scaling, per-tile, or random.

### Step 1: Add element-wise diagnostics to the test script

Add a `_debug_dv()` function (or similar for the failing gradient) that
reports:

1. **Zero fraction**: What percentage of elements are zero in the failing
   output vs reference? If >50% are zero → zeroing/race bug (like Bug 3).
   If 0% → stale accumulation or computation error.

2. **Non-zero element accuracy**: For non-zero elements, compute
   `ratio = orig/ref` statistics (mean, median, std). If ratio ≈ 1.0 with
   small std → small perturbation. If ratio varies wildly → data corruption.

3. **Per N-tile pattern**: For each BLOCK_N=128 tile along the N dimension,
   show zero fraction and max error. Identifies which outer loop tiles are
   corrupted.

4. **Per batch-head classification**: Classify each `(z, h)` as all-zero,
   partial/wrong, or fully-correct. Shows if corruption is per-CTA.

5. **Per-head N-tile heatmap**: Compact visualization showing `.` (ok),
   `X` (fail), `Z` (all-zero) for each N-tile of each head.

6. **Sample bad elements**: Print first 10 elements where `|orig-ref| > atol`
   with indices, values, and ratio.

### Step 2: Persistent grid CTA analysis

For persistent kernels, map each tile to its CTA and outer loop iteration:

```python
total_tiles = n_tiles * Z * H
num_progs = min(NUM_SMS, total_tiles)
for prog_id in range(num_progs):
    tile_idx = prog_id
    while tile_idx < total_tiles:
        bhid = tile_idx // n_tiles
        pid = tile_idx % n_tiles
        # check error for this tile
        tile_idx += num_progs
```

Report:
- **Iteration 0 vs 1+**: If iteration 0 always passes and iteration 1+
  fails → persistent outer loop race (accumulator not properly reset between
  tiles).
- **Failure rate**: What fraction of subsequent tiles fail? If ~18% →
  probabilistic race. If 100% → deterministic bug.

### Step 3: Scale testing

Run multiple configs to find the threshold:
- Small config (e.g., Z=2,H=4): ≤1 tile per CTA → no outer loop
- Medium config (e.g., Z=4,H=8): ~2 tiles per CTA
- Large config (e.g., Z=8,H=16): ~7 tiles per CTA
- Run borderline configs 3× to check if failures are deterministic or
  probabilistic

Key insight: If the bug is probabilistic (FAIL/PASS/FAIL across runs at
the same config), it's a genuine race condition, not a logic error.

### Example command

```bash
cd /data/users/mren/MetaMain2/triton
conda run -n metamain2 bash -c \
  'CUDA_VISIBLE_DEVICES=1 TRITON_USE_META_PARTITION=1 \
   TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1 \
   python python/tutorials/test_tlx_bwd_from_fused_attention.py'
```

---

## Option 2: Compare TTGIR and PTX Files

**When to use**: After the error pattern is characterized and you need to
find the structural cause. Three sets of files provide a triangulation:

| Directory | Kernel | Status |
|-----------|--------|--------|
| `tlx-b64/` | TLX with BLOCK_M=64 | ✅ Correct |
| `b128/` | AutoWS with BLOCK_M=128 | ✅ Correct |
| `b64/` | AutoWS with BLOCK_M=64 | ❌ Failing |

### TTGIR comparison (check first)

TTGIR is the highest-level IR that shows the warp_specialize structure,
barrier placement, and TMEM lifecycle. Compare:

1. **Partition structure**: How many partitions? Which task IDs? What
   operations are in each partition?

2. **TMEM lifecycle for the failing accumulator** (e.g., dv):
   - Where is it zeroed? (tmem_store of zeros)
   - Where is it accumulated? (tc_gen5_mma with useC)
   - Where is it read? (tmem_load)
   - Which partitions perform each step?

3. **Barrier/token flow**: For each TMEM buffer, trace:
   - init_barrier → wait_barrier → arrive_barrier chains
   - Which operand positions in warp_specialize they map to
   - Whether the same barrier is waited on and arrived by the correct
     partitions

4. **Key differences to look for**:
   - **Extra tmem_store of zeros**: Does autoWS have `tmem_store dense<0.0>`
     that TLX doesn't? This creates cross-partition zeroing races.
   - **Different partition assignments**: Is an operation in a different
     task ID between TLX and autoWS?
   - **Missing or extra barriers**: Does one version have more
     wait_barrier/arrive_barrier calls than the other?
   - **useC flag differences**: Does the MMA use `useC=false` (zero-init)
     vs `useC=true` (accumulate) differently?

### PTX comparison (check second)

PTX reveals the physical implementation of barriers and TMEM addressing.
Compare:

1. **mbarrier.init counts**: All barriers should be initialized with
   `count=1`. If count > 1, a single arrive won't advance the phase.

2. **SMEM offsets for barriers**: Extract all `mbarrier.init` and
   `mbarrier.try_wait`/`mbarrier.arrive` instructions. Verify that the
   SMEM offsets match between partitions (the same physical barrier is
   waited on and arrived).

   ```bash
   # Find all mbarrier.init with SMEM offsets
   grep -n 'mbarrier.init' file.ptx
   # Find all try_wait/arrive near zeroing stores
   grep -n 'mbarrier.try_wait\|mbarrier.arrive' file.ptx
   ```

3. **tcgen05.st format**: Compare the store format for TMEM operations:
   - `x16.b32` vs `x32.b32` (different column counts per warp)
   - The number of registers (16 vs 32)
   - The TMEM address register and offset

4. **tcgen05.mma operand addresses**: Verify that the MMA reads from the
   same TMEM address that the store writes to.

### Example: Finding the root cause via TTGIR comparison

The BWD dV accuracy issue was found by comparing `tmem_store` counts:

```bash
# TLX has 1 tmem_store (ppT only) — NO zeroing
grep -c 'tmem_store' tlx-b64/_attn_bwd_ws.ttgir   # → 1

# AutoWS has 4 tmem_stores (dk zeros + dv zeros + 2× ppT)
grep -c 'tmem_store' b64/_attn_bwd_persist.ttgir    # → 4
```

The extra `tmem_store dense<0.0>` in autoWS (reduction partition, task 0)
creates a cross-partition race that TLX avoids entirely by relying on the
MMA's `useC=false` flag.

---

## Option 3: Focus on Missing Synchronization or Early Releases

**When to use**: When the error pattern shows a race condition (probabilistic
failures, iteration-0-pass / iteration-1+-fail pattern).

### Trace the TMEM accumulator lifecycle

For each TMEM accumulator (dk, dv, dq), trace the full lifecycle:

```
ZERO: tmem_store 0 → TMEM         (which partition? which barriers?)
ACCUMULATE: tc_gen5_mma → TMEM    (inner loop, useC flag)
COMMIT: tc_gen5_commit             (signals committed barriers)
READ: tmem_load ← TMEM            (which partition? which barriers?)
CONSUMED: arrive_barrier           (signals consumed to allow next zeroing)
```

### Check for these specific issues

1. **Cross-partition zeroing race**: The reduction partition zeros the TMEM
   before the computation partition finishes reading it from the previous
   iteration. Look for:
   - `tmem_store` of zeros in a DIFFERENT partition than `tmem_load`
   - Whether a guard channel (`isSameIterGuard`) exists between them
   - Whether the guard channel barrier is correctly arrived AFTER the
     tmem_load and waited BEFORE the tmem_store

2. **Missing back-edge channel**: The dependency
   `tmem_load(current iter) → tmem_store(next iter)` may not be modeled.
   In `handleOperandD()`, check if the wrap-around channel and guard channel
   are created for this TMEM alloc.

3. **Barrier phase mismatch**: The wait uses phase `NOT(tile_idx % 2)` but
   the arrive uses a different phase convention. Verify the parity argument
   in `mbarrier.try_wait.parity` matches the arrive pattern.

4. **Redundant zeroing**: If the MMA uses `useC=false` on the first inner
   iteration, the preceding `tmem_store` of zeros is redundant. The
   `tmem_store` creates a cross-partition dependency that wouldn't exist if
   it were eliminated.

### Verify barrier wiring through warp_specialize

The `warp_specialize` op passes barriers as operands to all partitions.
Each partition receives them as block arguments at the same positions:

```
warp_specialize(op0, op1, ..., opN)
  default(arg0, arg1, ..., argN)     ← reduction
  partition0(arg0, arg1, ..., argN)  ← gemm
  partition1(arg0, arg1, ..., argN)  ← load
  partition2(arg0, arg1, ..., argN)  ← computation
```

To verify a barrier is correctly wired:
1. Find the arrive_barrier in the producer partition → get the parameter name
2. Map the parameter name to its operand position (count from 0)
3. Find the wait_barrier in the consumer partition → get its parameter name
4. Verify it's at the SAME operand position

### Verify in PTX

Map barrier SMEM offsets between partitions:

```bash
# Find init addresses
grep 'mbarrier.init' file.ptx | grep 'add.*r1472'

# Find wait addresses near zeroing
grep -B5 'tcgen05.st.*r1315' file.ptx | grep 'try_wait'

# Find arrive addresses after tmem_load
grep -A5 'tcgen05.wait::ld' file.ptx | grep 'mbarrier.arrive'
```

The SMEM offset (`%r1472 + OFFSET`) must match between the wait and arrive.

---

## Option 4: Override TTGIR to Test Hypotheses

**When to use**: When you've identified a specific hypothesis (e.g.,
"the zeroing tmem_store is causing the race") and want to validate it by
modifying the compiled IR.

### Approach A: Override TTGIR to use TLX's pattern

1. Take the failing autoWS TTGIR (`b64/_attn_bwd_persist.ttgir`)
2. Edit it to match the TLX pattern for the specific operation under test
3. Recompile from the modified TTGIR and run the accuracy test

Example modifications:
- Replace the ppT subslice/reinterpret chain with TLX's simpler version
- Remove the `tmem_store dense<0.0>` for dk/dv (rely on `useC=false`)
- Change barrier placement

### Approach B: Remove redundant zeroing

If the MMA already uses `useC=false` on the first inner iteration, the
`tmem_store` of zeros in the reduction partition is redundant. Remove it:

1. Find the `tmem_store` in the default partition:
   ```
   ttng.tmem_store %cst_28, %dv_123, %true  ← zeros dv
   ttng.tmem_store %cst_28, %dk_117, %true  ← zeros dk
   ```
2. Remove these instructions but **keep all surrounding barrier
   waits/arrives intact** — the barrier protocol must remain consistent
   so other partitions don't deadlock
3. The MMA's `useC=false` will handle the zeroing without needing
   cross-partition synchronization

**Validated**: This approach was confirmed to fix the BM64 dV accuracy
issue. TTGIR override with dk/dv zeroing removed produced **ALL PASS**
with **0.0 error** for dV (matching TLX perfectly).

### Approach C: Move zeroing to the same partition

Instead of removing the zeroing, move it to the same partition as the
MMA (gemm partition, task 1). This eliminates the cross-partition
synchronization requirement:

1. Change the `async_task_id` on the `tmem_store` from `array<i32: 0>`
   (reduction) to `array<i32: 1>` (gemm)
2. Remove the cross-partition barriers for the zeroing
3. The zeroing and MMA are now in the same partition — no barrier needed

### Validation

After any TTGIR modification:
1. Recompile from the modified TTGIR
2. Run the accuracy test with the probabilistic config (e.g., Z=4,H=8 3×)
3. If all 3 runs pass → the hypothesis is confirmed
4. If still failing → the hypothesis was wrong, try another modification

---

## Key Files

| File | Purpose |
|------|---------|
| `python/tutorials/test_tlx_bwd_from_fused_attention.py` | Test harness with `_debug_dv()` diagnostics |
| `python/tutorials/fused-attention-ws-device-tma.py` | AutoWS BWD kernel source |
| `third_party/tlx/tutorials/blackwell_fa_ws_pipelined_persistent.py` | TLX BWD kernel source |
| `b64/_attn_bwd_persist.ttgir` | AutoWS TTGIR (failing) |
| `b64/_attn_bwd_persist.ptx` | AutoWS PTX (failing) |
| `b128/_attn_bwd_persist.ttgir` | AutoWS TTGIR (working, different block size) |
| `b128/_attn_bwd_persist.ptx` | AutoWS PTX (working) |
| `tlx-b64/_attn_bwd_ws.ttgir` | TLX TTGIR (working, same block size) |
| `devmate/codepart/ws_code_partition_bwd_findings.md` | Detailed findings for all bugs found |

## Key Compiler Files

| File | Purpose |
|------|---------|
| `CodePartitionUtility.cpp` | `handleOperandD()` — creates guard channels for TMEM accumulators |
| `WSCodePartition.cpp` | `insertAsyncComm()` — inserts barrier wait/arrive for channels |
| `WSMemoryPlanner.cpp` | TMEM allocation and column reuse |
| `TMEMAlloc1D.cpp` | `createTMEMDesc()` — computes colStride for TMEM encodings |
| `WSLowerToken.cpp` | Lowers ProducerAcquire/ConsumerRelease to mbarrier ops |

## Lessons Learned

1. **TLX is the ground truth**: When autoWS fails, compare with TLX first.
   The structural difference (e.g., extra tmem_store, different partition
   assignment) is usually the root cause.

2. **Redundant zeroing is dangerous**: If the MMA uses `useC=false`, a
   separate `tmem_store` of zeros in a different partition creates an
   unnecessary cross-partition race. The compiler should recognize this as
   dead code.

3. **Probabilistic failures need repeated runs**: Run borderline configs
   (just enough tiles for multi-tile CTAs) at least 3 times. A single pass
   doesn't prove correctness.

4. **Barrier wiring correctness ≠ race freedom**: Even when all barrier
   SMEM offsets, phases, and init counts are verified correct, a race can
   still exist if the barrier protocol itself is flawed (e.g., the
   dependency it protects is redundant, and removing the source of the
   dependency is the actual fix).
