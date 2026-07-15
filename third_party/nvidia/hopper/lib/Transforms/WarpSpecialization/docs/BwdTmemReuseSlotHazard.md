# FA-bwd 3-Buffer TMEM Reuse Slot Hazard (accuracy)

This documents a correctness bug in the FA backward kernel's 3-buffer TMEM reuse
(`_BWD_DOT_ATTRS_TMEM`) that produced **NaN**, the fix, why it could not be
caught by a simple code-partition check, and the design of a late-pass
barrier-aware analysis that *could* catch it.

Related:
- Kernel: `third_party/tlx/tutorials/fused_attention_ws_device_tma.py`
- Reuse-group machinery: [ReuseGroups.md](ReuseGroups.md)
- Regression test: `test_bwd_tmem_dsT_reuse_3group` (in the kernel file)
- Hazard lit spec: `test/Hopper/WarpSpecialization/ws_reuse_slot_hazard_xfail.mlir`

## The reuse pattern

`_BWD_DOT_ATTRS_TMEM` (selected by `bwd_config_idx=2`) packs three logical
buffers into **one** TMEM allocation, `buffer.id=5`, to fit the 512-column TMEM
budget:

```
qkT: ... opndD,tmem,1,2
dpT: ... opndD,tmem,1,5      # dpT MMA result          (writes buffer-5)
dv:  ... opndD,tmem,1,7
dq:  ... opndD,tmem,1,5      # dq MMA result           (writes buffer-5)
dk:  opndA,tmem,1,5 ...      # dk reads dsT from TMEM   (reads  buffer-5)
```

The kernel computes `dsT = pT * (dpT - Di)` in registers, then both `dq` and
`dk` consume `dsT`:

```python
dq = tl.dot(tl.trans(dsT), k, attrs=...["dq"])   # dq result -> buffer-5
dk += tl.dot(dsT, tl.trans(qT), attrs=...["dk"])  # dk reads dsT from buffer-5
```

`dsT` is materialized into the buffer-5 slot (a `memdesc_reinterpret` of `dpT`'s
TMEM, f32→f16, sub-slice `N=0`) by a `tmem_store` in the **computation**
partition (task 3). The **dk** MMA reads it as operand A in the **gemm**
partition (task 1). The **dq** MMA writes its own result into the same slot,
also in the gemm partition.

So the single physical slot is time-multiplexed; the required order each
iteration is:

```
dpT MMA writes slot  ->  computation reads dpT, computes dsT  ->
  computation stores dsT into slot  ->  dk reads dsT from slot  ->
    dq writes its result into slot  ->  ...
```

## The bug (prior to the fix)

The kernel emitted **dq before dk** (`tl.dot(...dq...)` then `tl.dot(...dk...)`).
dq and dk are both gen5 MMAs in the **same** partition (gemm), so program order
*is* execution order and no barrier can reorder them. The dq MMA therefore
**overwrote the slot before dk read `dsT`**, so dk read dq's output (garbage)
instead of `dsT`. Result: `dv`/`dk`/`dq` came back **NaN** (`ws` non-persistent,
`bwd_config_idx=2`).

This sat behind a separate **deadlock** (the reuse-group write-after-read
`producer_acquire` for dq was emitted on a token the dk gen5 consumer never
releases — see the deadlock fix in `WSCodePartition.cpp`). Once the deadlock was
fixed, the NaN surfaced.

Confirming experiments:
- No reuse-WAR for dq → dq clobbers `dsT` → NaN.
- A real reuse-WAR (dq waits for dk) → **deadlock**, because dk is scheduled
  *after* dq in the same partition (a barrier cannot reorder them).
- `compute-sanitizer --tool racecheck` → **0 hazards** (it is a deterministic
  wrong-data bug, not a shared-memory race; TMEM is not visible to racecheck).
- Giving `dsT` its own TMEM buffer (no reuse) → `OutOfResources: 576 > 512`
  (the reuse is mandatory for the budget).

## The fix

Emit **dk before dq** in the kernel:

```python
dsT = pT * (dpT - Di)
dsT = dsT.to(dtype)
dk += tl.dot(dsT, tl.trans(qT), attrs=...["dk"])   # was second; now first
dq = tl.dot(tl.trans(dsT), k, attrs=...["dq"])      # was first;  now second
```

### Why the swap is safe (semantics-preserving)

`dsT` is a **register** value, computed once (above). Both dots only *read*
`dsT`; `dq`'s result is not an input to `dk` and vice versa — there is no data
dependency between them. The `attrs` (`BWD_DOT_ATTRS`) do not change the math;
they only tell the compiler *where to place each MMA's operands* (which SMEM/TMEM
buffer). So reordering the two `tl.dot` calls cannot change any computed value —
`dk` and `dq` produce the same gradients either way.

### Why dk-before-dq fixes the accuracy bug

The shared slot (buffer-5) holds `dsT` and then `dq`'s result, in that order:
- `dk` **reads** `dsT` from the slot (operand A, `opndA,tmem,1,5`).
- `dq` **writes** its result into the same slot (operand D, `opndD,tmem,1,5`).

With `dk` emitted first, the per-iteration slot sequence becomes
`... dsT-write -> dk-read(dsT) -> dq-write ...` — `dk` observes the correct
`dsT`, and `dq` only overwrites the slot *after* `dk` is done with it. (`dq`
itself reads `dsT` from SMEM, not from this TMEM slot, so it is unaffected.)

Crucially, `dk` and `dq` are both gen5 MMAs in the **same** partition (gemm), so
the tensor core executes them in **source order** — the source-level reorder
*is* the execution-order guarantee. No synchronization barrier is needed (and,
as the deadlock experiment showed, none can substitute: a barrier cannot make a
later same-partition MMA run before an earlier one). This is exactly the
constraint the TLX reference (`blackwell_fa_ws_pipelined_persistent.py`) states:

> dk must read dsT_tmem BEFORE dq writes dq_tiles (same TMEM slot).

Verified by `test_bwd_tmem_dsT_reuse_3group` (fails with NaN without the
ordering, passes with it). Note the **persistent** backward
(`ws_persistent`) remains separately unusable (it needs 704 > 512 TMEM columns
and the pipeliner cannot predicate `descriptor_reduce`); the reuse fix is
validated on the usable non-persistent path.

## Why this is not caught in `doCodePartitionPost`

A natural idea is to reject the bad ordering during reuse-group validation. That
was tried (a program-order check: a channel's producer writing the slot before
another channel's consumer reads it, in the same partition, with cross-partition
read data) and it **false-positives**: it also fired on the *correct* default
config (`bwd_config_idx=0`).

The reason is timing in the pipeline: reuse-group validation runs in
`doCodePartitionPost` **before** the reuse-WAR barriers are inserted (those come
later in `insertAsyncComm`). "Writer before reader" is *normal* for valid
cross-partition reuse — the inserted barrier makes it safe (e.g. dpT writes the
slot, the computation partition overwrites it with `dsT`, then dk reads — all
correct even though dpT precedes dk). The dq/dk case is unsafe only because the
two are gen5 MMAs in the **same** partition with **no intervening establishment
of dk's data**, so no barrier can fix it. Program order alone cannot distinguish
"will be made safe by a barrier" from "unfixable."

## Proposed late-pass barrier-aware / aliasing analysis

A reliable compile-time check must run **after `insertAsyncComm`** (barriers
present) and **before `specializeRegion`** (still a single region with
`ttg.partition` attributes, so cross-partition program order and barrier edges
are both visible). Sketch:

1. **Group by physical slot.** Use the `ReuseConfig` groups (channels sharing
   `buffer.id`). For TMEM, resolve aliasing through `tmem_subslice` /
   `memdesc_reinterpret` / `memdesc_index` to (alloc, column-interval) so two
   accesses are known to overlap.

2. **Build a happens-before graph** from the synchronization edges already in
   the IR: `arrive_barrier`/`tc_gen5_mma` completion barriers → matching
   `wait_barrier` (by barrier SSA value + phase), plus intra-partition program
   order (gen5 MMAs in one partition execute in order).

3. **Verify each slot read observes its intended writer.** For every read R of
   the slot, let W_R be the writer whose data R is meant to consume (the writer
   that R's value is derived from — e.g. the `tmem_store` of `dsT` for dk). For
   every *other* writer W' of an overlapping region: require `R` happens-before
   `W'` **or** `W'` happens-before `W_R` in the graph. If some W' can run between
   W_R and R with no happens-before edge ordering it after R, it can clobber
   R's data → **emit an error** at W' / R.

   In the bug, W' = dq MMA, R = dk read, W_R = computation `tmem_store` of dsT.
   dq and dk are same-partition with dq before dk and no barrier ordering dk
   first, so dq is not happens-before W_R nor after R → hazard.

4. **Diagnostic.** Report the writer op, the reader op, and the shared
   `buffer.id`, suggesting the writer be ordered after the reader (e.g. emit dk
   before dq).

This is strictly more than a heuristic — it needs the barrier graph and the
aliasing resolution — which is why it belongs in a dedicated post-`insertAsyncComm`
verification pass rather than in `doCodePartitionPost`.
`ws_reuse_slot_hazard_xfail.mlir` is the executable spec for it.

## main4 reconciliation: keep the group out of the A6 hub

When this stack is rebased onto `origin/main`, the base now carries the A6
**whole-allocation-overwrite hub** (`isWholeAllocationOverwriteReuseOwner`, from
the upstream "backwards QK barrier depends on all aliasing buffers + zeroing"
change). That re-exposes this exact hazard through a new path: the `{dpT, dq,
dsT}` group's representative is a `useC=false` MMA, so it satisfies
`isWholeAllocationOverwriteReuseOwner`, and the N>2 dispatch routes the group to
A6. A6's hub back-edges are designed for **spatial packing** (siblings in
*distinct* columns of the owner); applied to this **full-overlap** group (all
channels at `buffer.offset=0`) they emit the wrong synchronization, so the `dq`
overwrite again races `dk`'s read of `dsT` → wrong `dq` (same symptom as the
original bug).

The fix gates A6 to spatial packing only: A6 fires solely when the group spans
**≥2 distinct `buffer.offset`s** (`isSpatialPacking`). The `{dpT, dq, dsT}` group
(all offset 0) is therefore excluded from A6 and handled by the standard reuse
path + the dk-before-dq kernel ordering documented above. See
[ReuseGroups.md → "A6 applies only to spatial packing"](ReuseGroups.md) for the
gate and why column-overlap / block-based predicates cannot distinguish the two
shapes at this stage. Runtime regression: `fused_attention_ws_device_tma.py`'s
`test_bwd_tmem_dsT_reuse_3group` (fails with wrong `dq` if A6 captures the group;
passes with the gate).
