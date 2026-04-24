---
name: tma-illegal-instruction
description: >
  Diagnose CUDA "illegal instruction" / kernel crashes on Triton kernels that
  reference to TMA loads or stores (`make_tensor_descriptor`, `TensorDescriptor`,
  `descriptor.load`, `descriptor.store`, `tl.async_descriptor_load`, async TMA
  copies) as the source code line. Use when the user reports CUDA error 716,
  "an illegal instruction was encountered", segfault inside a TMA op, kernel hang
  followed by an illegal instruction trap, or a crash that only fires on the
  first or last tile of a launch. Covers the pattern where a TMA store/load is
  issued at an offset entirely past a tensor's shape — TMA does NOT silently mask
  out-of-bounds tile accesses; it traps. The root cause is almost never
  "missing in-kernel mask" — it is commonly a structural launcher /
  tile-mapping bug.
---

# TMA Illegal Instruction

## Symptom

CUDA reports "an illegal instruction was encountered" (error 716), or the
kernel crashes inside a TMA op, on a Triton kernel that uses TMA descriptors
(`TensorDescriptor`, `tl.make_tensor_descriptor`, `desc.load(...)`,
`desc.store(...)`, async TMA copies, etc.).

The crash is likely tile-dependent — appears only at certain grid values.
This is likely because the tile out of bounds is entirely past the
shape of the TME store.

## Diagnosis ladder

Walk these in order. Don't skip ahead — the first check is the cheapest and
the most often correct.

1. **Find the faoiling TMA p.** From the stack trace / sanitizer output / IR
   dump, identify which `descriptor.load(...)` or `descriptor.store(...)`
   crashed. Note the offsets it was called with (e.g.
   `[pid_m * BM, pid_n * BN]`) and the descriptor's declared `shape`.

2. **Reconstruct the failing tile's starting offset.** For the failing
   program/iteration, compute the literal integer offsets passed to the TMA
   op. For each axis `i` of the descriptor, ask: **is `off_i >= shape_i`?**
   If yes, that is the bug. The launcher / tile-mapping logic put a program
   in a region that does not exist.

3. **Confirm by debug messaging.** Determine either the grid or value
  (could be a jagged tensor) information that is causing the failure.
  Add a `tl.device_print` call to the kernel with an if that skips the
  operation. NOTE: This is the not a proper solution!

4. **Only after the structural bug is identified**, determine whether the right
   fix is launcher/grid dependent or runtime data dependent. If the latter,
   identify how this shape can be reached.

## Anti-pattern: "just add a mask"

The common temptation is to wrap the failing TMA op in
`if off_m < M and off_n < N:` (or to fall back to `tl.load` with a mask).
**Resist this.** It silences the symptom but:

- Hides the structural bug — the kernel is still launching programs that own
  no work, wasting a CTA per stray program.
- Often masks correctness issues elsewhere — if the kernel reached an
  out-of-bounds tile, the `tile_id` it computed for the *previous* tiles is
  also suspect.
- For epilogue stores, the masked-out tile's accumulator was still computed
  from junk loads further up the kernel — meaning some *other* tile may have
  written wrong data that the mask doesn't catch.

In-kernel masks are fine for genuinely ragged shapes (real K not a multiple
of BLOCK_K, etc.), but a TMA illegal instruction is a different signal — it
says "the launch contract is wrong", not "this iteration is ragged".

## Verify the fix

For the failing tile/iteration, the kernel should be able to assert
`off_i < shape_i` for every TMA op. The verification protocol:

1. Add temporary `tl.device_assert(off_i < shape_i, "...")` calls (or print
   the offsets) before the suspected TMA op and re-run with the same shape
   that crashed.
2. Confirm the assert fires at the same iteration the illegal instruction
   was hitting — that proves you found the actual offending access.
3. Apply the structural fix (launcher / grid / descriptor).
4. Re-run the same shape: the asserts no longer fire **and** the illegal
   instruction is gone. If the asserts pass but the crash remains, it is a
   different TMA op or a different bug class — go back to step 1 of the
   diagnosis ladder.

Removing `tl.device_assert` after verification is required; the structural fix
is what you ship. The code should NOT introduce a new if statement directly over
just the TMA operation (that is typically wrong).
