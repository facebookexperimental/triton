# Case Study (Path B): omninn Cross-Attention Triton 3.5 to 3.8 Regression

Condensed from the 2026-11-05 investigation of
`_persistent_shared_q_indexed_group_q_kernel` (omninn cross-attention,
reported in D130048765) on H100. This is the example where tritonparse's
built-in bisect could not be used and the file-group revert fallback
found the culprit.

## Why Path A failed the gate

- The reporter was on fbcode trees. Building the exact fbcode snapshot
  trees (`third-party/triton/legacy` for 3.5, `third-party/triton/3.8`)
  reproduced the report exactly (fbcode 3.5 spills matched all 7
  reported configs; the worst config was roughly 15% slower on 3.8).
- Building the GitHub release branch tips did not reproduce it: GitHub
  3.5 was itself slow on the worst config, so there was no good
  commit in any GitHub history to bisect from. Endpoint condition 1 of
  the gate failed.
- The reporter's own buck command, run on both versions, took about
  35 minutes on each for the autotune: the claimed compile-time
  regression was not real. Only the runtime regression was bisected.

## How Path B proceeded

1. Bench oracle on the worst config (BLOCK_BQ=1, BLOCK_N=32, warps=4,
   stages=3), same GPU back-to-back: fbcode 3.5 fast, fbcode 3.8 slow
   (about 15% apart), upstream base `71e79432` slow like 3.8. Note the
   upstream base was slow too, which later mattered.
2. tritonparse capture/parse/diff localized the difference below TTGIR.
   A TTGIR difference (one extra `convert_layout` for a loop-carried
   accumulator) looked causal but was a false lead: reverting
   RemoveLayoutConversions removed the conversion and the runtime stayed
   fast.
3. File groups on the fast tree were reverted toward the upstream base:
   RemoveLayoutConversions (still fast), Pipeliner files (still fast),
   Analysis (still fast), LinearLayoutConversions (still fast),
   Conversion/MemoryOpToLLVM (still fast), and finally the
   Conversion/TritonGPUToLLVM group including `Utility.cpp`, which
   turned the tree slow. Culprit inside that file.
4. The file diff was 136 lines; the operative hunk was in `getThreadId`:
   the fast tree had no threadIdx known-bits mask, the slow trees had
   `tid = b.and_(tid, b.i32_val(upperBound - 1))`.

## Two-direction confirmation

- Delete only the mask line on the fbcode 3.8 tree: the worst config
  recovered almost all of the regression (spills also dropped slightly).
- Delete only the mask line on the upstream base: it became as fast as
  fbcode 3.5.

## Provenance (found after the bisect)

The mask was introduced upstream by triton-lang PR #7249 and reverted
on upstream's release/3.5.x branch by PR #8211 (because of a ptxas
WARPGROUP.DEPBAR.LE mis-insertion family, the same mechanism class as
this regression). That revert never entered facebookexperimental's
release/3.5.x. Internally, D91090707 backported the revert into the
fbcode stable tree (2026-01-22), and a stable-to-legacy promote carried
it into the 3.5 endpoint; 3.8 kept the mask in a refined
total-num-warps form. The culprit therefore appeared in no backport
list: the fast behavior arrived by tree promote, which is why the
skill's Path B tells you to check internal promote history and pickaxe
when the listed backports come up empty. A separate candidate list (146
backports in the fast tree absent from 3.8 history, 3 of them NVIDIA
compiler commits) was checked and was not responsible; single-hunk
confirmation had already accounted for the full gap.

## Lessons this case contributes to the skill

- Reproduce the reporter's environment; substituting GitHub releases
  for fbcode trees made the regression vanish and would have produced a
  false "cannot reproduce".
- Two parallel build lanes with per-lane TRITON_HOME were enough; a
  shared TRITON_HOME let parallel builds wipe each other's LLVM
  download, and one bench run measured a stale install and had to be
  discarded after checking `triton.__file__`.
- Cherry-pick chains of the recorded backport list mostly do not apply
  (26 of 146 clean; 7 of 37 compiler candidates cumulatively). File-group
  reverts converge where cherry-picks do not, at the price of API-drift
  build fixes along the way.
