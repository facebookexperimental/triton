# gfx950 LDS WAR race from a dropped `lgkmcnt(0)` at a loop-header barrier

A warp-pipelined BF16 addmm on MI350X (gfx950) gives sparse, large errors. This happens when the
explicit LDS wait from `ConvertWarpPipeline` is absent. The root cause is in LLVM's
`SIInsertWaitcnts`, and is fixed upstream by
[llvm/llvm-project@437f88f61b](https://github.com/llvm/llvm-project/commit/437f88f61b16e5564713d0058433f4b34c41bf51)
(#220357).

## What goes wrong

1. Each loop iteration ends with `ds_read`s of LDS slot `(i + 1) % 3`. The next iteration starts
   with a local barrier (`fence release` + `s_barrier` + `fence acquire`). After the barrier,
   another warp group issues an async global-to-LDS copy into that slot.
2. `SIMemoryLegalizer` emits `S_WAITCNT_soft lgkmcnt(0)` for the release fence, which is correct.
3. `SIInsertWaitcnts` visits the loop header first with only the preheader state, which has no
   pending LDS access. It decides the soft wait is redundant and erases it. The backedge then
   brings in the pending `ds_read`s, but the wait is gone. The result is
   `s_waitcnt vmcnt(2)` + `s_barrier`, so the reads cross the barrier and race with the copy.

## Files

| File | Purpose |
|---|---|
| `waitcnt-loop-soft-wait.ll` | Minimal lit test (`llc` + `FileCheck`); no GPU needed |
| `kernel_nofix.ll` | Full kernel LLIR without the Triton workaround |
| `kernel_fix.ll` | Full kernel LLIR with the Triton workaround (explicit `s_waitcnt lgkmcnt(0)`) |
| `check.sh` | Runs the lit test and the loop-head check on both kernels against a given `llc` |
| `addmm_warppipe.py` | Runnable GEMM; counts failing launches out of 1000 |

## Lowering check (any machine)

```
./check.sh /path/to/llc [/path/to/FileCheck]
```

Or run the lit test directly with `llvm-lit waitcnt-loop-soft-wait.ll`.

## GEMM (MI350X)

Requirements: a build of [facebookexperimental/triton](https://github.com/facebookexperimental/triton)
(it provides the TLX extensions) and a ROCm build of PyTorch.

```
python addmm_warppipe.py                  # kernel as this Triton build compiles it
python addmm_warppipe.py --strip-ds-wait  # remove the explicit lgkmcnt(0) waits from the LLIR
```

`--strip-ds-wait` uses Triton's stage-inspection hook to delete
`llvm.amdgcn.s.waitcnt(i32 49279)` (`lgkmcnt(0)`) from the LLIR before code generation.
This gives an A/B comparison on one build.

## Expected results

| LLVM used for AMD code generation | `check.sh` | GEMM, default | GEMM, `--strip-ds-wait` |
|---|---|---|---|
| Without 437f88f61b (for example `ce3529423abd`) | lit FAIL, `kernel_nofix` FAIL, `kernel_fix` PASS | 0 / 1000 | 999-1000 / 1000 failing |
| With 437f88f61b | all PASS | 0 / 1000 | 0 / 1000 expected |

Measured on MI350X with Triton's AMD codegen built against an LLVM without the fix:
- Default: 0 / 1000 failing launches.
- `--strip-ds-wait`: 999-1000 / 1000 failing launches (two runs), 2–8k mismatched elements each, max abs diff about 40.

The "with 437f88f61b" GEMM row is only checked at the ISA level. With the patched `llc`,
`kernel_nofix.ll` compiles to the same instructions as `kernel_fix.ll`.
