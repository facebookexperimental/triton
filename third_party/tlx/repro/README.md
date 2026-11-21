# gfx950 packed FP32 subtraction repro

Two confirmed standalone repros exercise MFMA32 on four waves and packed
subtraction on four other waves in an eight-wave workgroup. The TLX version
uses `tlx.warp_predicate` and inline assembly to fix the exact instructions.
Neither repro needs attention, softmax, or LDS.

Paths below are relative to the OSS TLX repository root after export.

## TLX

Source: `third_party/tlx/repro/repro_tlx_cross_wave.py`.
Requires ROCm Torch and TLX-enabled Triton on gfx950. With that environment,
run the source directly with Python:

```bash
HIP_VISIBLE_DEVICES=0 python3 third_party/tlx/repro/repro_tlx_cross_wave.py --launches 50
```

Exit 2 means a numerical mismatch; exit 0 means no mismatch was observed.
Use `--mode 1` for MFMA16, `--mode 2` for VALU, `--mode 3` for NOP,
`--form 1` for commuted operands, or `--form 2` to remove op_sel.

## HIP

Source: `third_party/tlx/repro/repro.hip`.
With a stock ROCm toolchain:

```bash
hipcc --offload-arch=gfx950 -O3 -o gfx950_packed_add_repro third_party/tlx/repro/repro.hip
HIP_VISIBLE_DEVICES=0 ./gfx950_packed_add_repro 0 50 1024 4096 65536 0
```

Arguments: mode, launches, blocks, MFMA iterations, ADD iterations, form.
Modes and forms match the TLX controls.

## Captured AMDGCN

`third_party/tlx/repro/tlx_cross_wave_m0_f0_w0.amdgcn`
is the saved failing TLX kernel. Only debug-directory paths have been changed
to OSS paths; executable instructions are unchanged. The source also writes
generated AMDGCN beside itself on the first launch. The packed subtract is:

```asm
v_pk_add_f32 v[36:37], v[32:33], v[34:35] op_sel:[0,1] neg_lo:[0,1] neg_hi:[0,1]
```

On gfx950 / ROCm 7.0, the default TLX case failed in 50/50 launches;
each of the five controls passed 50 launches. Lanes 48-63 produced low 158
instead of 158 - 652 = -494, while high -351 was correct. The final TLX
version also failed with `--waves-per-eu 2`; a prior clean version did not
isolate occupancy. These observations establish a reproducer, not compiler
versus silicon attribution or a production-safe workaround.
