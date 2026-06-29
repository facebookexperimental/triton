---
name: running-with-buck
description: >
  How to build and run GPU targets under Buck in fbcode. Use when invoking
  buck2 run / buck2 build for any GPU benchmark, test, or kernel — selecting
  the GPU architecture and CUDA version, using @mode/opt and the beta Triton
  modifier, passing environment variables through, and running from the right
  directory. Covers the
  general requirements plus the B200/GB200 (b200a, CUDA >= 12.8) and GB300
  (b300a, CUDA >= 13.0) hardware requirements.
---

# Running GPU Targets with Buck

This skill covers the mechanics of building and running **GPU** targets
under Buck in fbcode. It is target-agnostic: substitute your own
`<buck target>` and program arguments.

## General requirements

1. **Run from `fbsource/fbcode`.** `cd` to `fbsource/fbcode` before
   invoking Buck. The `@mode/opt` flags (and other `@mode/...` files) only
   resolve when Buck is run from there.
2. **Use `@mode/opt`.** It provides the core GPU build configuration.
   `@mode/opt` generally sets up the GPU build, but some tritonbench targets
   still pass `-c fbcode.enable_gpu_sections=true` explicitly — add it if a
   target's GPU sections fail to build.
3. **Build against the beta Triton with `-m ovr_config//triton:beta`.** This
   directory *is* the beta Triton compiler. Pass this modifier so the target
   builds and runs against the beta Triton in this tree rather than the
   default/stable Triton — without it, changes made here are not exercised.
4. **Select the GPU architecture** with `-c fbcode.nvcc_arch=<arch>` and,
   where required, the **CUDA version** with
   `-c fbcode.platform010_cuda_version=<ver>` (see *Hardware
   requirements*).
5. **Environment variables prefix the `buck2 run`** and are forwarded to
   the launched process:
   ```bash
   <ENV VARS> buck2 run @mode/opt -m ovr_config//triton:beta -c fbcode.nvcc_arch=<arch> [-c fbcode.platform010_cuda_version=<ver>] \
     <buck target> -- <program args>
   ```
6. **`buck2 build` vs `buck2 run`.** Use `buck2 build <target>` to compile
   only (e.g. to surface a compile failure without executing); use
   `buck2 run <target> -- <args>` to build and run. Program arguments go
   after `--`.

## Hardware requirements

Pick the arch (and CUDA version) for the single GPU you are targeting:

| Hardware | `fbcode.nvcc_arch` | `fbcode.platform010_cuda_version` |
| --- | --- | --- |
| Hopper (H100) | `h100a` | (default) |
| Blackwell B200 / GB200 | `b200a` | `>= 12.8` |
| Blackwell GB300 | `b300a` | `>= 13.0` |

Notes:
- **B200 / GB200** require CUDA `>= 12.8`. Existing tritonbench targets pin
  `12.8`; use the version your build expects.
- **GB300** requires arch `b300a` **and** CUDA `>= 13.0`.
- Set the CUDA version explicitly whenever a minimum applies, since the
  platform default may be older than the arch requires.

## Examples

**Blackwell GB300** (`b300a`, CUDA 13.0), from `fbsource/fbcode`:
```bash
buck2 run @mode/opt -m ovr_config//triton:beta \
  -c fbcode.nvcc_arch=b300a \
  -c fbcode.platform010_cuda_version=13.0 \
  <buck target> -- <program args>
```

**Blackwell B200 / GB200** (`b200a`, CUDA `>= 12.8`):
```bash
buck2 run @mode/opt -m ovr_config//triton:beta \
  -c fbcode.nvcc_arch=b200a \
  -c fbcode.platform010_cuda_version=12.8 \
  <buck target> -- <program args>
```

**Hopper** (`h100a`):
```bash
buck2 run @mode/opt -m ovr_config//triton:beta \
  -c fbcode.nvcc_arch=h100a \
  <buck target> -- <program args>
```

**With env vars forwarded** (e.g. enabling a feature for the run):
```bash
SOME_ENV=1 buck2 run @mode/opt -m ovr_config//triton:beta -c fbcode.nvcc_arch=b300a \
  -c fbcode.platform010_cuda_version=13.0 \
  <buck target> -- <program args>
```

**Compile only** (surface a build/compile failure without running):
```bash
buck2 build @mode/opt -m ovr_config//triton:beta -c fbcode.nvcc_arch=b300a \
  -c fbcode.platform010_cuda_version=13.0 \
  <buck target>
```
