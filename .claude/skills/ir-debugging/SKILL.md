---
name: ir-debugging
description: >
  Debug Triton compilation by dumping IR at each stage (TTIR, TTGIR, LLVM, PTX).
  Use when investigating compilation failures, kernel performance, register
  spills, or when user asks to inspect IR output. Covers TRITON_KERNEL_DUMP,
  MLIR_ENABLE_DUMP, LLVM_IR_ENABLE_DUMP, TRITON_DUMP_PTXAS_LOG, and related env vars.
---

# IR Debugging

## Environment variables

| Env var | What it does |
|---|---|
| `TRITON_KERNEL_DUMP=1` | Dump IR at every compilation stage to `~/.triton/dump/` |
| `TRITON_PRINT_AUTOTUNING=1` | Use human-readable per-config subdirectories instead of hashes (combine with KERNEL_DUMP) |
| `TRITON_KERNEL_DUMP_BEST_CONFIG=1` | Dump IR only for the winning autotuned config (re-compiles with dumping, avoids noise) |
| `MLIR_ENABLE_DUMP=1` | Dump MLIR IR during pass execution (filter by kernel: `MLIR_ENABLE_DUMP=_kernel`) |
| `LLVM_IR_ENABLE_DUMP=1` | Dump LLVM IR (print-after-all) |
| `NVPTX_ENABLE_DUMP=1` | Dump NVPTX backend IR |
| `TRITON_DUMP_PTXAS_LOG=1` | Dump ptxas assembler logs (register usage, spills) |
| `TRITON_INTERPRET=1` | Run kernels in interpreter mode (no GPU needed) |
| `TRITON_ALWAYS_COMPILE=1` | Bypass cache, force recompilation |
| `TRITON_DUMP_TTGIR_TO_TLX=1` | Dump TTGIR back to TLX Python (reverse-engineer IR) |

## Decision tree: what are you debugging?

- **"Kernel produces wrong results"**
  → `TRITON_INTERPRET=1` to run on CPU, or `TRITON_KERNEL_DUMP=1` to inspect IR at each stage
- **"Kernel is slow / register spills"**
  → `TRITON_DUMP_PTXAS_LOG=1` to check register usage and spills
- **"Which autotuned config won and why?"**
  → `TRITON_KERNEL_DUMP_BEST_CONFIG=1 TRITON_PRINT_AUTOTUNING=1`
- **"Need to see MLIR passes"**
  → `MLIR_ENABLE_DUMP=1` (optionally filter: `MLIR_ENABLE_DUMP=_my_kernel`)
- **"Need to see final PTX/LLVM"**
  → `LLVM_IR_ENABLE_DUMP=1` and/or `NVPTX_ENABLE_DUMP=1`
- **"Cached result is stale"**
  → `TRITON_ALWAYS_COMPILE=1` to force recompilation

## Common combos

```bash
# Full dump of best config with readable directory names
TRITON_KERNEL_DUMP_BEST_CONFIG=1 TRITON_PRINT_AUTOTUNING=1 python my_kernel.py

# Debug register pressure
TRITON_DUMP_PTXAS_LOG=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py

# Inspect MLIR passes for a specific kernel
MLIR_ENABLE_DUMP=_my_kernel TRITON_ALWAYS_COMPILE=1 python my_kernel.py

# Full IR pipeline dump
TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1 python my_kernel.py
```

## Reference files

- Full Python knobs: `python/triton/knobs.py`
- C++ env vars: `include/triton/Tools/Sys/GetEnv.hpp`
