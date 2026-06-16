---
name: debug-failing-gpu
description: >
  Recover from GPU-busy / GPU-unavailable failures. Use when a command (pytest,
  python, a TLX/Triton kernel run, a benchmark) fails with errors indicating the
  GPU is busy, out of memory, or unavailable — e.g. "CUDA error: out of memory",
  "all CUDA-capable devices are busy or unavailable", "CUDA-capable device(s) is/are
  busy or unavailable", "RuntimeError: No CUDA GPUs are available", "device-side
  assert", or a hang on the first CUDA call. Runs find_working_gpu.sh to locate a
  healthy GPU and re-runs the failed command pinned to it via CUDA_VISIBLE_DEVICES.
---

# Debug Failing GPU

A command failed because the GPU it landed on is busy, out of memory, or in a bad
state. Find a GPU that actually works and re-run the command pinned to it.

## When to trigger

Any failure whose root cause is the *device*, not the code. Common signatures:

- `CUDA error: out of memory` / `torch.cuda.OutOfMemoryError`
- `all CUDA-capable devices are busy or unavailable`
- `CUDA-capable device(s) is/are busy or unavailable`
- `RuntimeError: No CUDA GPUs are available`
- `CUDA error: device-side assert triggered`
- A kernel/test that hangs on the first CUDA call

Do **not** use this for kernel logic bugs, compilation errors, or numerical
mismatches — those are not device-health problems.

## Procedure

1. Run the scanner:
   ```bash
   bash third_party/tlx/find_working_gpu.sh
   ```
2. Read the final line, `WORKING_GPUS=...` (e.g. `WORKING_GPUS=0,2,3`). These are
   **physical** GPU indices.
3. Pick the first working index. If several are free, mention them so the user can
   parallelize across GPUs.
4. Re-issue the original failing command with `CUDA_VISIBLE_DEVICES=<idx>`:
   - If the command had **no** `CUDA_VISIBLE_DEVICES`, prepend one.
   - If it **already set** `CUDA_VISIBLE_DEVICES`, **replace** that value — do not
     stack two assignments.

## If no GPU works

`WORKING_GPUS=` is empty. The GPUs may be held by your own stuck processes:

1. Clear them: `third_party/tlx/killgpu.sh`
2. Re-run `bash third_party/tlx/find_working_gpu.sh`.
3. If still empty, all GPUs are occupied by other users — report that and stop;
   there is nothing to switch to.

## Command-rewrite examples

```bash
# No device set -> prepend
pytest third_party/tlx/tutorials/testing/test_correctness.py
# becomes
CUDA_VISIBLE_DEVICES=2 pytest third_party/tlx/tutorials/testing/test_correctness.py

# Device already set -> replace, don't stack
CUDA_VISIBLE_DEVICES=4 third_party/tlx/denoise.sh python bench.py
# becomes
CUDA_VISIBLE_DEVICES=2 third_party/tlx/denoise.sh python bench.py
```

Note: `denoise.sh` defaults to device 4 when `CUDA_VISIBLE_DEVICES` is unset
(`third_party/tlx/denoise.sh:6`), so always set it explicitly when wrapping a
benchmark with `denoise.sh` after a failure.
