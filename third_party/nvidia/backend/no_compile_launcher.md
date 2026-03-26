# No-Compile Launcher (`TRITON_USE_NO_COMPILE_LAUNCHER`)

## What It Is

The no-compile launcher is a pure-Python ctypes-based alternative to Triton's
default C-compiled kernel launcher. Instead of generating C source code and
invoking `gcc -O3` to produce a shared library (`.so`) for each kernel, it
constructs the launch parameters in Python and calls `cuLaunchKernelEx` directly
via ctypes.

## Why It Exists

The `gcc -O3` compilation step for each kernel's launcher adds latency before
the first kernel launch. On cluster environments like GB300, this typically
takes 50-100ms per kernel, but under heavy CPU contention (where CPU cores are
shared across many processes), it can take up to ~50 seconds per kernel due to
resource contention as `gcc` competes for scarce CPU time. The ctypes launcher
eliminates this compilation entirely, replacing it with pure-Python argument
packing that completes in <1ms regardless of CPU load.

## Safety

The ctypes launcher is functionally equivalent to the C launcher:

- **Same CUDA API**: Both call `cuLaunchKernelEx` with the same `CUlaunchConfig`
  struct layout (grid dims, block dims, shared memory, launch attributes).
- **Same argument packing**: Pointer arguments go through the same
  `cuPointerGetAttribute` validation. Float arguments use the same
  pack-to-storage-type logic (fp16, bf16, fp32, fp64). Integer arguments are
  cast to the same ctypes widths. Tensor descriptor arguments (both host-side
  and TMA hardware descriptors) are expanded and passed identically.
- **Same launch attributes**: Cooperative grid, PDL (programmatic stream
  serialization), cluster dimensions, and cluster scheduling policy are set
  identically.
- **Same hook contract**: `launch_enter_hook` and `launch_exit_hook` are called
  at the same points.

## How to Enable

```bash
export TRITON_USE_NO_COMPILE_LAUNCHER=1
```

When the knob is unset or `0`, the default C-compiled launcher is used.

## Known Limitations

- **tuple signature arguments**: Not yet supported.

## Performance Characteristics

| Metric | C Launcher | ctypes Launcher |
|--------|-----------|-----------------|
| Launcher creation time (GB300, typical) | 50-100ms | <1ms |
| Launcher creation time (GB300, heavy CPU contention) | up to ~50s due to resource contention | <1ms |
| Kernel launch latency | Negligible | Negligible |
| Runtime correctness | Reference | Equivalent |
