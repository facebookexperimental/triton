---
name: amd-att-trace
description: Collect, validate, package, and inspect rocprofv3 Advanced Thread Trace bundles for AMD GPU kernels. Use for ATT collection, viewer-ready artifacts, or loop cycle windows; do not trigger for ordinary performance benchmarks.
---

# AMD Advanced Thread Trace

Use `scripts/att_trace.py` for the reproducible parts of ATT collection. Do not
recreate rocprofv3 commands or decide that a raw `.att` file is viewer-ready by
inspection alone.

## Authorization and scope

- Collect a new trace only when the user explicitly asks to run or profile a
  workload. Reading, validating, or packaging an existing trace is read-only.
- Check GPU occupancy before collection and choose an idle device. Do not stop
  another user's job or run `killgpu.sh` unless explicitly authorized.
- Keep ATT diagnostics separate from correctness and end-to-end performance
  measurements. A trace is not a correctness result or a stable benchmark.
- Never overwrite an existing non-empty output directory or archive.

## Workflow

1. Establish the application command, kernel regex, steady-state dispatch
   number, GPU, and output label. State any inferred dispatch numbering.
2. Run `att_trace.py probe`. Use a profiler that exposes `--att` and a matching
   decoder directory. Prefer a profiler and target application from compatible
   ROCm installations.
3. Use `collect --dry-run` when command selection or quoting is uncertain.
4. Collect exactly the requested dispatch. Use a unique output directory.
5. Treat collection as successful only when the automatic viewer-bundle
   validation passes.
6. Package the complete trace root with `package`; never transfer only the raw
   `.att` files.
7. Report the trace root, archive and checksum if packaged, selected profiler
   and decoder, kernel regex, dispatch number, and decoded UI directory.

## Commands

```bash
python scripts/att_trace.py probe

python scripts/att_trace.py collect \
  --output /tmp/my-att \
  --name my-kernel \
  --gpu 0 \
  --kernel-regex 'my_kernel' \
  --dispatch 6 \
  -- <application> <arguments>

python scripts/att_trace.py validate /tmp/my-att
python scripts/att_trace.py package /tmp/my-att

python scripts/att_trace.py window /tmp/my-att \
  --loop-pc 0x2388 --first 10 --last 14
```

Iteration windows are zero-based by default and end at the next loop-marker
occurrence after `--last`. Choose a loop marker that appears once per loop
iteration; barriers can produce multiple ATT events. Record unrolling such as
"one outer iteration represents two logical K tiles" instead of silently
equating outer-loop and logical-tile iterations.

## References

- Read [references/rocprofv3.md](references/rocprofv3.md) when choosing ATT
  options, a profiler installation, or a dispatch.
- Read [references/viewer-format.md](references/viewer-format.md) when
  validating decoded output, transferring a bundle, or deriving cycle windows.
