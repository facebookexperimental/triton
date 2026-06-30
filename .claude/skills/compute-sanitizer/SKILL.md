---
name: compute-sanitizer
description: >
  Run NVIDIA compute-sanitizer (memcheck, racecheck, initcheck, synccheck)
  against a Triton/TLX kernel to find runtime memory and synchronization bugs.
  Use when a kernel produces wrong results, crashes with an illegal/misaligned
  access, or is suspected of a shared-memory data race or invalid barrier usage
  — especially warp-specialized (WS) kernels using mbarriers, named barriers,
  TMA copies, or MMA accumulators. This is a runtime check: it runs the real
  kernel via its reproduce command, so it needs a working GPU and is 10-100x
  slower than a normal run.
---

# Compute Sanitizer

`compute-sanitizer` is NVIDIA's runtime correctness checker. It instruments the
real kernel launch and reports memory and synchronization errors that static IR
analysis cannot see. Use it to confirm (or rule out) out-of-bounds accesses,
uninitialized reads, shared-memory data races, and illegal barrier usage in
Triton/TLX kernels.

It complements the static investigations: `barrier-visualization` reasons about
the *intended* barrier protocol from the IR, while `compute-sanitizer` observes
what *actually* happens at runtime. When a race or sync error fires, cross-check
the two.

## The four tools

Select with `--tool`; `memcheck` is the default.

- **memcheck** — out-of-bounds and misaligned global/local/shared accesses, plus
  device-side allocation/leak errors. First tool to run; catches the classic
  TMA-past-end and bad-index bugs.
- **racecheck** — hazards on **shared memory** (RAW / WAR / WAW). Highly relevant
  to WS producer/consumer kernels that stage data through SMEM buffers. Does NOT
  check global-memory races.
- **synccheck** — illegal use of barriers and synchronization primitives
  (divergent `bar.sync`, mismatched arrive/wait counts, invalid cluster/async
  barrier usage). Directly relevant to mbarrier / named-barrier WS code.
- **initcheck** — reads of uninitialized **global** memory. Useful when output
  has nondeterministic garbage rather than a hard crash.

## Setup

1. **Locate the binary.** It is often not on PATH. Prefer
   `$COMPUTE_SANITIZER_BIN` if set, else `command -v compute-sanitizer`, else
   `/usr/local/cuda/bin/compute-sanitizer`, else the newest
   `/usr/local/cuda-*/bin/compute-sanitizer`.
2. **Get the reproduce command** (a `pytest`/`python` invocation). Use the
   smallest failing shape/config — sanitizer overhead makes large grids
   impractical. Prefer a single test id over a whole file.
3. **Pin a healthy GPU.** Run `bash third_party/tlx/find_working_gpu.sh`, take a
   `WORKING_GPUS` index, and prepend `CUDA_VISIBLE_DEVICES=<idx>`. See the
   `debug-failing-gpu` skill. If a run hangs past your timeout, run
   `third_party/tlx/killgpu.sh`.
4. **Enable source mapping.** Triton emits line info by default; keep it on
   (do not set `TRITON_DISABLE_LINE_INFO=1`) so errors map to `file:line`. Pass
   `--show-backtrace yes` for host+device backtraces.

## Invocation

General form (the sanitizer wraps the whole command, env vars go before it):

```bash
CUDA_VISIBLE_DEVICES=<idx> \
  <compute-sanitizer> --tool memcheck \
    --error-exitcode 1 \
    --show-backtrace yes \
    --log-file <out_dir>/memcheck.log \
    --target-processes all \
    python -m pytest -s -x "<test_id>"
```

Useful flags:

- `--error-exitcode 1` — return non-zero when any error is found (scripting).
- `--log-file <path>` — capture the report (`%p` expands to PID if needed).
- `--target-processes all` — follow child processes (pytest workers, subprocs).
- `--kernel-name-exclude` / `--kernel-name <regex>` — filter to the Triton
  kernel (names look like `_attn_fwd_...`, `matmul_kernel`, etc.) to cut noise.
- `--launch-timeout <s>` — bound a single launch.
- memcheck: `--leak-check full`, `--padding <bytes>` (catch off-by-a-few OOB).
- racecheck: `--racecheck-report all` (hazards + analysis).
- `--print-limit 0` — do not truncate the error list while triaging.

Run the tools in order — **memcheck → racecheck → synccheck → initcheck** — each
into its own log. Stop early only with a stated reason (e.g. memcheck already
found the crashing OOB, or an earlier tool hung).

## Triton/TLX specifics

- **First run JIT-compiles** the kernel (slow, unrelated to sanitizer). Let the
  compile complete; do not mistake compile time for a hang.
- **racecheck is shared-memory only.** WS kernels stage operands through SMEM, so
  a missing producer/consumer barrier shows up here as a WAR/RAW hazard on the
  SMEM buffer. A global-memory race will NOT be reported.
- **synccheck + mbarriers.** Invalid named-barrier / mbarrier / cluster-barrier
  usage (e.g. wrong arrive count, divergent participation) surfaces here. Pair
  with `barrier-visualization` to map the offending barrier to a partition.
- **TMA / async copies.** Out-of-bounds TMA tile accesses crash with an illegal
  instruction at runtime rather than masking — memcheck localizes the launch;
  also see the `tma-illegal-instruction` skill for the structural launcher-bug
  pattern.
- **Possible noise.** Some async/TMA/`cp.async` paths may emit benign warnings or
  unsupported-feature notes on certain driver/toolkit versions. Note them as
  caveats; do not treat a warning as a confirmed bug without corroboration.

## Interpreting output

- Each tool ends with `========= ERROR SUMMARY: N errors`. `0 errors` = clean for
  that tool. racecheck prints `RACECHECK SUMMARY`.
- A real finding includes the access kind (e.g. *Invalid __global__ read of size
  16*), the address, the kernel name, and — with line info — the `file:line`.
  Capture the **first** error; later ones are often cascades.
- "Clean" is a result, not a failure. Report it plainly: a clean memcheck +
  racecheck + synccheck materially narrows a wrong-result bug toward
  compiler/logic rather than memory/sync.

## Reporting

Return a compact matrix plus triage:

```text
tool       | exit | errors | first finding (kernel @ file:line) | mapped?
memcheck   |  0   |   0    | -                                  | n/a
racecheck  |  1   |   3    | WAR hazard on smem buf @ k.py:142  | yes
synccheck  |  0   |   0    | -                                  | n/a
initcheck  |  0   |   0    | -                                  | n/a
```

Then state the most likely root cause (OOB vs. shared-memory race vs.
uninitialized read vs. illegal sync), the implicated TLX/Triton construct, and
the next action (e.g. narrow with a kernel filter, or hand a race/sync hit to
`barrier-visualization`).

Do not run performance benchmarks.
