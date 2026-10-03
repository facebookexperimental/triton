# Copilot Code Review Instructions — facebookexperimental/triton

These instructions guide GitHub Copilot's automated PR review for this repo (a
Meta fork of Triton). They are a **free, diff-only baseline**. The deliberate,
GPU-aware review lives in `.claude/reviewers/` and runs on `/claude review`.

Keep reviews concise. Report findings as bullet points with `file:line`
references. If a section has no issues, say nothing about it.

## 1. New tests must declare a hardware gate

When a PR **adds or renames a test** (a `def test_*` function, a
`pytest.mark.parametrize` case, or a `pytestmark` assignment) under any test
path — e.g. `python/test/unit/**`, `third_party/tlx/tutorials/testing/**`, or
any `*_test.py` — check that it is properly gated to the hardware it requires.

Flag a new test as **missing a hardware skip** when it exercises
backend/architecture-specific behavior but has **no** guarding `skipif` or
architecture branch. The canonical gates in this repo come from
`triton._internal_testing`:

- Vendor: `is_cuda()`, `is_hip()`
- NVIDIA arch: `is_hopper()`, `is_hopper_or_newer()`, `is_blackwell()`,
  `is_hopper_or_blackwell()`, or
  `torch.cuda.get_device_capability()[0]` checks (e.g. `== 10` for Blackwell,
  `>= 9` for Hopper+)
- AMD arch: `is_hip_cdna3()`, `is_hip_cdna4()`, `is_hip_gfx1250()`,
  `is_hip_rdna4()`, etc.
- Feature: `requires_tma`, and similar markers

Typical correct forms:

```python
@pytest.mark.skipif(not is_hip(), reason="Requires HIP runtime")
@pytest.mark.skipif(is_blackwell(), reason="Not tested on Blackwell")
@pytest.mark.skipif(is_hip(), reason="warp specialization is not supported on hip devices")
@pytest.mark.skipif(is_cuda() and torch.cuda.get_device_capability()[0] != 10,
                    reason="Requires compute capability == 10")
```

Raise a finding when, for example:

- A test uses TMA / `tcgen05` / Blackwell-only TMEM or MMA features but is not
  gated to `is_blackwell()` (or capability `== 10`).
- A test uses `wgmma` / Hopper warp specialization but is not gated to
  `is_hopper_or_newer()`.
- A test is AMD-specific (HIP intrinsics, CDNA/RDNA paths) but is not gated to
  `is_hip()` / the relevant `is_hip_*` arch, or conversely a CUDA-only test is
  not excluded on HIP via `is_hip()`.

Do **not** flag tests that are intentionally vendor-agnostic (pure
host/IR/lit-style logic with no device execution) — those need no gate.
When a `reason=` string is missing on a `skipif`, note it as a minor nit.

## 2. Keep PRs small and focused

Prefer small, single-purpose PRs. When a PR mixes unrelated concerns, call it
out and suggest splitting. Signals that a PR is doing too much:

- Unrelated changes bundled together (e.g. a bug fix + a refactor + a new
  feature, or edits across several independent subsystems with no shared
  reason).
- Drive-by reformatting, renames, or whitespace churn mixed into a logic change
  — these bury the real diff and should be a separate PR.
- A diff large enough that a reviewer cannot reasonably reason about it in one
  pass; recommend breaking it into a reviewable sequence.

Keep this advisory and brief: when the PR is already tight and single-purpose,
say nothing. The goal is reviewability, not a hard size limit.

## 3. New TLX tutorial kernels must follow the reference structure and be tested

When a PR adds a **new TLX tutorial kernel** (a new `*.py` under
`third_party/tlx/tutorials/`), check that it follows the structure of the
Blackwell GEMM reference kernels (`blackwell_gemm_ws.py`,
`blackwell_gemm_clc.py`, `blackwell_gemm_pipelined.py`, `blackwell_gemm_2cta.py`)
and is wired into **both** the correctness suite and a perf benchmark. Flag any
of the following as missing:

- **Public entrypoint** — a host wrapper with the same shape as the reference
  (e.g. a `matmul(...)` / equivalent function), an autotune config helper, and
  `@triton.jit` device functions — rather than ad-hoc top-level code.
- **Correctness wiring** — the kernel is imported into
  `third_party/tlx/tutorials/testing/test_correctness.py`
  (`from triton.language.extra.tlx.tutorials.<name> import ... as _<name>`),
  registered in the relevant `CONFIGS` table, and has a `test_<name>` that runs
  it against the PyTorch reference (parametrized over dtypes, e.g. fp16/bf16),
  matching how `test_blackwell_gemm_ws` is set up.
- **Perf wiring** — the kernel is added to the matching perf harness (e.g.
  `testing/test_blackwell_gemm_perf.py` for GEMM, `test_blackwell_fa_perf.py`
  for FA): imported, listed among the benchmarked providers/versions, and
  compared against the reference library via `triton.testing.perf_report`.
- **Hardware gate** — both the new tests are gated to the kernel's target arch
  (`is_blackwell()`, `is_hopper_or_newer()`, the relevant `is_hip_*`, …), per
  rule 1.

A new kernel added without a correctness test, without a perf test, or that
diverges from the reference kernel layout should be called out. Do not flag
edits to existing kernels under this rule — it applies to newly added kernels.
