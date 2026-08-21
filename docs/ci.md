# CI

[fbtriton](../README.md) &rsaquo; CI

All workflows live in [`.github/workflows/`](../.github/workflows/). The
GPU test workflows run on push, pull request, and a nightly schedule; nightly
failures are filed as issues automatically.

## GPU test workflows

| Workflow | Runner | Jobs |
|----------|--------|------|
| [`b200.yml`](../.github/workflows/b200.yml) | `nvidia-dgx-b200` | `b200-meta-triton-test`, `b200-tlx-test`, `b200-gluon-test` |
| [`h100.yml`](../.github/workflows/h100.yml) | `linux-gcp-h100` | `h100-meta-triton-test`, `h100-tlx-test` |
| [`mi350.yml`](../.github/workflows/mi350.yml) | `linux-fb-triton-mi350-1` (gfx950 / CDNA4) | `mi350-meta-triton-test`, `mi350-tlx-test`, `mi350-gluon-test` |
| [`torchtlx.yml`](../.github/workflows/torchtlx.yml) | `nvidia-dgx-b200`, `linux-fb-triton-mi350-1` | `b200-torchtlx-test`, `mi350-torchtlx-test` |

What each job class runs:

- **`*-meta-triton-test`** — TritonBench performance coverage, against
  [meta-pytorch/tritonbench](https://github.com/meta-pytorch/tritonbench).
  Skips are listed in `.ci/tritonbench/fbtriton_skip_tests.yaml`.
- **`*-tlx-test`** — TLX unit tests (`python/test/unit/language/test_tlx_*.py`),
  the tutorial correctness suite
  (`third_party/tlx/tutorials/testing/test_correctness.py`), and runtime unit
  tests. The `test_tlx_*.py` glob covers manual TLX (`tlx.async_tasks`) and
  structurally excludes the autoWS suites, which are tested separately.
- **`*-gluon-test`** — `pytest python/test/gluon/` (see [Gluon](#gluon) below).
- **`*-torchtlx-test`** — `test_torchtlx_templates.py` and
  `test_torchtlx_fusions.py`, against a nightly PyTorch.

## Compiler tests

[`compiler.yml`](../.github/workflows/compiler.yml) runs the lit test suite
(`lit-tests`) on `ubuntu-latest` — no GPU required. This is where compiler-side
[AutoWS](compiler.md) coverage lives.

[`ci.yml`](../.github/workflows/ci.yml) runs integration tests and pre-commit.

## Pre-commit

[`pre-commit.yml`](../.github/workflows/pre-commit.yml) enforces formatting;
[`pre-commit-autofix.yml`](../.github/workflows/pre-commit-autofix.yml) can apply
fixes on demand. Run it locally before pushing:

```bash
pre-commit run --all
```

## Nightlies and releases

| Workflow | Purpose |
|----------|---------|
| [`nightly-fbtriton.yml`](../.github/workflows/nightly-fbtriton.yml) | Build and publish nightly `.dev` wheels |
| [`wheels_fb.yml`](../.github/workflows/wheels_fb.yml) | Build fbtriton wheels (`TRITON_WHEEL_NAME=fbtriton`) |
| [`publish_fbtriton.yml`](../.github/workflows/publish_fbtriton.yml) | Publish to PyPI |
| [`create_release.yml`](../.github/workflows/create_release.yml) | Cut a release |
| [`release-testing.yml`](../.github/workflows/release-testing.yml) | Release validation |
| [`llvm-build.yml`](../.github/workflows/llvm-build.yml) | Build the pinned LLVM |
| [`documentation.yml`](../.github/workflows/documentation.yml) | Build docs |

## Nightly failure handling

Scheduled runs parse their JUnit output with
`.github/scripts/parse_junit_failures.py` and bucket failures per job. Three
workflows then act on that:

| Workflow | Purpose |
|----------|---------|
| [`report-nightly-failure.yml`](../.github/workflows/report-nightly-failure.yml) | File an issue for a nightly failure |
| [`reconcile-nightly-issues.yml`](../.github/workflows/reconcile-nightly-issues.yml) | Close issues that no longer reproduce |
| [`bisect-nightly-failure.yml`](../.github/workflows/bisect-nightly-failure.yml) | Bisect a nightly failure (manual dispatch) |

## TLX-AMD CI

AMD tutorial kernels are exercised by
[`mi350.yml`](../.github/workflows/mi350.yml) on a gfx950 (MI350 / CDNA4)
runner, mirroring the H100 job in [`h100.yml`](../.github/workflows/h100.yml):

- **`mi350-tlx-test`** — TLX unit tests (`python/test/unit/language/test_tlx_*.py`)
  plus the tutorial correctness suite
  (`third_party/tlx/tutorials/testing/test_correctness.py`). AMD and IKBO cases
  run; Hopper/Blackwell and gfx1250 cases auto-skip via the arch gates.
- **`mi350-meta-triton-test`** — TritonBench performance coverage (the AMD perf
  scripts in [TLX &rsaquo; Testing](tlx.md#testing) are for local runs;
  perf-regression tracking lives in TritonBench).

Both run on push, PR, and the nightly schedule; nightly failures are filed as
issues via `report-nightly-failure.yml`.

## Gluon

[Gluon](https://github.com/triton-lang/triton/tree/main/python/triton/experimental/gluon)
(`python/triton/experimental/gluon/`) is **upstream-synced and not a first-class DSL** here
(TLX is the focus) — do Gluon feature/bug work upstream, not in this fork. But since fbtriton
is a secondary Triton, we run **fundamental Gluon CI** so we don't silently break it.

**CI:** the `b200-gluon-test` / `mi350-gluon-test` jobs run `pytest python/test/gluon/` (every
`test_*.py`). It is **green**: ~200 passed, ~1600 skipped, 0 failed. The real signal is the
compile-only, target-agnostic frontend suite (`test_frontend.py`, one run covers NVIDIA + AMD
codegen via mock `GPUTarget`); the version-skewed cases below are skipped via
`python/test/gluon/conftest.py` rather than failing the job.

**Fork-side fixes** (Gluon frontend itself unmodified): `core.py` reduce `reduction_ordering`
compat, `semantic.py` `dot()` `allow_tf32` default, a `test_core.py` collection fix (a bad
cherry-pick left an `IndentationError`), and regenerated `test_frontend.py` goldens
(upstream-synced — overwritten on next sync).

**Skipped, needs upstream Gluon re-sync — TODO(gluon-ci):** the GPU-execution suites
(`test_core`, `test_lowerings`, `test_consan`, `test_fpsan`, one kernel in
`test_layout_format_view`) were synced from much newer upstream (e.g. `test_fpsan.py` via bundle
#1956) than the pinned Gluon frontend (`_semantic.py` ~2026-06-29). They require Gluon behavior
the frontend doesn't have yet — raw pointer `gl.load`/`gl.store` inferring a *distributed* layout
— so they fail wholesale with `expected ... distributed_type but got block_type`. The fix is to
sync the Gluon frontend forward (upstream cherry-pick / re-sync), not a local patch. Also skipped:
~9 frontend per-target golden tests (single inline golden can't match every parametrized target)
and the `create_lds_barrier_wait` pybind mismatch. `conftest.py` lists these; remove/trim it after
the re-sync.
