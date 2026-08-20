# TLX - Triton Low-level Language Extensions

## Introduction

TLX (Triton Low-level Language Extensions) is a low-level, warp-aware, hardware-near extension of the Triton DSL. It offers intrinsics and warp-specialized operations for fine-grained GPU control, hardware-oriented primitives for advanced kernel development, and explicit constructs for GPU memory, computation, and asynchronous control flow. TLX is designed for expert users pushing Triton closer to the metal.

Primarily targeting NVIDIA GPUs (for now), TLX extends Triton to support:

- Hardware-specific intrinsics (e.g., wgmma, async_copy, barrier)
- Shared and local memory allocation
- Instruction-level scheduling and control
- Cross-warpgroup synchronization


While this approach places more responsibility on the user, it reduces the compiler's role as a performance bottleneck. Although it may introduce divergence across hardware platforms, it empowers users to perform deeper, architecture-specific optimizations without relying solely on compiler heuristics.


## Nightly builds (fbtriton)

Nightly `.dev` wheels are published to a self-managed index (not PyPI):

    pip install --pre fbtriton \
      --index-url https://facebookexperimental.github.io/triton/nightly/simple/

Each nightly is built from the newest `main` commit whose GPU/CI checks are all
green. `triton.__version__` reports `3.8.0.dev<YYYYMMDD>+fb.git<hash>`. Nightlies
are retained for ~30 days. Formal releases remain on PyPI (`pip install fbtriton`).


## Gluon support

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
