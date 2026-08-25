# fbtriton

fbtriton is Meta's fork of [Triton](https://github.com/triton-lang/triton). It
tracks upstream closely and adds compiler and language work aimed at one thing:
giving kernel authors real control over warp-level execution on modern GPUs.

**Documentation lives on the project site:
[facebookexperimental.github.io/triton](https://facebookexperimental.github.io/triton/).**

That capability is offered at three levels, from "change nothing" to "write it
yourself":

| | You write | You get | Docs |
|---|---|---|---|
| **AutoWS** | ordinary Triton, plus `warp_specialize=True` on a loop | the compiler partitions the loop into producer/consumer warp groups | [Triton](https://facebookexperimental.github.io/triton/website/triton.html) |
| **torchTLX** | ordinary PyTorch | Inductor selects TLX-backed templates and fuses epilogues into them | [torchTLX](https://facebookexperimental.github.io/triton/website/torchtlx.html) |
| **TLX** | the kernel, explicitly | direct access to barriers, TMA, MMA, TMEM, clusters, warp specialization | [TLX](https://facebookexperimental.github.io/triton/website/tlx.html) |

Tracing, profiling, sanitizers, and benchmarking are covered under
[Tooling](https://facebookexperimental.github.io/triton/website/tooling.html).

Beyond warp specialization, the fork also adds **deterministic reductions** —
`reduction_ordering` on `tl.sum` / `tl.reduce` for bitwise-reproducible results
independent of `num_warps` and layout. See
[Reduction Ordering in Triton](third_party/tlx/doc/reduction_ordering.md).

## Install

fbtriton installs as the `triton` package — `import triton` will resolve to this
fork. **Uninstall upstream Triton first**; both distributions own the same
`triton/` directory, and installing one over the other leaves an inconsistent
environment rather than a clean error.

```bash
pip uninstall -y triton
pip install fbtriton
```

Nightly `.dev` wheels are published to a self-managed index (not PyPI):

```bash
pip install --pre fbtriton \
  --index-url https://facebookexperimental.github.io/triton/nightly/simple/
```

Each nightly is built from the newest `main` commit whose GPU/CI checks are all
green. `triton.__version__` reports `3.8.0.dev<YYYYMMDD>+fb.git<hash>`. Nightlies
are retained for ~30 days. Formal releases remain on PyPI. Binary wheels are
available for CPython 3.10-3.14.

**Compatibility.** fbtriton is intended as a drop-in replacement for upstream
Triton on a best-effort basis: it tracks upstream closely and existing Triton
kernels are expected to work unchanged. This is not a formal guarantee — if
something that works upstream breaks here, please file an issue.

## Build from source

```bash
git clone https://github.com/facebookexperimental/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e .
```

C++ changes require a rebuild to take effect; Python-only changes do not. Run
`pre-commit run --all` before sending a pull request.

## Hardware

| Vendor | Targets |
|--------|---------|
| NVIDIA | Hopper (`sm90`), Blackwell (`sm100`) |
| AMD | MI300 / CDNA3 (`gfx942`), MI350 / CDNA4 (`gfx950`), RDNA4 (`gfx1250`) |

Support varies per feature. Every operation on the
[TLX pages](https://facebookexperimental.github.io/triton/website/tlx.html) is
tagged with the targets it runs on, and AutoWS requires sm90 or newer.

## Testing

```bash
# TLX tutorial kernels (arch-gated; irrelevant cases skip)
pytest third_party/tlx/tutorials/testing/test_correctness.py

# TLX language tests
pytest python/test/unit/language/test_tlx_*.py

# AutoWS
pytest python/test/unit/language/test_autows_*.py

# torchTLX
pytest python/test/unit/language/test_torchtlx_*.py
```

Performance scripts and per-target invocations are on
[Build, install, and test](https://facebookexperimental.github.io/triton/website/install-and-test.html).

## CI

GPU test workflows run on push, pull request, and a nightly schedule; nightly
failures are filed as issues automatically.

| Workflow | Runner | Jobs |
|----------|--------|------|
| [`b200.yml`](.github/workflows/b200.yml) | `nvidia-dgx-b200` | `b200-meta-triton-test`, `b200-tlx-test`, `b200-gluon-test` |
| [`h100.yml`](.github/workflows/h100.yml) | `linux-gcp-h100` | `h100-meta-triton-test`, `h100-tlx-test` |
| [`mi350.yml`](.github/workflows/mi350.yml) | `linux-fb-triton-mi350-1` (gfx950 / CDNA4) | `mi350-meta-triton-test`, `mi350-tlx-test`, `mi350-gluon-test` |
| [`torchtlx.yml`](.github/workflows/torchtlx.yml) | `nvidia-dgx-b200`, `linux-fb-triton-mi350-1` | `b200-torchtlx-test`, `mi350-torchtlx-test` |

[`compiler.yml`](.github/workflows/compiler.yml) runs the lit suite on CPU;
[`ci.yml`](.github/workflows/ci.yml) runs integration tests and pre-commit.
Nightly failures are triaged by
[`report-nightly-failure.yml`](.github/workflows/report-nightly-failure.yml),
[`reconcile-nightly-issues.yml`](.github/workflows/reconcile-nightly-issues.yml),
and [`bisect-nightly-failure.yml`](.github/workflows/bisect-nightly-failure.yml).

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

## Editing the site

The site is generated, not hand-written. Page bodies live in
[`website/content/`](website/content/) (the TLX and torchTLX reference) and
[`website/guide_content.py`](website/guide_content.py) (curated guides). After
editing either, regenerate and commit the HTML:

```bash
cd website && python3 generate_tlx_site.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bugs and feature requests for TLX,
AutoWS, and torchTLX belong here; anything in core Triton or Gluon belongs
[upstream](https://github.com/triton-lang/triton).

## Further reading

- [TLX paper](https://arxiv.org/abs/2605.10905)
- [TLX talk, 2025 Triton Developer Conference](third_party/tlx/doc/TLX-triton-conference.pdf)
- [TLX talk, 2026 GPU Mode](third_party/tlx/doc/PerformanceOptimizationWithTLX.pdf)
- [Barrier support in TLX](third_party/tlx/doc/tlx_barriers.md)

## License

MIT — see [LICENSE](LICENSE).
