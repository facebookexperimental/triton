# fbtriton

fbtriton is Meta's fork of [Triton](https://github.com/triton-lang/triton). It
tracks upstream closely and adds compiler and language work aimed at one thing:
giving kernel authors real control over warp-level execution on modern GPUs.

That capability is offered at three levels, from "change nothing" to "write it
yourself":

| | You write | You get | Docs |
|---|---|---|---|
| **AutoWS** | ordinary Triton, plus `warp_specialize=True` on a loop | the compiler partitions the loop into producer/consumer warp groups | [docs/compiler.md](docs/compiler.md) |
| **torchTLX** | ordinary PyTorch | Inductor selects TLX-backed templates and fuses epilogues into them | [docs/torchtlx.md](docs/torchtlx.md) |
| **TLX** | the kernel, explicitly | direct access to barriers, TMA, MMA, TMEM, clusters, warp specialization | [docs/tlx.md](docs/tlx.md) |

Beyond warp specialization, the fork also adds **deterministic reductions** —
`reduction_ordering` on `tl.sum` / `tl.reduce` for bitwise-reproducible results
independent of `num_warps` and layout. See
[docs/compiler.md](docs/compiler.md#reduction-ordering).

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
are retained for ~30 days. Formal releases remain on PyPI.

Binary wheels are available for CPython 3.10-3.14.

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
`pre-commit run --all` before sending a PR.

## Hardware

| Vendor | Targets |
|--------|---------|
| NVIDIA | Hopper (`sm90`), Blackwell (`sm100`) |
| AMD | MI300 / CDNA3 (`gfx942`), MI350 / CDNA4 (`gfx950`), RDNA4 (`gfx1250`) |

Support varies per feature — each operation in [docs/tlx.md](docs/tlx.md) is
tagged with the targets it runs on, and [AutoWS](docs/compiler.md) requires sm90 or
newer.

## Also in this repo

**Gluon** (`python/triton/experimental/gluon/`) is upstream-synced and is **not**
a first-class DSL here — send Gluon feature work and bug fixes upstream, not to
this fork. We run fundamental Gluon CI so it does not silently break; see
[docs/ci.md](docs/ci.md).

## Testing and CI

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

Workflows, runners, nightly failure handling, and per-project CI coverage are
documented in [docs/ci.md](docs/ci.md).

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
