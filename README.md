# fbtriton

fbtriton is Meta's downstream fork of
[Triton](https://github.com/triton-lang/triton), consolidating GPU compiler and
DSL innovations we develop for our own workloads while keeping the delta from
upstream as small as possible. It is continuously synchronized with upstream and
powers GPU training and inference workloads across Meta's services.

## Components

| | | Docs |
|---|---|---|
| **TLX** | A low-level, warp-aware, hardware-near extension of the Triton DSL | [docs/tlx.md](docs/tlx.md) |
| **AutoWS** | A compiler optimization that partitions a kernel's operations into specialized warp groups | [docs/compiler.md](docs/compiler.md) |
| **torchTLX** | TLX primitives pushed into Inductor's template and fusion infrastructure | [docs/torchtlx.md](docs/torchtlx.md) |

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

Support varies per feature: each operation in [docs/tlx.md](docs/tlx.md) carries
a hardware tag, and [AutoWS](docs/compiler.md) requires sm90 or newer.

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

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Further reading

- [FBTriton infra: upstream ingestion, hierarchical validation — ideals vs. realities](https://pytorch.org/blog/fbtriton-infra-upstream-ingestion-hierarchical-validation-ideals-vs-realities/)
- [TLX paper](https://arxiv.org/abs/2605.10905)
- [TLX talk, 2025 Triton Developer Conference](third_party/tlx/doc/TLX-triton-conference.pdf)
- [TLX talk, 2026 GPU Mode](third_party/tlx/doc/PerformanceOptimizationWithTLX.pdf)
- [Barrier support in TLX](third_party/tlx/doc/tlx_barriers.md)

## License

MIT — see [LICENSE](LICENSE).
