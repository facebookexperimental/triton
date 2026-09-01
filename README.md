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
| **TorchTLX** | ordinary PyTorch | Inductor selects TLX-backed templates and fuses epilogues into them | [TorchTLX](https://facebookexperimental.github.io/triton/website/torchtlx.html) |
| **TLX** | the kernel, explicitly | direct access to barriers, TMA, MMA, TMEM, clusters, warp specialization | [TLX](https://facebookexperimental.github.io/triton/website/tlx.html) |

[Compiler features](https://facebookexperimental.github.io/triton/website/compiler.html)
covers AutoWS tuning and deterministic reductions;
[Tooling](https://facebookexperimental.github.io/triton/website/tooling.html)
covers tracing, profiling, sanitizers, and benchmarking;
[CI](https://facebookexperimental.github.io/triton/website/ci.html) covers
workflows, runners, and test coverage.

## Install

```bash
pip uninstall -y triton      # both distributions own the same triton/ directory
pip install fbtriton
```

Nightly wheels, source builds, supported hardware, and compatibility notes are
on the [project site](https://facebookexperimental.github.io/triton/).

### uTLX: standalone TLX for upstream Triton

TLX also ships standalone as
[`triton-utlx`](https://pypi.org/project/triton-utlx/), which provides the same
`tlx` module as a Triton plugin rather than as a fork of Triton:

```bash
pip install torch
pip install triton-utlx

export TRITON_PLUGIN_PATHS=$(python -c \
  "import utlx_plugin, os; print(os.path.join(os.path.dirname(utlx_plugin.__file__), 'libutlx.so'))")
```

Plugins load only into a Triton built with `TRITON_EXT_ENABLED`. The `triton`
that ships with a PyTorch release has it on by default; to use a Triton you
build yourself, enable it at build time from a checkout of
[upstream Triton](https://github.com/triton-lang/triton):

```bash
TRITON_EXT_ENABLED=ON pip install -e . --no-build-isolation
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for building from source, running the
test suites, and editing the documentation site. Bugs and feature requests for
TLX, AutoWS, and TorchTLX belong here; anything in core Triton or Gluon belongs
[upstream](https://github.com/triton-lang/triton).

## License

MIT — see [LICENSE](LICENSE).
