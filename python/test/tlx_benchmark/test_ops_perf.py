"""pytest front end over every ``bench_<op>.py`` in this directory.

One generic discoverer rather than a shim per op, so "one benchmark file per
op" stays literally true. The CLI in each ``bench_<op>.py`` is the primary
interface -- this exists for the junitxml that the b200 reporting pipeline
already consumes.

Every option is shared with that CLI (see ``conftest.py``), so the two front
ends cannot drift into different behaviour.
"""

import importlib
import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

BENCH_MODULES = sorted(p.stem for p in _HERE.glob("bench_*.py"))


@pytest.mark.parametrize("module_name", BENCH_MODULES)
def test_op_perf(module_name, pytestconfig):
    bench = importlib.import_module(module_name)
    if not bench.supported():
        pytest.skip(f"{bench.OP} has no implementation for this device")

    kwargs = {}
    if pytestconfig.getoption("--replicates"):
        kwargs["replicates"] = pytestconfig.getoption("--replicates")
    results, env = bench.run(
        space=pytestconfig.getoption("--space"),
        measure_compile=pytestconfig.getoption("--measure") in ("compile", "all"),
        strict=pytestconfig.getoption("--strict-env"),
        **kwargs,
    )
    from _harness import baseline as baseline_mod
    from _harness import report as report_mod

    print("\n" + report_mod.render(results, env, pytestconfig.getoption("--json") or bench.DEFAULT_JSON))

    if pytestconfig.getoption("--update-baseline"):
        print(f"baseline written: {baseline_mod.save(bench.OP, bench.ARCH, results, env)}")
        return
    if pytestconfig.getoption("--guard") == "enforce":
        bad = report_mod.failures(results)
        assert not bad, "\n".join(f"{r.case.key}: {'; '.join(r.notes)}" for r in bad)
