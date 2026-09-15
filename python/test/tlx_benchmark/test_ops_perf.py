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

    fwd_only, bwd_only = pytestconfig.getoption("--fwd-only"), pytestconfig.getoption("--bwd-only")
    if fwd_only and bwd_only:
        raise pytest.UsageError("--fwd-only and --bwd-only are mutually exclusive")

    results, env = bench.run(
        space=pytestconfig.getoption("--space"),
        head=pytestconfig.getoption("--head"),
        synthetic=pytestconfig.getoption("--synthetic"),
        cold_compile_mode=pytestconfig.getoption("--cold-compile"),
        latency_mode=pytestconfig.getoption("--latency-measure-mode"),
        directions=("fwd", ) if fwd_only else ("bwd", ) if bwd_only else None,
    )
    if not results:
        pytest.skip(f"{bench.OP} has no cases matching the current filters")
    from _harness import report as report_mod

    print("\n" + report_mod.render(results, env,
                                   pytestconfig.getoption("--json") or bench.default_json(),
                                   getattr(bench, "EXTRA_COLUMNS", ())))

    bad = report_mod.failures(results)
    assert not bad, "\n".join(f"{r.case.key}: {'; '.join(r.notes)}" for r in bad)
