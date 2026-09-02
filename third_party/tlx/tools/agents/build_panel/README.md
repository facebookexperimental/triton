# Build panel

This panel inherits [the general panel](../general_panel/README.md). It owns rebuilding
and correctness validation, not performance conclusions.

## 1. Rebuild policy

- After any native or compiler change, run `make` from the Triton repository
  root before testing or benchmarking. A worker must not evaluate a compiler
  finding against a stale build.
- Do not rebuild for Python-only changes or changes under
  `python/triton_kernels`.
- `pre-commit run --all` is not a required build or validation step. Run focused
  formatting, lint, syntax, and test commands appropriate to the changed files.

## 2. Test policy

- Run pytest with `-s --tb=short`. Select one test as
  `pytest path/to/file.py::test_name`.
- Keep GPU-only tests under `python/test/unit/` or `python/test/gluon/`, name
  them `test_<feature>_<condition>`, and avoid creating a new test file unless
  requested.
- For lit tests, compute `BUILD_DIR` with `build_helpers`, build `triton-opt`,
  then run `lit -v test/<path>.mlir` from that directory.
- When a compiler crash emits an MLIR reproducer, preserve the complete IR and
  metadata and run `triton-opt <file>.mlir --run-reproducer`.

The build report records the exact commands and statuses for the manager. A
successful build or test does not itself establish a performance conclusion.

The panel's `test_*.py` files and `testdata/` fixtures validate the executable
kernel-optimization infrastructure. Run them with:

```bash
python -m pytest third_party/tlx/tools/agents/build_panel/test_*.py -s --tb=short
```
