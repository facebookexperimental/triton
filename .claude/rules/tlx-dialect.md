---
globs:
  - "third_party/tlx/dialect/**"
---

# TLX Dialect (C++ / TableGen)

MUST rebuild after changes: `pip install -e . --no-build-isolation`

## Structure
- Backend registration: `third_party/tlx/dialect/triton_tlx.cc`
- TableGen files (`*.td`) define ops; C++ files implement them
- Op definitions: `third_party/tlx/dialect/include/IR/TLXOps.td`
- Transforms: `third_party/tlx/dialect/lib/Transforms/`

## Testing
- LIT tests in `test/`
- Correctness: `pytest third_party/tlx/tutorials/testing/test_correctness.py`
