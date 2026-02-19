---
globs:
  - "third_party/tlx/language/**"
---

# TLX Python DSL

Python-only: no rebuild needed.

## Testing
- `pytest third_party/tlx/tutorials/testing/test_correctness.py`

## API reference
For a curated cheatsheet of all TLX primitives (barriers, memory ops, TMA, MMA,
CLC, warp specialization), use the `tlx-api-reference` skill.

## Deep-dive docs
- Full API reference: `third_party/tlx/README.md`
- Barriers: `third_party/tlx/doc/tlx_barriers.md`
- Placeholder layouts: `third_party/tlx/doc/PlaceholderLayouts.md`
- Storage alias design: `third_party/tlx/doc/storage_alias_spec_design.md`
