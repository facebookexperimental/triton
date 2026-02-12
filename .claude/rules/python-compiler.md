---
globs:
  - "python/triton/**"
---

# Triton Python Compiler

Python-only: no rebuild needed.

## Key files
- Compiler pipeline: `python/triton/compiler/`
- Tuning knobs: `python/triton/knobs.py`
- Env vars recognized in C++: `include/triton/Tools/Sys/GetEnv.hpp`
