---
globs:
  - ".github/**"
---

# GPU CI and the CI health dashboard

Read `.github/GPU_CI.md` before changing GPU workflows. `gpu-tests.yml` is the
entry point; `h100.yml`, `b200.yml` and `mi350.yml` are reusable workflows it
calls.

The `/ci/` dashboard (`.github/scripts/ci_health*.py`) parses GPU job names,
step names and pytest output. When you rename a GPU job or test step, change
pytest reporting flags, or add a setup step, check the "CI health dashboard"
section of `GPU_CI.md` and run:

```bash
cd .github/scripts && python3 -m pytest -q test_ci_health.py test_ci_health_rules.py test_ci_health_logs.py
```
