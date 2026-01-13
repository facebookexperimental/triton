# Key Concepts
Compilation Flow: Python DSL → TTIR (Triton IR) → TTGIR (Triton GPU IR) → LLVM IR → PTX/AMDGPU

# CRITICAL: Always rebuild after modifying C++ code:
```bash
pip install -e . --no-build-isolation
```
C++ changes require recompilation to take effect. Python-only changes do not.

# CRITICAL: Always run formatter after modifying code:
```bash
pre-commit run --all
```

# CRITICAL: Always run denoise script when measuring performance:
Run `third_party/tlx/denoise.sh python third_party/tlx/tutorials/<KERNEL.py>`

# CRITICAL: Run killgpu.sh
Run `third_party/tlx/killgpu.sh` to kill if any test runs a few minutes

# Commit messages
Don't commit unless the user explicitly asks you to.
When writing a commit message, don't make a bullet list of the individual
changes. Instead, if the PR is large, explain the order to review changes
(e.g., the logical progression), or if it's short just omit the bullet list
entirely.
Disclose that the PR was authored with Claude.
