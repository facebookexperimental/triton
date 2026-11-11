---
name: bisect-version-regression
description: >
  Bisect a Triton compiler performance or correctness regression reported
  between two Triton versions (for example 3.5 vs 3.8). Use when a kernel is
  slower, spills more, or compiles much more slowly after a Triton upgrade,
  and the culprit commit or change must be found. Tries tritonparse's
  built-in bisect first when its feasibility checklist passes, and falls
  back to a file-group revert bisect when the endpoints only exist as
  internal snapshot trees. Requires a GPU for benchmarking.
---

# Bisect a Triton Version Regression

Find the commit (or single change) that made a kernel regress between two
Triton versions. The workflow has one governing principle, a shared setup,
and two bisect paths. Do not skip the decision gate: picking the wrong path
wastes days, because the two paths assume different kinds of endpoints.

## Governing principle: reproduce the reporter's environment

Endpoints always come from the reporter's own environment, never from a
substituted source of truth.

1. Determine what the reporter actually ran: a GitHub release branch or
   tag, a pip wheel, a facebookexperimental/triton branch, an fbcode
   snapshot tree, or a buck-built binary. Record the exact versions.
2. Reproduce that on both sides. A GitHub-release report is bisected
   between GitHub releases. Only an fbcode-internal report calls for
   fbcode snapshot trees (`third-party/triton/{legacy,3.8,...}`).
3. Verify both endpoints reproduce the regression before any bisect. If
   they do not, the environment is not aligned yet; that is not evidence
   the regression is absent.

A real failure of this principle: a regression reported against fbcode
trees did not reproduce between GitHub release tips, because the fast
fbcode tree owed its speed to an internal backport that never landed in
the GitHub history. The lesson is environment fidelity, not "prefer
fbcode" (see `references/omninn-xa-path-b-case-study.md`).

## Shared step 1: mini-repro and bench oracle

Extract a standalone mini-repro from the reported kernel:

- Strip application wrappers (custom ops, framework launch code); keep
  constexprs, strides, and grid identical to the report.
- Emit one JSON line per config with compile time, registers, spills,
  shared memory, and runtime. `references/bench-oracle-template.py` is
  the starting point.
- Measure on one machine and one GPU, back-to-back, with repeated runs
  and a guard band between the good and bad values. Values inside the
  guard band are "skip", never a forced good/bad call.
- Before trusting any number, print and check `triton.__file__` and
  `triton.__version__`. A stale install in the environment once produced
  a completely false endpoint.

This harness doubles as the bisect oracle: exit 0 = good, exit 1 = bad,
exit 125 = skip (the `git bisect` contract).

## Shared step 2: cross-check with the reporter's own command

Run the reporter's original reproduction command (a buck target, a test,
a benchmark script) on both versions before bisecting. Reports often mix
a real regression with an artifact; in one case a ~30-minute autotune
compile time turned out equally slow on both versions, and only the
runtime regression was real. Bisect only what
survives this cross-check.

## Shared step 3: localize with tritonparse (evidence, not verdict)

Use tritonparse to capture, parse, reproduce, and diff compilations:

```bash
buck run //pytorch/tritonparse:tritonparse -- parse <trace_dir> --out <parsed_dir>   # fbcode
tritonparseoss parse <trace_dir> --out <parsed_dir>                                  # OSS
```

Treat every single IR difference as a hypothesis, not a conclusion. A
TTGIR diff that looks causal (for example one extra `convert_layout`)
can disappear under a revert while the runtime stays fast; the real
difference may live below TTGIR. IR diffs narrow the search; only
measurements convict.

## Decision gate: which bisect path?

Use Path A (tritonparse built-in bisect) only when **all** hold:

1. Good and bad endpoints are commits in one git history that a local
   clone can check out (triton-lang/triton, facebookexperimental/triton,
   llvm-project, or pytorch). Tree-only endpoints fail this.
2. The mini-repro reproduces at both endpoints through the oracle
   contract above (0/1/125), with the guard band encoded in the wrapper.
3. Intermediate commits build with one fixed recipe (`pip install -e .`
   by default); transient build breaks are acceptable because
   `--build-fail-action skip` is the default.
4. The LLVM pin (`cmake/llvm-info.json` or `cmake/llvm-hash.txt`) is
   constant across the range, or the full workflow below is acceptable
   (Triton bisect, then LLVM-descriptor comparison, then (Triton, LLVM)
   pair testing, then LLVM bisect). Never pin an arbitrary Triton commit
   against an arbitrary LLVM: treat `(Triton commit, its own declared
   LLVM pin)` as one unit. Build/API incompatibilities return skip
   (125), never good or bad.

If any condition fails, use Path B. The most common failure: the good
endpoint exists only as an internal snapshot-tree combination whose
behavior comes from backports or promotes that never landed in the
GitHub history, so there is no good commit to check out.

## Path A: tritonparse built-in bisect (default)

Install tritonparse (OSS: `pip install tritonparse`, or from the
meta-pytorch/tritonparse repo; fbcode: the buck target above). Then:

```bash
# Triton bisect. For facebookexperimental/triton, --triton-repo meta is
# required (the default is oai = triton-lang/triton).
tritonparseoss bisect --triton-dir <triton_clone> \
    --test-script <oracle_script> \
    --good <good_commit> --bad <bad_commit> --triton-repo meta

# Full workflow: also detect an LLVM bump at the culprit and bisect LLVM.
# <pairs.csv> comes from `tritonparseoss compat-build`.
tritonparseoss bisect --triton-dir <triton_clone> \
    --test-script <oracle_script> \
    --good <good_commit> --bad <bad_commit> --commits-csv <pairs.csv>

# Test (Triton, LLVM) pairs from a CSV to pin the LLVM range first.
tritonparseoss bisect --pair-test --triton-dir <triton_clone> \
    --test-script <oracle_script> --commits-csv <pairs.csv> \
    --good-llvm <llvm1> --bad-llvm <llvm2>

# LLVM-only bisect against a fixed Triton commit.
tritonparseoss bisect --llvm-only --triton-dir <triton_clone> \
    --test-script <oracle_script> --triton-commit <fixed_commit> \
    --good-llvm <llvm1> --bad-llvm <llvm2>

# PyTorch-repo bisect.
tritonparseoss bisect --target torch --torch-dir <pytorch_clone> \
    --test-script <oracle_script> --good <c1> --bad <c2>
```

Operating rules:

- Prefer a same-LLVM subrange when the full version range crosses LLVM
  bumps; shrink the range to where the pin is constant first.
- `--build-fail-action skip` (default) is what you want; use `abort`
  only when the build infrastructure itself looks broken.
- `--resume` / `--status --state <file>` continue or inspect a run.
  Results are `found`, `ambiguous`, `aborted`, or `error`, with a
  `*_bisect_result.json` and `*_git_bisect.log` per attempt.
- `ambiguous` (skipped commits) means report the candidate set. Never
  present a candidate as the culprit.
- After a culprit commit is found, isolate inside the commit: apply its
  hunks one at a time on the parent and re-measure. A culprit commit is
  not finished until one hunk (or a proven combination) flips the oracle.

Worked example: `references/jfav4-path-a-case-study.md`.

## Path B: file-group revert bisect (fallback)

Use when the endpoints are trees, not commits (typically fbcode snapshot
trees against an upstream base).

1. Build the fast tree and the slow tree exactly as reported, plus the
   upstream base the fast tree was derived from. Verify the oracle
   separates them (fast vs slow) on one GPU, back-to-back.
2. Group the differing compiler files between the fast tree and the base
   (for example by pass area: layout conversion, pipeliner, lowering to
   LLVM, analysis). Revert one group at a time in the fast tree toward
   the base, rebuild, and re-measure. A group whose revert turns the
   kernel slow contains the culprit. Budget for API-drift build fixes
   (renamed accessors, anonymous-namespace overloads, new TableGen ops,
   missing LLVM artifacts); fix only what the build needs and keep a
   list of every fix.
3. Narrow group -> file -> hunk the same way. When a single file will
   not revert cleanly, revert the smallest compilable subset.
4. Do not rely on cherry-pick chains of internal backports: the recorded
   backport list may be lexicographic rather than application order (in
   one case only 26 of 146 applied cleanly, 7 of 37 compiler candidates
   cumulatively). If the backport list and the file groups both come up
   empty, check the internal promote history (for example the fbsource
   history of `third-party/triton/{stable,legacy}`) and pickaxe for the
   suspect code; the culprit may have arrived by a tree promote, not by
   any listed backport.

Worked example: `references/omninn-xa-path-b-case-study.md`.

## Shared final step: independent confirmation

Whatever path found the suspect, confirm it in both directions:

1. Apply only the suspected hunk on the slow tree: it must become fast.
2. Apply its inverse on the fast tree (or on the upstream base): it must
   become slow.
3. Re-measure both with the same oracle, same GPU, back-to-back. Report
   the numbers from both directions, plus registers/spills when they
   move.

Only then name the culprit. If neither direction flips the oracle, the
suspect is wrong; return to the last verified narrowing step.

## Environment recipe

Source builds (needed whenever a wheel does not exist for an endpoint):

```bash
conda create -n <lane> python=3.12
conda install -c conda-forge gcc=13.2.0 gxx=13.2.0 cmake ninja pybind11 \
    libstdcxx-ng lld zstd zlib libgcc-ng
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
pip uninstall -y triton          # remove the bundled Triton first
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export TRITON_HOME=<per-lane-dir>   # never shared between parallel builds
git submodule update --init         # e.g. third_party/sleef
pip install -e . --no-build-isolation
```

fbcode snapshot trees may need `LLVM_ENABLE_WERROR=OFF` and small host
build workarounds; keep those in the copied tree only, and record them.
`references/build-lane-template.sh` packages one lane (clone, env,
build, verify) so two lanes can run in parallel worktrees without
interfering. Two lanes are enough; more parallelism rarely helps because
each measurement still needs the same GPU.

## Reporting

Report: the reporter's environment and endpoint versions; the oracle
and its guard band; which path was used and why the gate chose it; the
culprit commit or hunk; and the two-direction confirmation numbers.
Separate the commit-level culprit from any downstream pass-level cliff
and from any release-safe mitigation; all three can be true at once and
they answer different questions.
