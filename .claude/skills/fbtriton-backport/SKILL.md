---
name: fbtriton-backport
description: >
  Use when backporting commits onto an fbtriton release branch (release/3.7.x etc.):
  planning which commits/PRs to bring over, a cherry-pick that conflicts or comes out a
  franken-commit, a pick that drags in unrelated files, or a commit whose test or
  machinery is missing on the release branch. Applies to both routine point-release
  updates and targeted (e.g. AMD-perf) backports.
---

# fbtriton backport — pick, map dependencies, resolve into clean shims

The full arc of a release backport in one place:
**Phase 1 triage → Phase 2 map dependencies → Phase 3 pick clean → Phase 4 resolve the
messy ones (match-main or a documented shim) → Phase 5 build & validate → Phase 6 finalize.**

**Finalizing the pick list is two steps** (Phase 1 = candidates, Phase 2 = machinery):
1. **The candidate list** — *what we actually want*: either a point-release scan of all **fix**
   commits between the cut and `main`, or an **LLM-curated** list for a stated purpose (e.g. the
   AMD-perf commits). These are the "selected" picks.
2. **The machinery** — *what those candidates need*: the dependency commits each candidate sits
   on top of (base kernels, refactors, prereqs) that aren't on the release branch yet. Found by
   tracing each candidate's touched-file history back to the cut (codified in Phase 2).

Worked case study of the nastiest dependency class:
[`partial-port-collision.md`](partial-port-collision.md).

## Golden rules (govern every phase)

1. **Lower divergence from Meta `main`.** `main` is the answer key: every pick /
   resolution should move release *toward* `main`, never into a franken-state that
   matches neither `main` nor clean release. A region that matches `main` exactly
   (modulo unrelated later evolution) is safe. *Caveat:* `main` lags OpenAI, often a
   lot — see Phase 4 for when "match main" does and does not apply.
2. **Prefer an explicit `[partial]`/`[shim]` commit over a wholesale pick or a
   hand-merge.** When a commit conflicts or is *mixed* (wanted change bundled with
   unrelated CI/lint/other-subsystem changes), extract only the wanted slice into its
   own tagged commit — don't cherry-pick the whole thing and resolve/drop files.
   (`[partial]` = subset of one PR · `[shim]` = bridge assembled from >1 source.)
3. **Document the divergence.** Every shim/resolution records (a) why the clean pick
   fails and (b) the reasoning (slice taken, from where, what's omitted and why) — in
   the commit message, and for a batch in `.backports/<version>/backport_<version>.md`. No
   silent surgery.
4. **Never re-pick a change already on release.** If release inherited an adapted form
   (pre-cut), re-picking the upstream original double-applies / regresses it. Confirm
   with the Phase 2 probe before picking.

## Where things live

Artifacts under `.backports/<version>/` (e.g. `.backports/3.7.2/`, `.backports/3.7.3/`):
- `candidates.tsv` — sweep input (written by `backport-sync.sh`; deterministic mode only).
- **`backport_<version>.md`** — the single living doc, **built up over multiple rounds — don't
  try to fill it all at once.** Start it with just the purpose & scope + initial candidate list,
  then append each phase's output as you get it: dependency stacks, pick/skip decisions, shim
  resolutions, land order, build/test results, perf numbers. Sections left empty/TBD between
  rounds are expected. Format it like `.backports/3.7.3/backport_3.7.3.md`.
- `commit-dag.html` — an interactive pick/dependency DAG for review (same dir).

Don't build during triage.

## The two repos

- **OpenAI Triton**: upstream `triton-lang/triton`, cloned at `~/github/triton`.
- **Meta Triton**: our fork `facebookexperimental/triton`, cloned at `~/github/fbtriton`.
  Release branch e.g. `release/3.7.x`; fork point (branch cut) e.g. `3e55adf1b`.

## Data & network rule (if run in a sandboxed session)

Read only what the task needs: local git repos, and for commit/PR lookups the upstream
GitHub repos `triton-lang/triton` and `llvm/llvm-project`. **Don't push or send data
anywhere** (no `git push`, no `gh pr create`/comment, no uploads). Everything produced
stays as local files. If a lookup is blocked, say so and stop — don't route around it.

---

## Phase 1 — Step 1: build the candidate list (what we want)

### Start here — ask the user two things

Before gathering any candidates, confirm with the user (don't guess — the answer picks a
very different commit set, fixes-only vs features):

1. **Purpose of the backport** — a *regular point-release update* (catch the release branch
   up), or a *specific ask* (e.g. "the AMD perf commits", a named feature / set of PRs)?
2. **Target release branch** — which `release/X.Y.z`? That fixes the fork/cut point
   (`CUT=$(git merge-base origin/release/X.Y.z origin/main)`) every later step uses.

Route on the answer:

- **Regular point update → run the deterministic sweep** (candidate source (a) below).
- **Specific ask → scan `CUT..main_head` yourself** and curate a goal-scoped pool (source
  (b) below).

Both then funnel into the same job (drop → scope → map dependencies → order); only the
input pool differs.

### Candidate source (a) — regular point update: deterministic sweep

Run `.claude/skills/fbtriton-backport/backport-sync.sh` (pass `RELEASE=release/X.Y.z`). It fetches live, resolves
the frontier, and writes `candidates.tsv`: *everything* on the frontier since the last sync,
from two sources — `source=openai` (OpenAI's release branch) and `source=meta-main` (our
`main`). Columns: `source  present  class  sha  subject`; the `#` header pins the range SHAs.
Filtering the sweep down (Step "Drop"/"Scope" below) is the main work.

### Candidate source (b) — specific ask: scan `CUT..main_head`

Start from the user's intent (e.g. "the AMD perf work prod needs", often sourced from a chat
/ PRs / Phabricator diffs) and *gather* the relevant commits yourself, scoped to the goal:
```bash
CUT=$(git merge-base origin/release/X.Y.z origin/main)
git log --oneline "$CUT"..origin/main -- <goal-paths>          # e.g. third_party/amd third_party/tlx
git log --oneline "$CUT"..origin/main | grep -iE '<keywords>'  # e.g. amd|gfx9|mfma|decode|gemm
```
Cross-check the hits against the user's named PRs/diffs, then group into feature stacks. The
pool is already goal-scoped, so the main work is **dependency mapping** (Phase 2) and
confirming each candidate is **landable** (on `main`; flag anything "not on main yet ⏳").
Capture the result in `backport_<version>.md` (format like `.backports/3.7.3/backport_3.7.3.md`).

The `present`/scope hints are *hints* — verify against real code and real dependencies;
never trust a PR-number grep (squashed bundles break it).

### Read the PR + commit message for every candidate (don't decide from the diff alone)

The message carries what the diff can't, and it drives the pick decision:

```bash
git show -s <sha>                          # fbtriton commit msg — intent, deps, Differential Revision
gh pr view <PR> --repo triton-lang/triton  # upstream PR discussion (openai candidates only)
```
Read it for: **intent** (fix vs feature), **stated dependencies** ("depends on #X",
stacked-diff parents → feed Phase 2), whether a **revert** is a provisional fallback that
names its real fix (→ likely reject), whether the change was **superseded/reverted later**,
and **what tests it ships**. Decide pick/skip from the message + real code, not the diff alone.

- **Tests are part of the pick.** A PR's own tests are what validate the change — keep them
  with the code (never drop a PR's test as "noise") and run them in Phase 5. Note which tests
  each pick adds/edits now. A test file a PR *depends on* (created by an earlier PR) is often
  the silent machinery that makes the pick conflict — Phase 2 surfaces it (e.g. #2153's
  asyncmark test ← #10081).

### Drop what we don't need (both modes)

- **Already in the fork.** We pull OpenAI history in as squash commits titled
  `[Cherry-pick][BUNDLE] …`; each bundle's message lists the PR numbers it carried, so
  most AMD/CUDA/backend fixes are already here, buried in a bundle. Confirm by *code*,
  not SHA/PR number:
  ```bash
  git log origin/release/X.Y.z --grep='Cherry-pick.*BUNDLE' --oneline   # bundles on release
  git show <bundle-sha> | grep -oE '#[0-9]+' | sort -un                 # PRs a bundle brought
  git show <cand-sha> -- <file> | grep '^+'                             # what the candidate adds
  git show origin/release/X.Y.z:<file> | grep -F '<distinctive line>'   # already there?
  ```
  If present, drop. (A probe cherry-pick that comes out **empty** also = already there.)
- **Reverted inside the range.** If X is reverted by "Revert X" in-range, drop both; for
  revert→re-land chains keep only the net result.
- **Absent path.** Skip a fix that only touches code not on the release branch — check
  the file/symbol exists on `origin/release/X.Y.z` first.

### Decide what's in scope

**Deterministic mode:**
- **OpenAI commits — take everything else.** OpenAI already decided what belongs on
  their release branch; don't second-guess fix-vs-feature. Only skip release plumbing
  *we* own (version bumps, PyPI publishing, wheel CI).
- **Meta-main commits — fixes only** (per `RELEASE.md`): regressions & critical fixes
  (crashes, deadlocks, wrong results, leaks), fixes to just-shipped features, docs,
  release/CI fixes. **No new features.**

**LLM-curated mode:** the pool is already goal-scoped → scope = **in service of the goal
+ landable**. Keep a candidate if it advances the goal (an AMD-perf kernel or its
enabler) **and** it is on `main` (can't cherry-pick what hasn't landed — flag "not on
main yet ⏳" and hold). The fixes-only rule does *not* bind (a targeted feature backport
is features by definition), but every **Drop** rule above still applies.

**Reverts are a triage decision, not a rubber stamp.** A release-branch revert is often a
provisional fallback that names its own real fix ("in the hopes of fixing…", "so we have
a fallback"). Read the message and check whether Meta `main` **kept** the reverted code.
If main didn't revert, matching main means *rejecting* the revert. Cautionary **#9445**:
reverted async-copy-by-default on gfx950, called itself a fallback for #9431 (already in
our LLVM), main keeps async-copy on — taking it broke the MI350 FA kernels.

**The LLVM pin is a special case** — skip the OpenAI pin commit, but don't ignore it. Our
pin lives in `cmake/llvm-info.json` (`llvm_hash` + `build_number`); we build/host our own
prebuilt LLVM. OpenAI pins a different LLVM in `cmake/llvm-hash.txt` — cherry-picking it
edits a file our build ignores and points at an LLVM we never built. So skip the commit
but **flag the pin delta** (old vs new hash) so a human can decide to advance our own LLVM
(a deliberate infra step: bump `llvm-info.json`, rebuild — not a cherry-pick).

---

## Phase 2 — Step 2: find the machinery the candidates need

Every Step-1 candidate rides on commits that landed on `main` before it. Any that aren't on
the release branch yet are **machinery** you must pull in (or shim) — miss one and the pick
conflicts, comes out empty, or builds a broken tree. This step's product is the
**dependency graph + land order**. Do it in two passes: **(1) codified trace** (predict the
machinery up front), then **(2) probe** (confirm on a clean release tip).

### Kinds of dependency

- **Base** — the feature's own foundational commit (`#2040` base decode; `#2157` base
  a16w16 GEMM). Always precedes its follow-ups.
- **Prereq feature** — a sibling the commit builds on, usually named in the PR stack
  (`#2157` needs `#2152` AGPR + `#2156` pinned layouts + `#2153` async_wait).
- **Cross-chain** — a prereq in *another* group, so one stack must land before another
  (GEMM's `#2156` needs layout-infra `#2017` → the GEMM stack depends on the layout stack).
- **Machinery (silent)** — an intervening commit that *doesn't look related* but the
  file's history reveals it; never in a PR description, surfaces only as a conflict/empty
  on probe (`#2281` needs `#2149`'s decode rewrite; `#2158` needs `#2054`/`#2146`/`#2151`;
  `#2153` needs `#10081`'s asyncmark test file).

### Codify it — trace each candidate's touched-file history back to the cut

Silent machinery is findable mechanically, before any cherry-pick: a candidate's diff hunks are
written against `main`'s version of each file it touches, so a commit that landed on that file
between the cut and the candidate is a dependency **iff** its change makes the candidate no
longer 3-way-merge cleanly onto release.

**Use the helper** [`find-machinery.sh`](find-machinery.sh) — it **recursively unfolds the full
machinery down to the cut** and prints a **suggested plan** (heuristic, each line labeled with
commit + PR#) — the pick / `[partial]` / `[shim]` call stays yours:
**③** shim / pick-upstream-original first (bundle ⇒ find the real upstream PR; inherited-partial
collision ⇒ Phase 4b) · **①** `git cherry-pick -x` in the given order (oldest→newest; each is clean
once the ones above it are in) · **✓** already present (identity-checked via ancestry / the `-x`
trailer — *not* PR number, which collides). If the candidate itself is already on the baseline it
says so and stops.
```bash
.claude/skills/fbtriton-backport/find-machinery.sh <candidate-sha> --onto <wip-stack-tip>
# --onto = your WIP pick-stack during Step 2 (so the closure is the EXTRA machinery beyond
# candidates already picked). Omit it to measure vs bare release (pulls in every other candidate).
```
Under the hood, per commit: `CUT=$(git merge-base release main)`; for each modified file, 3-way
merge (base=`<sha>^`, ours=onto, theirs=`<sha>`) — if it conflicts, its deps are
`git log "$CUT".."<sha>^" -- <file>`; the tool BFS-expands each dep the same way until the cut.

The plan's buckets are suggestions, not orders — treat them as a starting point and decide per
commit:
- **③ worth a judgment call** — a squashed `[Cherry-pick][BUNDLE]` (usually better to pick the
  *real upstream PR* it contains, not the bundle — `git log --all --oneline --grep='(#<upstream-PR>)'`),
  or a commit that collides with an inherited partial port → often a `[shim]`/`[partial]` (Phase 4b).
- **① likely clean picks** — `git cherry-pick -x` in the printed order (oldest→newest); still
  confirm, and a "clean" one may read better as a `[partial]`/`[shim]`.
- **✓ already present** — skip (identity-verified).

Worked example: `find-machinery.sh #2290 --onto <stack>` unfolds in one run to ③ bundle `#1981`
(whose `getMaxElementsPerThread` is upstream `#10059`) → ① `#2150` → candidate `#2290` —
the clean chain `#10059 → #2150 → #2290`.

### The probe — confirm on a CLEAN release tip (not your WIP branch)

```bash
git checkout -b probe origin/release/X.Y.z
git cherry-pick -x <sha>
git diff --name-only --diff-filter=U               # which files conflict
git status --porcelain | grep -E '^(DU|UD|AU|UA)'  # add/del (absent-on-branch) conflicts
# clean   → pick it.
# empty   → already present → drop (already-in-fork).
# conflict on <file> → something landed between the cut and <sha>:
git log --oneline <fork-point>..<sha> -- <conflicting-file>   # the intervening commit(s) = the dep
git cherry-pick --abort; git checkout -; git branch -D probe
```
Attribute the conflict:
```bash
git log origin/release/X.Y.z --oneline -1 -- <file>   # who put it on release
git merge-base --is-ancestor <that-sha> <fork-point> && echo "INHERITED at cut"
git show -s <that-sha>                                 # "Port … from <branch>" = hand-port
```
Also check the true upstream (`~/github/triton`) for the *current* form (flag names,
logic) to pick the divergence reference correctly — usually fbtriton `main`, not upstream.

### What the probe tells you (→ which phase resolves it)

- **Clean** → pick in Phase 3.
- **Empty** → already present → drop.
- **Missing dependency, clean-pickable** → pull the intervening commit into the stack
  (Phase 3). E.g. `#2158` ← `#2054`/`#2146`/`#2151` (`Fixup.cpp`).
- **Collision or mixed commit → a shim (Phase 4b).** Two triggers:
  - **A. Collision with an inherited partial port** — the commit conflicts because an
    earlier *adapted* slice is already on release (pre-cut hand-port). Shim = the
    *un-ported* slice. (`#2153`/`#10081` vs inherited `#1847` — see
    [`partial-port-collision.md`](partial-port-collision.md).)
  - **B. Wholesale import of a mixed commit** — it applies but bundles wanted + unrelated
    changes (a `[CI]`-titled commit that also rewrites a kernel). Tell:
    `git show --stat <sha>` lists files *outside* the subsystem you're backporting. Shim =
    the *wanted* slice only. (`#2149`.)

### Group into stacks + land order

Cluster picks into dependency stacks (base + follow-ups + machinery). Order
**topologically**: base → prereqs → feature → follow-ups; cross-chain deps set the order
*between* stacks (foundational stack first). Record it — Phase 3 just replays it. (v3.7.3
landed as `L → B → A → M`: layout infra first because GEMM depends on it; decode and addmm
independent.)

---

## Phase 3 — Pick (two-pass)

Work on a throwaway branch off `origin/release/X.Y.z`, `git cherry-pick -x`, in the order
Phase 2 produced. Two passes because clean picks are low-risk and hand edits are where
things break:

- **Pass 1 — clean picks in bulk.** Apply the list; clean → keep; conflict → record and
  `--abort` (defer to pass 2); empty → drop. Build+test **once** at the end (Phase 5); if the
  build breaks, bisect.
- **Pass 2 — troubled picks, one at a time.** Resolve each (Phase 4), then build+test that
  one commit before the next, so a failure maps to a single resolution. Log every outcome
  in `.backports/<version>/backport_<version>.md`: picked / reduced-to-slice (what dropped & why) /
  rejected (why).

A pass-1 conflict is often a **missed dependency** — re-run the Phase 2 probe rather than
forcing the merge.

---

## Phase 4 — Resolve the messy picks

Two resolution shapes. Pick by what the Phase 2 probe found.

### 4a. Ordinary drift conflict → resolve to match `main`

Our fork drifted from OpenAI (FpSan bundle, lint-autofix, newer APIs), so a fine fix may
not apply cleanly — but the same fix is usually already integrated on Meta `main`, which
shows the correct end result.

1. Resolve so the changed region matches `main`, then prove it (empty diff = same as main):
   ```bash
   diff <(git show <main_head>:<file> | sed -n '/<region start>/,/<region end>/p') \
        <(sed -n '/<region start>/,/<region end>/p' <file>)
   ```
2. **Keep both the fix and our fork-only additions** (e.g. `runDeadIterArgElimination`,
   `PruneLocalStoreOfReshapeConvert`). A plain auto-merge can silently keep a line main
   removed — the diff-against-main catches that.
3. **Upstream reshuffle:** when upstream splits/renames something we also changed (one
   `RewritePatternSet` → `convertCleanup` + `scfCleanup`), place our addition into the new
   structure the way main did, then diff against main.
4. **Prefer the real upstream commit** over another repo's release adaptation — it's often
   already in our repo on a merge branch with its honest SHA:
   `git log --all --oneline --grep='(#<upstream-PR>)'`. (#10132: prefer upstream
   `ca21b1b95`'s `runDeadIterArgElimination` over OpenAI's older
   `populateForOpDeadArgumentElimination`.)
5. **Caveat — `main` lags OpenAI.** For a recent OpenAI commit main hasn't bundled yet,
   there is nothing on main to match — resolve to OpenAI's intent while preserving fork
   adaptations (pybind11 isolation in `PluginUtils.h`, FpSan, newer APIs). "Not on main" is
   **not** a reason to skip. And **don't invert the answer key**: `merge-base
   --is-ancestor` first — if main lags, "matches main" can mean "reverted a fix main hasn't
   caught up to."

### 4b. Mixed/collision → author a `[partial]`/`[shim]` commit

For trigger A (collision with inherited partial port) or B (mixed commit): extract only
the wanted slice, don't import wholesale.

**Convergence test — which slices to include.** Include a file's slice only if it
*meaningfully converges that file to `main`*:
```bash
git diff origin/release/X.Y.z origin/main -- <file> | grep -c '^[+-]'   # total gap
git show <sha> -- <file>                            | grep -c '^[+-]'   # this slice
```
- slice ≈ whole gap → the commit is the file's only delta → **include** (→ `== main`).
- slice ≪ gap → the file is dominated by other divergence you're not backporting → **skip**.
- (v3.7.3: `amd_pa_decode.py` slice = whole gap → include; `h100.yml`/`mi350.yml` 6/34 &
  12/129 → skip.)

**Title tag + PR#.** Keep the source PR's original subject + PR# so `backport-sync`'s
PR#/title matching still recognizes it; prefix a tag saying *how* it's partial:
- **`[partial]`** — a documented subset of one PR (`[partial] [CI] Isolate torchTLX … (#2149)`
  = its decode files only).
- **`[shim]`** — a bridge assembled from >1 source (`[shim] [AMD][gfx9] Restore token-aware
  … (#10081)` = #10081's `Pipeline.cpp` call + the test from `origin/main`, to enable #2153).

**Build it** — take only the wanted files; materialize support files from `origin/main` (the
fork's adapted form), not the raw upstream commit:
```bash
git show <sha> -- <wanted-file> [<wanted-file> ...] | git apply --3way   # wanted code/tests
git show origin/main:<path> > <path>                                     # support file, fork's form
git add <wanted files> && git commit -F - <<'MSG' … MSG
```
Before applying a C++ hunk, confirm the symbols it references exist on release:
`git grep -c <symbol> origin/release/X.Y.z -- <dirs>`.

**Commit message template:**
```
[partial|shim] <ORIGINAL PR subject> (#NNNN)   # [partial]=subset of 1 PR · [shim]=bridge from >1 source

Divergence: <why a clean pick of <#PR/sha> fails or is wrong here>. E.g. release
inherited <#PR/sha> pre-cut (a partial, adapted port that rewrote <files> with
<flag/behavior changes>); or <#PR> is a mixed commit that also drags <unrelated
files>. Verified on a clean release tip (conflict is from <source>, not our picks).

Shim: takes only <wanted slice/files> from <#PR / origin/main>; OMITS <files>
because <reason>. Lowers divergence: release gains only <the needed change>, keeps
<inherited lineage> untouched. Shimmed files verified == origin/main (modulo unrelated
later evolution).
```

**Verify:**
```bash
git diff --quiet origin/main HEAD -- <shimmed-file> && echo "== main (faithful)"
```
A match (modulo features/refactors that landed on `main` after the cut) confirms the shim
equals what the routine bundle would have brought, minus the collisions/irrelevant parts.
Re-run the deferred tail to confirm dependent picks now apply clean.

### Red flags — STOP and re-check

If you catch yourself doing any of these, stop — it violates a Golden Rule and you're about
to raise divergence from `main`:

- **Wholesale import + resolve/drop** of a mixed commit → drags unrelated divergence, leaves
  a franken-commit. Shim the slice (rule 2).
- **Re-pick a commit already present in adapted form** (inherited pre-cut) → double-apply /
  regress; maximal divergence (rule 4).
- **"Match main" when main lags** → inverts the answer key; check `merge-base --is-ancestor`
  first (rule 1 caveat).
- **Silent conflict surgery** → undocumented resolution (rule 3): put it in the commit message
  + `backport_<version>.md`.

---

## Phase 5 — Build & validate

Don't finalize on green cherry-picks alone — a clean apply can still miscompile or fail a
kernel test. Build the wheel and run the tests, **including the ones the picked PRs ship**.

### Build the wheel

```bash
.claude/skills/fbtriton-backport/build-wheel.sh   # prompts default sensibly; override via env vars
```
Stages googletest + the prebuilt LLVM matching `cmake/llvm-info.json` + gcc-toolset-14 (all
cached after the first run), then `uv build --wheel` → `dist/`. Builds off the current branch
(your pick-stack tip). Off a `release/*` branch the wheel carries a `+git<sha>` suffix — fine
for testing; the real `<X.Y.Z>+fb` bump is Phase 6. If it breaks, bisect the pick stack.

### Test the wheel

```bash
.claude/skills/fbtriton-backport/test-wheel.sh    # or: WHEEL=dist/<wheel>.whl .claude/skills/fbtriton-backport/test-wheel.sh
```
Installs the wheel into a throwaway venv with the torch build matching the GPU (auto: ROCm on
AMD, CUDA on NVIDIA) and runs: a smoke kernel, the launch.h crash tests, and the **arch-gated
TLX correctness suite** (on AMD only `amd`/`ikbo` kernels run, `-k "amd or ikbo"`; the rest
auto-skip). On hang: `third_party/tlx/killgpu.sh`. (v3.7.3 baseline: smoke ✓, launch.h
42-skip on AMD, TLX 69 passed / 4 skip on gfx950.)

### Run the tests the picked PRs ship (the generic suite is not enough)

Each picked PR usually adds/updates its own tests — those are what actually exercise what you
backported, and they may not be in the arch-gated suite. From each pick, find and run them:
```bash
git show <sha> --stat | grep -E 'test|\.mlir'      # tests this pick adds/edits
# MLIR/LIT (C++): build triton-opt, then   llvm-lit -v test/<path>/<file>.mlir
# python kernels:  pytest -q third_party/tlx/tutorials/testing/test_correctness.py -k "<kernel>"
```
A **shim that materialized a test from `main`** (Phase 4b) *must* make that test pass — it's
the proof the shimmed machinery works (e.g. the #10081 asyncmark LIT under the #2153 shim). If
a pick's test isn't covered by the arch-gated run, run it explicitly and record it.

### Perf validation — two comparisons (required for a perf-motivated backport)

Correctness green doesn't mean fast. Run **two** GPU comparisons (optional for a routine point update):

1. **main HEAD ↔ backport — completeness:** did we capture every perf commit the target cases need?
2. **prior release ↔ backport — no regression:** shared/standard kernels must not regress. (The
   backported kernels are usually *net-new* on the old release — they can't even run there, so the
   regression check is on the shared paths + the comparison is "absent → present".)

Give each section a one-line **Status** verdict in the doc.

**Build every reference wheel FROM SOURCE at the exact SHA — do NOT use the published nightly pip
wheel.** The nightly lags `main` HEAD and can be *older* than your backport (missing your just-picked
commits), so it is an invalid reference and will mislead you. Build with `build-wheel.sh` from a
detached `git worktree` at the SHA (copy the skill dir in so the script resolves), install each wheel
in its own venv, and de-shadow the `triton-rocm` that ROCm torch drags in (see `test-wheel.sh`).

**Completeness check.** `git fetch` first, then per target-kernel file:
`git log --cherry-pick --right-only <branch>...origin/main -- <file>` → the commits on main not in the
backport. A kernel **byte-identical** to main HEAD ⇒ no kernel commit missed. Confirm on-GPU that each
missing commit is immaterial (or pick it).

**Know how each harness imports its kernel (this is a trap that silently invalidates A/Bs):**
- `test_amd_*_perf.py` imports the kernel from the **installed wheel** (`triton.language.extra.tlx…`)
  → each wheel runs *its own bundled kernel*; driving it with a different venv swaps the kernel. To
  isolate the *compiler*, the two wheels' kernels must be **byte-identical** (`diff` them first).
- `gfx9_gemm/**/bench.py` imports `from matmul_kernel import …` → the kernel **next to the bench**;
  run each wheel's own bench+kernel (if `bench.py` is identical and only `matmul_kernel.py` differs,
  that's a clean shipped-vs-shipped comparison).
- **Byte-identical kernel source ≠ identical compiled perf** — the compiler differs; *measure*, never
  claim "parity by construction."

**Measurement hygiene (AMD/MI350):**
- Wrap runs in `third_party/tlx/denoise.sh` (NUMA pin + clock lock). It locks via
  `sudo rocm-smi --setperfdeterminism <MHz>` — **`--setperflevel high` is "Not supported" on MI350**.
  Select the GPU with `HIP_VISIBLE_DEVICES` (ROCm torch also honors `CUDA_VISIBLE_DEVICES`).
- perfdeterminism locks **sclk only, not mclk** → bandwidth-bound kernels (paged decode) stay noisy;
  compute-bound (GEMM) lock cleanly.
- **Noise floor:** run the same config twice (A/B/A) and trust a delta only if it exceeds the A-vs-B
  spread. **Warm up** first — a `TRITON_USE_C_DISPATCHER` cold-start fallback corrupts the first
  timing (produces implausible 0.2–0.5× outliers). Prefer the **within-run TLX/rocBLAS ratio**
  (DVFS-robust) over cross-run absolutes. Same-GPU-sequential avoids device variance but risks
  thermal ordering; two locked GPUs in parallel avoid ordering but add ~2% device bias — report the caveat.
- **tritonbench** covers standard ops (`gemm`, …) for the no-regression check; its `decoding_attention`
  needs CUDA-only `xformers` → unrunnable on ROCm. Per-op harnesses under `tutorials/testing/*_perf.py`
  are the targeted counterpart. Commands: `kernel-perf-testing` skill. `killgpu.sh` on hang.

**Record:** side-by-side numbers (`prev | backport | main`, with Δ) + the exact commands, in
`.backports/<version>/backport_<version>.md`. **Flag any regression before Phase 6** — it's a pick
decision to revisit (missing machinery / wrong shim slice / missing perf commit), not something to ship.

Record correctness pass/fail (with numbers + skips) and the perf signal in
`.backports/<version>/backport_<version>.md`.

---

## Phase 6 — Finalize

### Bump the version (two independent files — update both)

- `python/triton/__init__.py` — `__version__ = '<X.Y.Z>+fb'` (keep the `+fb` fork marker).
- `setup.py` — `TRITON_VERSION = "<X.Y.Z>" + get_triton_version_suffix()`. setup.py hardcodes
  the base *independently*, so it won't pick up the bump on its own — miss it and the wheel
  keeps the stale base. On a `release/*` branch the git suffix is empty (`triton-<X.Y.Z>`);
  off a release branch it appends `+git<sha>`.

Do **not** touch `docs/conf.py` (its `version`/`release` are empty and `smv_tag_whitelist`
tracks upstream's "Advance version" cadence, not point-release backports).

### Record it

Commit the bump with the backport doc (`.backports/<version>/backport_<version>.md` +
`commit-dag.html`). That doc is the running record — every pick/skip, shim/resolution
(target pick, conflict source, slice / omissions, `== main` verification), and build/test
+ perf results. Keep the class of problem documented in
[`partial-port-collision.md`](partial-port-collision.md).
