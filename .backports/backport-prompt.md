# How to triage backport candidates for a Meta Triton release

You get a list of candidate commits in `candidates.tsv` (columns: `source  present  class  sha  subject`).
The `#` line at the top pins the SHAs that bound each range. Your job: decide which commits we actually
want on the release branch, then write two files (an audit report and a pick list). Don't build anything.

## Data & network rule (must follow)

This session runs with sandboxed egress (`--secure-internet-mode`). Read only from the sources this task
needs: the local git repos, and — for commit/PR lookups — the upstream GitHub repos `triton-lang/triton` and
`llvm/llvm-project`. **Do not access any other website, and do not push or send any data anywhere:** no
`git push`, no `gh pr create`/comment, no pastes, no uploads, no POSTs to external APIs. Everything you produce
stays as local files under the output dir. If a needed lookup is blocked by the sandbox, say so and stop —
don't route around it.

## The two repos

- **OpenAI Triton**: the upstream repo `triton-lang/triton`, cloned at `~/github/triton`.
- **Meta Triton**: our fork `facebookexperimental/triton`, cloned at `~/github/fbtriton`. Release branch is
  e.g. `release/3.7.x`.

In the TSV, `source=openai` rows come from OpenAI's release branch, and `source=meta-main` rows come from
our main branch.

## Step 1: drop commits we don't need (applies to both sources)

**Already in the fork.** Don't trust the `present` column — it's just a PR-number grep and it's often wrong.
We pull OpenAI history into the fork as big squash commits titled `[Cherry-pick][BUNDLE] ...`, and each
bundle's message lists the PR numbers it brought in. So most AMD/CUDA/backend fixes are already here, buried
inside a bundle. For each OpenAI candidate, check the bundles first, then confirm by comparing the actual code:

```bash
git log origin/release/X.Y.z --grep='Cherry-pick.*BUNDLE' --oneline   # bundles on our release branch
git show <bundle-sha> | grep -oE '#[0-9]+' | sort -un                 # PRs a bundle brought in
git show <cand-sha> -- <file> | grep '^+'                             # what the candidate adds
git show origin/release/X.Y.z:<file> | grep -F '<distinctive line>'   # is it already there?
```

If the candidate's PR is in a bundle, or its code is already in the file, skip it. (Squashing breaks both
`git patch-id` and PR-number matching, so you have to check the real code, not SHAs.)

**Reverted inside the range.** If commit X is later reverted by "Revert X" within the range, drop both.
Watch for longer chains (revert, then re-land) and keep only the net result.

## Step 2: filter by source

**OpenAI commits — take everything else.** OpenAI already decided what belongs on their release branch, so
don't second-guess fix-vs-feature. The only extra thing to skip is release plumbing we own ourselves:
version bumps, PyPI publishing, and wheel CI.

**One exception to "don't second-guess": reverts.** A release-branch revert is often a provisional fallback,
not a settled decision — and its message usually says so. Read the PR/commit message and check whether **Meta
main made the same change**. If main kept the reverted code (didn't revert), then matching main means
*rejecting* the revert (see "Resolving conflicts" #6). #9445 is the cautionary case: it reverts
async-copy-by-default on gfx950, its own message calls it a fallback for the real fix #9431 (already in our
LLVM), and main keeps async-copy on (#1335) — taking it broke the MI350 FA kernels.

The **LLVM pin is a special case** — skip the OpenAI pin commit, but don't ignore it. We manage LLVM through
our own mechanism: our pin lives in `cmake/llvm-info.json` (`llvm_hash` + `build_number`) and we build and
host our own prebuilt LLVM tarballs. OpenAI pins a different LLVM in a different file (`cmake/llvm-hash.txt`,
a `triton-lang/llvm-project` hash). Cherry-picking their pin commit would edit a file our build mostly
ignores and point at an LLVM we never built, so it would break or no-op — the commit is the wrong vehicle
for us. But the pin bump might carry a real LLVM fix we need. So: skip the commit, and **flag the pin delta**
in the report (old vs new LLVM hash) so a human can decide whether to advance our own LLVM. Advancing LLVM
is a deliberate infra step (bump `llvm-info.json`, rebuild our LLVM), not a cherry-pick.

**Meta main commits — fixes only.** Follow the cherry-pick rules in `RELEASE.md`: regression and critical
fixes (crashes, deadlocks, wrong results, memory leaks), fixes to features that just shipped, docs, and
release-branch/CI fixes. No new features.

Also skip a fix if it only touches code that isn't on the release branch. Check the file or symbol exists on
`origin/release/X.Y.z` before picking it.

## Step 3: write the output (next to `candidates.tsv`)

- **`audit-report.md`** — one row per candidate: `sha | source | subject | decision | reason`. Decision is
  one of: `pick`, `skip-plumbing`, `skip-feature`, `drop-reverted`, `already-in-fork`, `skip-absent-path`.
  Put anything big, feature-heavy, or borderline in a separate section so a human can eyeball it.
- **`pick-list.txt`** — the commits to cherry-pick, in order: OpenAI first, then Meta-main, oldest to newest.
  At the end, record the frontiers for next time: OpenAI = current release head, Meta-main = current main head.

## Picking the commits (later phase — don't run during triage)

Work on a throwaway branch off `origin/release/X.Y.z`, and use `git cherry-pick -x`. Do it in two passes.
Building after every single commit is too slow, and the risk isn't spread evenly: clean picks are low-risk,
while conflict resolutions (hand edits) are where things break.

**Pass 1 — clean commits, in bulk.** Cherry-pick the pick-list in order. If a commit applies cleanly, keep
going. If it conflicts, record it and skip it (`git cherry-pick --abort`); you'll handle it in pass 2. If a
cherry-pick comes out empty, the change is already there, so drop it. Once all the clean commits are on, build
once and run core tests once. They applied without conflict, so they're low-risk — one build at the end is
enough. If the build breaks, bisect to find the culprit.

**Pass 2 — troubled commits, one at a time.** For each commit you skipped in pass 1, cherry-pick it, resolve
the conflict (see "Resolving conflicts" below), then build and test that one commit before moving to the next.
Conflict resolutions are the risky part, so validate each on its own — that way a failure maps to a single
resolution. Record the outcome of each in `pick-log.md` as you go: resolved and picked, reduced to the
applicable part (and what you dropped and why), or rejected (and why) — so the audit trail stays accurate.

## Resolving conflicts

Rule of thumb: **if your resolved code matches Meta main exactly, the pick is safe.** Main already has this
fix, integrated and tested with our other changes, so matching main adds no new risk. Use Meta main
(`main_head`) as the answer key — don't hand-merge blind.

Why conflicts happen: our fork has drifted from OpenAI (the FpSan bundle, the lint-autofix pass, newer APIs).
The fix itself is fine, it just doesn't apply cleanly. And the same fix is often already on Meta main, so main
shows you the correct end result.

**Caveat — Meta main lags OpenAI, often by a lot.** The "match main" answer key only works when main actually
has the change. For recent OpenAI commits main hasn't bundled yet, there is nothing on main to match — main is
*not* the answer key. Resolve to OpenAI's intent while preserving our fork-specific adaptations (e.g. the
pybind11 isolation in `PluginUtils.h`, the FpSan additions, newer local APIs), and accept that the correct end
result may be code that isn't on main yet. Two corollaries:
- **"Not on main" is not a reason to skip an OpenAI commit.** Main being behind is the normal case, not a
  signal to drop. Skip only for: already-in-fork (bundled), reverted-in-range, release plumbing we own, or a
  genuine fork incompatibility — and for the last, *adapt* the commit (as with the pybind11 isolation) rather
  than reject it when it has value.
- **Don't invert the answer key.** Matching main proves safety only when main is ahead of (or level with) the
  commit. If main lags, "matches main" can mean "reverted a fix main just hasn't caught up to yet" — check the
  direction before trusting it.

1. Resolve so the changed region matches Meta main, then prove it (empty diff means it's the same as main):
   ```bash
   diff <(git show <main_head>:<file> | sed -n '/<region start>/,/<region end>/p') \
        <(sed -n '/<region start>/,/<region end>/p' <file>)
   ```

2. Keep both the fix and our local additions. Don't drop fork-only code the upstream commit never knew about
   (like `runDeadIterArgElimination` or `PruneLocalStoreOfReshapeConvert`). A plain auto-merge can quietly
   keep a line that main actually removed — the diff against main catches that.

3. Common case: upstream reshuffles code we also changed. Upstream splits or renames something (e.g. one
   `RewritePatternSet` becomes `convertCleanup` + `scfCleanup`) while we had added our own item to the old
   version. Git can't tell where our item belongs, so it conflicts. Put our addition into the new structure
   the same way main did, then diff against main.

4. Prefer the real upstream commit over another repo's release adaptation. The original upstream commit is
   often already in our repo (on merge branches like `tlx-amd-meta`, with its real SHA) even though
   main/release only carry it squashed inside a bundle. Cherry-pick that original — it has an honest message
   and provenance — instead of, say, OpenAI's hand-edited release copy:
   ```bash
   git log --all --oneline --grep='(#<upstream-PR>)'   # find the real upstream commit in the repo
   ```

5. Watch for API drift. OpenAI's release branch can lag main and adapt a fix to an older API. Our release
   branch may already have the newer API (pulled in via a bundle), so OpenAI's version is wrong for us —
   matching Meta main gives the correct newer-API form.
   Example: #10132. OpenAI's `b7fa781f9` uses the old `populateForOpDeadArgumentElimination`; Meta main and
   the upstream original `ca21b1b95` use `runDeadIterArgElimination`. Match main, and prefer `ca21b1b95`.

6. Read the commit/PR message, not just the diff — especially on reverts. The message reveals intent the diff
   hides. Watch for stopgaps that name their own replacement ("in the hopes of fixing…", "ideally land #Y
   instead", "so we have a fallback"): the real fix is often already in our tree, so the right resolution is to
   reject, not force-apply. For any revert, check whether Meta main also reverted — if main kept the original,
   matching main means dropping the pick. This also guards the rule of thumb above: a revert that resolves
   cleanly can still leave the region *not* matching main, so run the diff-against-main check on it too.
   Example: #9445. A one-line revert that resolved cleanly next to fork-only `is_fpsan_supported`, but silently
   flipped gfx950 async-copy **off** — the opposite of main — and broke the AMD FA kernels. Its message flagged
   it as a fallback for #9431, which our LLVM already carries. Correct call: reject (match main = async-copy on).

## Bump the version (release finalization)

After the picks land, bump the fork release version. **Two files, and they are independent — update both:**

- `python/triton/__init__.py` — `__version__ = '<X.Y.Z>+fb'` (runtime version; keep the `+fb` fork marker).
- `setup.py` — `TRITON_VERSION = "<X.Y.Z>" + get_triton_version_suffix()`. setup.py hardcodes the base version
  *independently* of `__init__.py`, so it will **not** pick up the bump on its own — miss this and the built
  wheel keeps the stale base (e.g. `triton-3.7.0+git<sha>` instead of `3.7.2`). On a `release/*` branch the
  git suffix is empty (wheel = `triton-<X.Y.Z>`); off a release branch it appends `+git<sha>`.

Do **not** touch `docs/conf.py`: its `version`/`release` are empty strings and its `smv_tag_whitelist` tracks
upstream's "Advance version" cadence, not point-release backports (v3.7.1 didn't update it either).

Commit the bump together with the triage audit trail (`candidates.tsv` + `pick-log.md`).
