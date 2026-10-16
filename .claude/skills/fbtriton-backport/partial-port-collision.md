# Backport hazard: split-lineage partial-port collision

A recurring, hard-to-diagnose class of backport/cherry-pick problem. Written up from the
v3.7.3 `#2153` / `#10081` / `#10194` / `#1847` case (2026-07) so the next person recognizes it
fast instead of re-deriving it over a dozen probes.

## Name / category

**Split-lineage partial-port collision** — also: *inherited adapted-port divergence*,
*out-of-order dependency ingestion across a branch cut*.

A fix enters the fork **twice, by two different paths**, and a **branch cut lands between them**,
so the release branch inherits a *partial, adapted* slice of the fix while the *full/original*
version stays on `main`. Later, cherry-picking the original (or a follow-up that depends on it)
**collides with the inherited slice** on the overlapping files — even though the branch "already
has the fix" and "never picked the bundle."

## Why it's confusing (the tells)

- "We never picked that bundle, so why is there a conflict?" — the conflict isn't from the bundle;
  it's from a *different, earlier* commit that carried an adapted slice of the same change.
- Part of the commit cherry-picks **clean**, part **conflicts** — the split maps exactly onto
  *which files the earlier partial port already touched*.
- The conflicting file often has **more/renamed content** than the commit you're picking
  (release is "ahead" on that file), so naive "match main" resolution would **regress** it.
- The dependency looks like a deep cascade in `git log <file>` history, but on *your branch* only
  one or two files actually conflict. (Don't confuse file *history* with branch *conflicts*.)

## Anatomy of the v3.7.3 case (worked example)

Upstream dependency chain: **#10081** (base: token-aware asyncmark wait counts, adds the SWP
`updateWaits` call + 3 tests) → **#10194** (follow-up BlockPingpong fix).

Two ingestion paths into fbtriton:
1. **Early hand-port (partial + adapted):** `#1847` ported *#10194's block-pingpong half* from a
   feature branch onto `main` on **06-30**, adapting it (flag renamed `gfx-arch` →
   `arch-generation-name`, behavior tuned to conservative `num=0`). It did **not** bring #10081's
   SWP `Pipeline.cpp` call or the standalone asyncmark tests.
2. **Routine upstream bundle (full):** bundle `#2031` brought **#10081 + #10194 together** on
   **07-15** — correct, but late, and it was later *dropped* from the v3.7.2 backport (fixes-only).

**The branch cut (`release/3.7.x`, 07-06) fell between them.** So release inherited `#1847` (the
adapted partial slice) and never got #10081.

Result when trying to backport later:
- `#10081` → conflicts on **only** `amd-block-pingpong-asyncmark-multi-token.mlir` (the file #1847
  rewrote); its `Pipeline.cpp` + 2 new tests apply **clean** (the un-ported slice).
- `#10194` → conflicts on `WGMMAPipeline.cpp` + `BlockPingpong.cpp` (exactly what #1847 ported).
- Verified on a **clean release tip** (no local picks) → the collision is purely the inherited
  `#1847`, not the bundle and not our own picks.

## Diagnosis playbook

1. **Reproduce on a clean release tip**, not your WIP branch, to rule out your own picks:
   `git checkout -b probe origin/release/<x>; git cherry-pick -x <sha>; git diff --name-only --diff-filter=U`
2. **Find who put the conflicting file on release:**
   `git log origin/release/<x> --oneline -1 -- <conflicting-file>` → usually an *inherited* commit.
3. **Is that commit inherited or backported?**
   `git merge-base --is-ancestor <that-sha> <fork-point>` → YES = inherited at the cut.
4. **Read that commit's message** — a "Port … from <feature-branch>" description = a hand-port;
   check which upstream PRs it claims vs which files it actually touched (the *partial* set).
5. **Split clean-vs-conflict by file** — the clean files are the un-ported slice; the conflicting
   files are the already-ported (adapted) slice.
6. **Check the true upstream** (`~/github/triton`) for the *current* form (flag names, logic) — it
   may differ from both your branch and the commit you're picking; decide divergence direction
   against the right reference (usually fbtriton `main`, not upstream).

## Resolution: least-divergence shim, not a re-pick

Do **not** cherry-pick the original commit wholesale — it re-applies the already-present slice in
an older/foreign form, producing a **franken-state that matches neither `main` nor the branch**
(maximal divergence), and may regress the adapted file.

Instead, **build a minimal shim = only the un-ported slice, adapted to the branch's lineage:**
- Apply just the hunks/files the branch is genuinely missing (here: #10081's `Pipeline.cpp`
  `updateWaits` call), verifying the referenced symbols exist on the branch.
- Materialize any needed test/support file from **fbtriton `main`** (not the raw upstream commit),
  so it matches the fork's adapted form (flag names, etc.).
- **Deliberately omit** the files the inherited partial port already owns (here: BlockPingpong.cpp
  and the block-pingpong test) so nothing collides with it.
- Land it as a clearly-labeled `[shim]` commit documenting what slice it takes, from where, and
  what it omits and why.

This lands the target fix (#2153, with its test) while keeping the inherited (#1847) lineage
untouched — the lowest-divergence outcome.

## Two triggers, one remedy (the shim)

The "extract the minimal slice as a shim" remedy applies to **two** distinct situations. Both
showed up in v3.7.3:

- **Variant A — collision with an inherited partial port** (the #10081/#2153 case above). The
  commit you want *conflicts* because an earlier adapted slice is already on the branch. Shim =
  the **un-ported** slice (the files that apply clean), adapted to the branch's lineage.
- **Variant B — wholesale import of a mixed commit** (the **#2149** case). The commit applies
  *cleanly* but **bundles wanted + unrelated changes**. #2149 is titled "[CI] Isolate torchTLX…"
  yet also carries the 226-line `amd_pa_decode.py` rewrite that #2281/#2288 depend on. Importing
  it wholesale (even with the DU/conflict files resolved) drags **unrelated CI/lint divergence**
  (`.github/workflows/*.yml`, `.pre-commit-config.yaml`) onto the release branch and leaves a
  franken-commit (some files dropped, some kept-ours). Shim = the **wanted** slice only
  (`git show <sha> -- <wanted files> | git apply --3way`), omitting everything else.

Tell for Variant B: after the import, `git show --stat <your-commit>` lists files **outside** the
subsystem you were backporting. If a "[CI]"/"[BE]"/"lint" commit is touching kernel code (or vice
versa), it's a mixed commit — shim the slice, don't import the whole thing.

## Which slices to include — the convergence test

Aim the shim at **lowest divergence from Meta `main`**, not "smallest change to release."
For each file a mixed commit touches, include its slice **only if that slice meaningfully
converges the file to `main`**:

```bash
git diff origin/release/<x> origin/main -- <file> | grep -c '^[+-]'   # total gap to main
git show <sha> -- <file>                          | grep -c '^[+-]'   # what this slice closes
```

- **slice ≈ whole gap** → the commit is the file's only delta → **include** (file becomes
  `== main`; future picks touching it apply clean — this is the "easier next pick" payoff).
- **slice ≪ gap** → the file is dominated by *other* divergence you are **not** backporting →
  **skip**. Partial convergence there does **not** ease the next pick (the file still conflicts
  on the rest) and only adds unwanted content.

v3.7.3 #2149: `amd_pa_decode.py` slice = the file's whole gap → included (== main). The CI files
`h100.yml`/`mi350.yml` had 34-/129-line gaps to main of which #2149 closed only 6/12 (main moved
far ahead via other CI commits) → excluded; converging them is a separate CI-scope backport, not
this shim's fragment.

## Shim commit conventions

- **Title = the source PR's original subject + PR#**, prefixed with a tag that says *how*
  it's partial (keeps `backport-sync`'s PR#/title matching working — PR recognized as handled,
  not re-offered — and preserves provenance):
  - **`[partial]`** — a subset of a single PR (e.g. `[partial] [CI] Isolate torchTLX … (#2149)`,
    decode files only).
  - **`[shim]`** — a constructed bridge assembled from >1 source to enable a pick
    (e.g. `[shim] [AMD][gfx9] Restore token-aware … (#10081)` = #10081's `Pipeline.cpp` call
    + the test from `main`, to enable #2153).
  - Rule of thumb: slice of one commit → `[partial]`; glued from multiple sources → `[shim]`.
- **Body records both** the divergence reason and the shim reasoning (slice taken, files
  omitted + why, `== main` verification). See the commit-message template in the
  `fbtriton-backport` skill (Phase 4b), which this file is co-located under (`SKILL.md`).

## Verify the shim is faithful

A shim should reproduce exactly the base commit's slice that the routine bundle would have given —
no more, no less. Confirm by diffing the shimmed files against fbtriton `main`:

```bash
git diff --quiet origin/main HEAD -- <shimmed-file>   # match = faithful
```

A match (modulo unrelated later evolution that postdates the release cut) means the shim equals
"what the bundle would have brought, minus the collisions/irrelevant parts" — and you correctly
avoided a wholesale bundle/commit pick. In v3.7.3 the asyncmark test file was byte-identical to
`main`; the `Pipeline.cpp`/`WGMMAPipeline.cpp` files differed only by unrelated features/refactors
that landed on `main` after the cut (the normal release-lag baseline every pick shares).

## Prevention

- **Don't hand-port a follow-up (#10194) ahead of its base (#10081).** If an urgent port is
  needed, port the base first, or clearly mark the port as partial + adapted and file the base as
  a follow-up so the routine bundle sync reconciles it.
- **Bundle dependency-complete slices in order** so `base → follow-up` never split across bundles.
- **Record adapted forks** (flag renames like `gfx-arch` → `arch-generation-name`) somewhere the
  next backporter will see them — silent renames are what turn a clean pick into a conflict.
- **Cut release branches at a bundle boundary**, not mid-window between a hand-port and the bundle
  that completes it.
