---
name: documentation
description: >
  Structural rules for user-facing documentation in this repo — README.md and
  docs/*.md. Use when deciding where a piece of documentation belongs, adding a
  doc for a new feature or project, moving or restructuring existing docs, or
  renaming a doc that other files reference. Covers the three-level docs
  hierarchy, the organizing scheme and tag vocabulary in docs/tlx.md, link
  conventions, and the checks that a restructure preserved everything.
---

# Documentation structure

This skill is about **where things go**, not how to write them.

## Three levels

| Level | File | Job | Length |
|-------|------|-----|--------|
| 1. Router | `README.md` | What fbtriton is; route to the right project | ~150 lines |
| 2. Project | `docs/{tlx,compiler,torchtlx,ci}.md` | Landing page + reference for one project | as needed |
| 3. Deep dive | `third_party/.../docs/*.md`, `docs/design/*.md` | Design and internals | as needed |

A new project gets one level-2 doc and a row in the README's project table. It
does not get README prose beyond that row.

## Where a given piece goes

- **Does a first-time visitor need it to decide whether to use fbtriton?**
  → `README.md`. Nothing else goes there.
- **Does a user of one project need it?** → that project's `docs/*.md`.
- **Does only a contributor changing that subsystem need it?** → a deep-dive doc
  next to the code, linked from the project doc.

Routing for the categories that get misfiled most often:

| Content | Goes in |
|---------|---------|
| API reference | `docs/tlx.md` — never `README.md` |
| CI, workflows, runners, test commands, skip-lists, known-failing suites | `docs/ci.md` — never a project doc, never `README.md` |
| Compiler internals, pass pipelines, bug post-mortems | the subsystem's own `docs/` dir; link, don't inline |
| Engineering status, TODOs, "needs upstream re-sync" | `docs/ci.md` or a deep-dive doc — never a landing page |
| Install, versioning, wheels | `README.md` only |

## docs/tlx.md organizing scheme

**One functional list, not a vendor split.** Sections group operations by what
they do: Memory, Global memory access, Async memory access, Tensor core,
Synchronization, Warp specialization, Clusters and CLC, Layout control and
diagnostics, Other.

A vendor-specific op goes in the functional section matching its behaviour,
beside its counterpart on the other vendor — `buffer_load_to_local` sits with
`async_load` under Async memory access. Do not reintroduce an AMD or NVIDIA
appendix; that split is what previously produced duplicate entries.

**Tag vocabulary.** Every op entry carries a hardware tag using the arch ids from
`triton._internal_testing`, the same vocabulary as `docs/torchtlx.md`:

```
sm90   sm100   sm90+              NVIDIA
gfx942  gfx950  gfx1250           AMD
gfx942+                           AMD, ranged
amd                               all AMD targets
```

A trailing `?` marks unconfirmed availability. An absent vendor means the doc
makes no claim for that vendor, which is not the same as unsupported — preserve
the distinction when editing tags.

**Heading depth.** `##` per functional section, `###` per op or sub-topic, `####`
for its parts. `#####` exists only inside Clusters and CLC; do not go deeper.

## Link conventions

Links are relative to the file's own directory, so anything in `docs/` needs
`../` to reach repo paths:

```
README.md   -> docs/tlx.md            third_party/tlx/doc/x.pdf
docs/tlx.md -> compiler.md             ../third_party/tlx/tutorials/x.py
```

## Cross-references to update when moving or renaming a doc

Code and CI reference docs by path and will not follow a move:

```bash
grep -rn "README.md\|docs/.*\.md" .github/workflows/ .claude/ CLAUDE.md
```

Known dependents: `.github/workflows/b200.yml` and `mi350.yml` point at the Gluon
section of `docs/ci.md`; `.claude/rules/*.md` and `.claude/skills/*/SKILL.md`
point at project docs.

## Checks after a structural change

**Every relative link resolves:**

```bash
cd "$(git rev-parse --show-toplevel)"
for d in README.md docs/*.md; do
  base=$(dirname "$d")
  grep -oE '\]\(([^)h#][^)]*)\)' "$d" | sed -E 's/^\]\(//; s/\)$//' | sed 's/#.*//' \
  | grep -v '^$' | sort -u | while read -r f; do
      [ -e "$base/$f" ] || echo "$d -> MISSING: $f"
    done
done
```

**Nothing was dropped in a move.** Compare counts before and after and account
for every delta individually — do not accept "close enough":

```bash
grep -c '^- `tlx\.' <old> <new>              # op entries
grep -oE '\*\*\[[^]]+\]\*\*' <old> | wc -l   # hardware tags
grep -c '^ *```' <old> <new>                 # code fences
```

**Formatting:** `pre-commit run --all` (`trailing-whitespace`,
`end-of-file-fixer`, `check-yaml`). If `pre-commit` is unavailable, check
`grep -rn ' $'` on changed files and validate any touched YAML.

## Scope boundary

Restructuring and rewriting are separate tasks. When the job is structural:

- Move prose verbatim. Reordering sections is structural; rewording is not.
- Fix only unambiguous defects — dead links, wrong API names, exact duplicates.
- Leave semantic conflicts in place and report them. Deciding which of two
  contradictory statements is correct is the owner's call, not a formatting pass.
