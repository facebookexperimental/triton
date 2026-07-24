#!/usr/bin/env bash
# find-machinery.sh — Phase-2 (Step 2) helper for the fbtriton-backport skill.
# Given a candidate commit, RECURSIVELY unfold the full set of machinery commits it needs
# (all the way down to the release-branch cut) and print one ordered action plan:
#   ① cherry-pick (clean, oldest→newest)   ③ shim / pick-upstream-original (bundles/collisions)
#   ✓ already present (skip)
#
# "Machinery" = a commit on <main> since the cut whose change to a file the candidate touches
# makes the candidate no longer 3-way-merge cleanly onto <onto> (real `git merge-file` check —
# the same algorithm cherry-pick uses). Each such commit is then unfolded the same way.
#
# Usage:  find-machinery.sh <candidate-sha> [--release <ref>] [--onto <ref>] [--main <ref>]
#   defaults: --release origin/release/3.7.x   --main origin/main   --onto <=release>
#   --onto : 3-way-merge against THIS ref instead of bare release. Point it at your WIP pick-stack
#            so the closure is the EXTRA machinery beyond candidates already picked (much smaller;
#            vs bare release the closure also pulls in every other candidate).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO_ROOT"

RELEASE="origin/release/3.7.x"; MAIN="origin/main"; ONTO=""; CAND=""; MAX=400
while [[ $# -gt 0 ]]; do case "$1" in
  --release) RELEASE="$2"; shift 2;;
  --onto)    ONTO="$2"; shift 2;;
  --main)    MAIN="$2"; shift 2;;
  --max)     MAX="$2"; shift 2;;
  -h|--help) sed -n '2,19p' "$0"; exit 0;;
  *)         CAND="$1"; shift;;
esac; done
[[ -n "$CAND" ]] || { echo "usage: $0 <candidate-sha|#PR> [--release <ref>] [--onto <ref>] [--main <ref>]" >&2; exit 2; }
ONTO="${ONTO:-$RELEASE}"

# accept "#2289" or "2289" (a PR number) → resolve to the commit on <main> for that PR.
if [[ "$CAND" =~ ^#?[0-9]+$ ]]; then
  pr="${CAND#\#}"
  mapfile -t _m < <(git log "$MAIN" --fixed-strings --grep="(#$pr)" --format='%H %s' 2>/dev/null)
  [[ ${#_m[@]} -gt 0 ]] || { echo "no commit on $MAIN mentions (#$pr) — pass a SHA instead" >&2; exit 2; }
  CAND="${_m[0]%% *}"
  echo "resolved #$pr → $(git rev-parse --short "$CAND")  ${_m[0]#* }"
  (( ${#_m[@]} > 1 )) && { echo "note: ${#_m[@]} commits mention (#$pr) (PR#s can collide) — using the newest; pass a SHA to override:";
                           printf '      %s\n' "${_m[@]}"; }
  echo
fi
CUT="$(git merge-base "$RELEASE" "$MAIN")"

# already_on_onto <sha> <onto> : present by IDENTITY (ancestor, or -x cherry-pick trailer carries
# its full SHA). NOT by PR number — fbtriton/upstream reuse numbers (old "[DOCS] (#2320)" != coalesce #2320).
already_on_onto() {
  local full; full="$(git rev-parse "$1" 2>/dev/null)" || return 1
  git merge-base --is-ancestor "$1" "$2" 2>/dev/null && return 0
  git log "$2" --fixed-strings --grep="cherry picked from commit $full" --oneline 2>/dev/null | grep -q .
}

# machinery_of <sha> : print "shorthash<TAB>subject" for each commit on a file <sha> touches that
# CONFLICTS 3-way onto ONTO. Empty output ⇒ <sha> cherry-picks clean onto ONTO.
machinery_of() {
  local sha="$1" st f base ours theirs
  while IFS=$'\t' read -r st f; do
    [[ -n "$f" ]] || continue
    case "$st" in A*|D*) continue;; esac
    base="$(mktemp)"; ours="$(mktemp)"; theirs="$(mktemp)"
    git show "$sha^:$f" >"$base"   2>/dev/null || :
    git show "$sha:$f"  >"$theirs" 2>/dev/null || :
    if git cat-file -e "$ONTO:$f" 2>/dev/null; then
      git show "$ONTO:$f" >"$ours" 2>/dev/null || :
      if git merge-file -q -p "$ours" "$base" "$theirs" >/dev/null 2>&1; then
        rm -f "$base" "$ours" "$theirs"; continue          # clean → not a conflicting file
      fi
    fi
    rm -f "$base" "$ours" "$theirs"
    # conflicting file → precise deps: only commits that touched the SAME LINES the candidate edits
    # (git log -L on each hunk's old-side range). Pure additions (count 0) depend on no prior line.
    git diff -U0 "$sha^" "$sha" -- "$f" 2>/dev/null \
      | sed -nE 's/^@@ -([0-9]+),?([0-9]*) .*/\1 \2/p' | while read -r s c; do
        c="${c:-1}"; [[ "$c" -eq 0 ]] && continue
        git log -L "$s,$((s + c - 1)):$f" -s --format='%h%x09%s' "$CUT..$sha^" 2>/dev/null \
          | grep -aE "^[0-9a-f]{7,}$(printf '\t')"
      done
  done < <(git diff-tree --no-commit-id --name-status -r "$sha")
}

if already_on_onto "$CAND" "$ONTO"; then
  echo "candidate $(git rev-parse --short "$CAND") $(git show -s --format='%s' "$CAND" | grep -oE '\(#[0-9]+\)' | tail -1) is already present on ${ONTO##*/} — nothing to do."
  exit 0
fi

# is_mixed <sha> : touches CI/build/meta files alongside code ⇒ likely a [partial] target — a
# full pick would drag its unrelated machinery, so treat as terminal (don't recurse) and flag ③.
is_mixed() {
  git diff-tree --no-commit-id --name-only -r "$1" | grep -qE '^\.github/|^\.pre-commit-config|^\.ci/|(^|/)(BUCK|TARGETS)$'
}

# --- BFS transitive closure from the candidate down to the cut ---
declare -A SEEN CLASS CSUBJ CTS REASON
queue=()
enq() { local l; while IFS= read -r l; do [[ -n "$l" ]] && queue+=("$l"); done < <(machinery_of "$1" | sort -u); }
enq "$CAND"
count=0; capped=0
while ((${#queue[@]})); do
  entry="${queue[0]}"; queue=("${queue[@]:1}")
  h="${entry%%$'\t'*}"; subj="${entry#*$'\t'}"
  full="$(git rev-parse "$h" 2>/dev/null)" || continue
  [[ -n "${SEEN[$full]:-}" ]] && continue
  SEEN[$full]=1; CSUBJ[$full]="$subj"; CTS[$full]="$(git show -s --format=%ct "$full" 2>/dev/null || echo 0)"
  count=$((count+1)); if (( count > MAX )); then capped=1; break; fi
  if already_on_onto "$h" "$ONTO"; then CLASS[$full]=skip; continue; fi
  if grep -q '\[BUNDLE\]' <<<"$subj"; then
    CLASS[$full]=shim; REASON[$full]="squashed bundle → pick the real upstream PR it contains, not the bundle"; continue
  fi
  if is_mixed "$h"; then
    CLASS[$full]=shim; REASON[$full]="mixed (CI/build + code) → likely a [partial]: slice only the files you need (not recursing its unrelated machinery)"; continue
  fi
  CLASS[$full]=pick
  enq "$h"                                                 # unfold this commit's own machinery
done

# --- gather + print ---
picks=(); shims=(); skips=()
for full in "${!SEEN[@]}"; do
  row="${CTS[$full]}|$(git rev-parse --short "$full")|${CSUBJ[$full]}|${REASON[$full]:-}"
  case "${CLASS[$full]}" in pick) picks+=("$row");; shim) shims+=("$row");; skip) skips+=("$row");; esac
done
emit() { local title="$1"; local -n a="$2"; [[ ${#a[@]} -eq 0 ]] && return
  echo "$title"
  printf '%s\n' "${a[@]}" | sort -t'|' -k1,1n | while IFS='|' read -r _ sha subj note; do
    [[ -n "$note" ]] && printf '     %s  %-7s %s\n            ↳ %s\n' "$sha" "$(grep -oE '\(#[0-9]+\)' <<<"$subj" | tail -1 | tr -d '()' || echo —)" "${subj:0:60}" "$note" && continue
    pr="$(grep -oE '\(#[0-9]+\)' <<<"$subj" | tail -1 | tr -d '()' || true)"
    printf '     %s  %-7s %s\n' "$sha" "${pr:-—}" "${subj:0:66}"; done; }

echo "candidate : $(git rev-parse --short "$CAND")  $(git show -s --format='%s' "$CAND")"
echo "release   : $RELEASE  (cut $(git rev-parse --short "$CUT"))   onto: ${ONTO}"
echo
echo "=== SUGGESTED PLAN (heuristic from file-level 3-way merges — verify each; you decide pick / [partial] / [shim]) ==="
if [[ ${#picks[@]} -eq 0 && ${#shims[@]} -eq 0 ]]; then
  echo "  Looks clean — $(git rev-parse --short "$CAND") appears to merge with no extra machinery."
  echo "  Suggest: git cherry-pick -x $(git rev-parse --short "$CAND")  (then probe/build to confirm)."
else
  emit "  ③ WORTH A JUDGMENT CALL — bundle or likely collision; consider the upstream original or a [partial]/[shim] (Phase 4b):" shims
  emit "  ① LIKELY CLEAN PICKS — suggested order (oldest→newest); should apply once the ones above are in:" picks
  emit "  ✓ ALREADY PRESENT — skip:" skips
  echo
  echo "  A workable order: consider the ${#shims[@]} item(s) in ③, then the ${#picks[@]} in ① top-to-bottom, then the candidate ($(git rev-parse --short "$CAND"))."
  echo "  These come from file-level 3-way merges — confirm with a probe. A ① 'clean' one may still read better as a"
  echo "  [partial]/[shim], and a ③ item may pick clean; the pick/partial/shim call is yours (Phase 3/4)."
fi
if [[ $capped -eq 1 ]]; then
  echo "  ⚠️  stopped at --max=$MAX nodes — closure is huge; re-run with --onto <your pick-stack> to narrow it."
fi
exit 0
