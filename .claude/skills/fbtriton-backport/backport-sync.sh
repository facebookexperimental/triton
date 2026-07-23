#!/usr/bin/env bash
# Deterministic discovery for the release backport sync.
# Judgment prompt supplied to Claude: .claude/skills/fbtriton-backport/SKILL.md
#
# Runs in YOUR shell (real network) — not Claude's sandbox — so the fetch +
# enumerate git ops are reliable and reproducible. It:
#   1. Fetches OpenAI's release branch live from github + Meta's release/main.
#   2. Resolves the OpenAI frontier (see "Frontier" below) and prints how.
#   3. Enumerates candidate commits `frontier..head` for each source.
#   4. Classifies each (fix / amd / plumbing / revert / bundle) + a presence hint.
#   5. Writes a candidates file and opens an INTERACTIVE Claude session for the
#      judgment + report (HEADLESS=1 for one-shot `claude -p`; NO_CLAUDE=1 to skip).
#      Claude is launched with --secure-internet-mode: it keeps the internet
#      access the triage needs (git/gh reads) but sandboxes egress so no repo data
#      can be pushed off-box. The prompt also instructs the agent accordingly.
#
# TWO frontiers (START of each range), both tracked in $ENV_FILE, resolved:
#   OpenAI-release: --openai-frontier -> env file -> bootstrap (match Meta release-tip
#     commits to OpenAI's release by inline hash / PR# / title) -> fork-point fallback
#   Meta-main:      --meta-frontier   -> env file -> bootstrap (release fork-point on main)
# Candidates = frontier..head for each. After a sync LANDS, advance BOTH:
#   .claude/skills/fbtriton-backport/backport-sync.sh --advance-head    (OpenAI->openai head, Meta->main head)
#
# The script does NO cherry-pick/build — those are Claude's job, or a later step.
#
# Usage:
#   .claude/skills/fbtriton-backport/backport-sync.sh                        # auto (env file / bootstrap)
#   .claude/skills/fbtriton-backport/backport-sync.sh --openai-frontier <sha|tag> --meta-frontier <sha>
#   .claude/skills/fbtriton-backport/backport-sync.sh --advance-head         # after a sync: frontiers := heads
# Artifacts: .backports/<version>/{candidates.tsv,report.md,.backport-sync.env}
# Env/flags: RELEASE, VERSION, OUTDIR, OPENAI_FRONTIER, META_FRONTIER, OPENAI_URL,
#            OPENAI_REPO, MAIN_REF, OUT, ENV_FILE, PROMPT_FILE, HEADLESS=1, NO_CLAUDE=1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RELEASE="${RELEASE:-release/3.7.x}"
OPENAI_URL="${OPENAI_URL:-https://github.com/triton-lang/triton.git}"
OPENAI_REPO="${OPENAI_REPO:-$HOME/github/triton}"   # local OpenAI clone (for fork-point merge-base)
MAIN_REF="${MAIN_REF:-origin/main}"
OPENAI_FRONTIER="${OPENAI_FRONTIER:-}"
META_FRONTIER="${META_FRONTIER:-}"
VERSION="${VERSION:-}"          # defaults to the release version (release/3.7.x -> 3.7.x)
OUTDIR="${OUTDIR:-}"            # defaults to .backports/<version>
OUT="${OUT:-}"                  # defaults to <outdir>/candidates.tsv
ENV_FILE="${ENV_FILE:-}"        # defaults to <outdir>/.backport-sync.env
PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/.claude/skills/fbtriton-backport/SKILL.md}"
ADVANCE_HEAD=""

while [[ $# -gt 0 ]]; do case "$1" in
  --release)         RELEASE="$2"; shift 2;;
  --openai-frontier) OPENAI_FRONTIER="$2"; shift 2;;
  --meta-frontier)   META_FRONTIER="$2"; shift 2;;
  --openai-url)      OPENAI_URL="$2"; shift 2;;
  --openai-repo)     OPENAI_REPO="$2"; shift 2;;
  --main-ref)        MAIN_REF="$2"; shift 2;;
  --out)             OUT="$2"; shift 2;;
  --outdir)          OUTDIR="$2"; shift 2;;
  --version)         VERSION="$2"; shift 2;;
  --env-file)        ENV_FILE="$2"; shift 2;;
  --prompt-file)     PROMPT_FILE="$2"; shift 2;;
  --advance-head)    ADVANCE_HEAD=1; shift;;
  --no-claude)       NO_CLAUDE=1; shift;;
  -h|--help) sed -n '2,30p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done

cd "$REPO_ROOT"
VERSION="${VERSION:-${RELEASE#release/}}"; VERSION="${VERSION//\//-}"   # release/3.7.x -> 3.7.x
OUTDIR="${OUTDIR:-$REPO_ROOT/.backports/$VERSION}"
OUT="${OUT:-$OUTDIR/candidates.tsv}"
ENV_FILE="${ENV_FILE:-$OUTDIR/.backport-sync.env}"
mkdir -p "$OUTDIR"

# env-file vars for this release (slug: release/3.7.x -> release_3_7_x)
SLUG="$(printf '%s' "$RELEASE" | tr '/.-' '___')"
OPENAI_VAR="OPENAI_FRONTIER_$SLUG"      # last-synced OpenAI-release commit
META_VAR="META_MAIN_FRONTIER_$SLUG"     # last-triaged Meta-main commit

record_kv() {   # upsert KEY=VAL in the env file, preserving other keys
  local key="$1" val="$2" tmp; tmp="$(mktemp)"
  { echo "# backport-sync frontier state (per release branch)"
    [[ -f "$ENV_FILE" ]] && grep -vE "^(#|$key=)" "$ENV_FILE" || true
    echo "$key=$val"; } > "$tmp"
  mv "$tmp" "$ENV_FILE"
}

echo ">>> [1] Fetch heads live (not from a stale mirror)"
git fetch -f "$OPENAI_URL" "$RELEASE:refs/remotes/openai/$RELEASE"
git fetch origin "$RELEASE" main
OPENAI_HEAD="$(git rev-parse openai/"$RELEASE")"
REL_REF="origin/$RELEASE"                        # Meta release: freshly-fetched remote-tracking ref
REL_HEAD="$(git rev-parse "$REL_REF")"
MAIN_HEAD="$(git rev-parse "$MAIN_REF")"

# --advance-head: record both heads as the new frontiers and exit (run after a sync lands)
if [[ -n "$ADVANCE_HEAD" ]]; then
  record_kv "$OPENAI_VAR" "$OPENAI_HEAD"
  record_kv "$META_VAR"   "$MAIN_HEAD"
  echo ">>> Advanced frontiers in $ENV_FILE:"
  echo "      OpenAI -> $(git rev-parse --short "$OPENAI_HEAD")   Meta-main -> $(git rev-parse --short "$MAIN_HEAD")"
  exit 0
fi

# bootstrap: newest Meta release-tip commit that maps to a commit on OpenAI's release
bootstrap_frontier() {
  local oair="openai/$RELEASE" H S B h pr m
  while IFS=$'\t' read -r H S; do
    B="$(git log -1 --format='%b' "$H")"
    for h in $(printf '%s\n' "$B" | grep -oiE 'cherry picked from commit [0-9a-f]{7,40}' | awk '{print $NF}'); do
      git merge-base --is-ancestor "$h" "$oair" 2>/dev/null && { echo "$h"; return 0; }
    done
    for pr in $(printf '%s\n%s\n' "$S" "$B" | grep -oE '#[0-9]+' | tr -d '#' | sort -urn); do
      m="$(git log "$oair" -1 --format='%H' --extended-regexp --grep="#${pr}([^0-9]|\$)" 2>/dev/null || true)"
      [[ -n "$m" ]] && { echo "$m"; return 0; }
    done
    m="$(git log "$oair" -1 --format='%H' --fixed-strings --grep="$S" 2>/dev/null || true)"
    [[ -n "$m" ]] && { echo "$m"; return 0; }
  done < <(git log "$REL_REF" --format='%H%x09%s' | head -80)
  return 1
}

echo ">>> [2] Resolve frontiers"
# Check the target branch's frontier state file FIRST — it's the source of truth;
# only bootstrap if it's missing (or a specific frontier isn't recorded yet).
if [[ -f "$ENV_FILE" ]]; then
  echo "    frontier state: $ENV_FILE (checked first)"
  # shellcheck disable=SC1090
  source "$ENV_FILE" 2>/dev/null || true
else
  echo "    frontier state: none at $ENV_FILE -> will bootstrap"
fi

# --- OpenAI-release frontier ---  (override flag > env file > bootstrap)
OPENAI_SRC="override"
if [[ -z "$OPENAI_FRONTIER" ]]; then
  OPENAI_FRONTIER="${!OPENAI_VAR:-}"; [[ -n "$OPENAI_FRONTIER" ]] && OPENAI_SRC="env file"
fi
if [[ -z "$OPENAI_FRONTIER" ]]; then
  if OPENAI_FRONTIER="$(bootstrap_frontier)" && [[ -n "$OPENAI_FRONTIER" ]]; then
    OPENAI_SRC="bootstrap (hash/PR#/title match)"; record_kv "$OPENAI_VAR" "$OPENAI_FRONTIER"
  else
    echo "    !! OpenAI bootstrap match failed — falling back to the release fork-point" >&2
    # fork-point = merge-base(release, main) on the OpenAI clone's remote refs (ancestor of openai/$RELEASE)
    OPENAI_FRONTIER="$(git -C "$OPENAI_REPO" merge-base "origin/$RELEASE" origin/main 2>/dev/null || true)"
    [[ -z "$OPENAI_FRONTIER" ]] && OPENAI_FRONTIER="$(git rev-list --max-parents=0 "openai/$RELEASE" | tail -1)"
    OPENAI_SRC="FALLBACK: release fork-point via $OPENAI_REPO (review everything since the branch cut!)"
    record_kv "$OPENAI_VAR" "$OPENAI_FRONTIER"
  fi
fi
git rev-parse -q --verify "$OPENAI_FRONTIER^{commit}" >/dev/null 2>&1 || {
  echo "ERROR: OpenAI frontier '$OPENAI_FRONTIER' is not a valid commit." >&2; exit 2; }

# --- Meta-main frontier (symmetric) ---  (override flag > env file > bootstrap)
META_SRC="override"
if [[ -z "$META_FRONTIER" ]]; then
  META_FRONTIER="${!META_VAR:-}"; [[ -n "$META_FRONTIER" ]] && META_SRC="env file"
fi
if [[ -z "$META_FRONTIER" ]]; then
  META_FRONTIER="$(git merge-base "$REL_REF" "$MAIN_REF")"   # bootstrap = release fork-point on main
  META_SRC="bootstrap (release fork-point on main)"; record_kv "$META_VAR" "$META_FRONTIER"
fi
git rev-parse -q --verify "$META_FRONTIER^{commit}" >/dev/null 2>&1 || {
  echo "ERROR: Meta-main frontier '$META_FRONTIER' is not a valid commit." >&2; exit 2; }

echo ">>> [3] Anchors  (candidates = frontier..head for each)"
printf '    OpenAI    %s : frontier %s .. head %s   (via %s)\n' "$RELEASE" \
  "$(git rev-parse --short "$OPENAI_FRONTIER")" "$(git rev-parse --short "$OPENAI_HEAD")" "$OPENAI_SRC"
printf '    Meta-main %s : frontier %s .. head %s   (via %s)\n' "$RELEASE" \
  "$(git rev-parse --short "$META_FRONTIER")" "$(git rev-parse --short "$MAIN_HEAD")" "$META_SRC"
echo "    (override: --openai-frontier / --meta-frontier <sha>; both tracked in $ENV_FILE)"

# reads "sha|subject" on stdin, emits "present<TAB>class<TAB>sha<TAB>subject"
classify() {
  local relref="$1" h s fixpr present class
  while IFS='|' read -r h s; do
    fixpr="$(printf '%s' "$s" | grep -oE '#[0-9]+' | head -1 | tr -d '#' || true)"
    present="new"
    if [[ -n "$fixpr" ]] && git log "$relref" --oneline | grep -qE "#${fixpr}\b"; then present="in-fork"; fi
    case "$s" in
      *Cherry-pick*BUNDLE*|*Cherry-pick*upstream*)                                  class="bundle";;
      *[Vv]ersion*|*pypi*|*PyPI*|*Release\ Only*|*Release-only*|*wheel*|*Wheels*|*timeout*|*DOCKER*|*examples/*|*Pin\ LLVM*|*toolchain-version*) class="plumbing";;
      *Revert*|*revert*)                                                            class="revert";;
      *AMD*|*MFMA*|*WMMA*|*gfx*|*GFX*)                                              class="amd";;
      *[Ff]ix*|*crash*|*SIGSEGV*|*assert*|*deadlock*|*race*|*hang*|*verify*|*correct*) class="fix";;
      *)                                                                            class="other";;
    esac
    printf '%s\t%s\t%s\t%s\n' "$present" "$class" "$h" "$s"
  done
}

echo ">>> [4] Enumerate + classify -> $OUT"
{
  printf '# release=%s\topenai_frontier=%s\topenai_head=%s\tmeta_frontier=%s\tmain_head=%s\n' \
    "$RELEASE" "$(git rev-parse "$OPENAI_FRONTIER")" "$OPENAI_HEAD" "$(git rev-parse "$META_FRONTIER")" "$MAIN_HEAD"
  printf '#source\tpresent\tclass\tsha\tsubject\n'
  git log --reverse --format='%h|%s' "$OPENAI_FRONTIER..openai/$RELEASE" | classify "$REL_REF" | sed 's/^/openai\t/'
  git log --reverse --format='%h|%s' "$META_FRONTIER..$MAIN_REF"         | classify "$REL_REF" | sed 's/^/meta-main\t/'
} > "$OUT"
echo "    OpenAI candidates: $(git rev-list --count "$OPENAI_FRONTIER..openai/$RELEASE")" \
     "| Meta-main candidates: $(git rev-list --count "$META_FRONTIER..$MAIN_REF")"

echo ">>> [5] Hand off to Claude (interactive by default; HEADLESS=1 for one-shot)"
[[ -f "$PROMPT_FILE" ]] || { echo "ERROR: prompt file not found: $PROMPT_FILE" >&2; exit 2; }
RUN_CONTEXT="Candidates for $RELEASE are in $OUT (pinned frontiers/heads in its header line). Write audit-report.md and pick-list.txt into $OUTDIR."
FULL_PROMPT="$(cat "$PROMPT_FILE")

$RUN_CONTEXT"

if [[ -n "${NO_CLAUDE:-}" ]] || ! command -v claude >/dev/null 2>&1; then
  echo "    (claude not invoked — NO_CLAUDE set or 'claude' not on PATH). Artifact: $OUTDIR/candidates.tsv"
  echo "    Triage manually:  claude --secure-internet-mode \"\$(cat $PROMPT_FILE)\"   (candidates in $OUT; write to $OUTDIR)"
elif [[ -n "${HEADLESS:-}" ]]; then
  # --secure-internet-mode: allow the git/gh reads the triage needs, but sandbox
  # egress so the agent can't send our data off-box (see the prompt's data rule).
  claude --secure-internet-mode -p "$FULL_PROMPT" | tee "$OUTDIR/report.md"
  echo ">>> Artifacts in $OUTDIR:  candidates.tsv, audit-report.md, pick-list.txt, report.md"
else
  echo ">>> Starting interactive Claude session for triage (needs a TTY)."
  claude --secure-internet-mode "$FULL_PROMPT"
  echo ">>> Session ended. Artifacts in $OUTDIR:  candidates.tsv, audit-report.md, pick-list.txt"
fi

echo ">>> [6] Format all files with pre-commit (repo venv) + auto-commit fixes"
PRE_COMMIT="$REPO_ROOT/.venv/bin/pre-commit"
if [[ -x "$PRE_COMMIT" ]]; then
  # pre-commit reformats in place and exits non-zero when it changes files.
  "$PRE_COMMIT" run --all-files || true
  git add -u                        # stage the reformatted (tracked) files only
  if git diff --cached --quiet; then
    echo "    no formatting changes to commit."
  else
    git commit -m "[lint] Auto-fix pre-commit formatting issues" \
               -m "Generated by backport-sync.sh step [6] (pre-commit run --all-files)."
    echo "    committed formatting fixes: $(git rev-parse --short HEAD)"
  fi
else
  echo "    (skipped: $PRE_COMMIT not found — create the venv or adjust the path)"
fi
