#!/usr/bin/env bash
# Claude PR Review Agents — shared entry point
#
# Usage:
#   ./run-review.sh                         # review current branch vs main
#   ./run-review.sh path/to/diff.patch      # review a diff file
#   gh pr diff 123 | ./run-review.sh        # review a PR via pipe
#   REVIEW_MODE=plain ./run-review.sh       # force plain mode (no GPU)
#   REVIEW_MODE=agentic ./run-review.sh     # force agentic mode
#
# Requires: python3, PyYAML, claude CLI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
YAML_FILE="$SCRIPT_DIR/reviewers.yaml"

# ── Mode detection ──────────────────────────────────────────────────────────

detect_mode() {
    if [[ -n "${REVIEW_MODE:-}" ]]; then
        echo "$REVIEW_MODE"
    elif nvidia-smi &>/dev/null; then
        echo "agentic"
    else
        echo "plain"
    fi
}

MODE="$(detect_mode)"

# ── Diff acquisition ───────────────────────────────────────────────────────

DIFF_FILE=""
CLEANUP_DIFF=false

acquire_diff() {
    if [[ $# -gt 0 && -f "$1" ]]; then
        DIFF_FILE="$1"
    elif [[ ! -t 0 ]]; then
        DIFF_FILE="$(mktemp /tmp/claude-review-diff.XXXXXX)"
        CLEANUP_DIFF=true
        cat > "$DIFF_FILE"
    else
        DIFF_FILE="$(mktemp /tmp/claude-review-diff.XXXXXX)"
        CLEANUP_DIFF=true
        (cd "$REPO_ROOT" && git diff main...HEAD) > "$DIFF_FILE"
    fi

    if [[ ! -s "$DIFF_FILE" ]]; then
        echo "Error: empty diff — nothing to review." >&2
        exit 1
    fi
}

# ── Cleanup ─────────────────────────────────────────────────────────────────

cleanup() {
    if $CLEANUP_DIFF && [[ -n "$DIFF_FILE" ]]; then
        rm -f "$DIFF_FILE"
    fi
    # Clean up per-reviewer temp files
    rm -f /tmp/claude-review-out.*.txt 2>/dev/null || true
}
trap cleanup EXIT

# ── Parse YAML and run reviewers ────────────────────────────────────────────

run_reviewers() {
    local diff_file="$1"
    local mode="$2"

    # Parse reviewers.yaml with Python — emits one JSON object per reviewer
    local reviewer_json
    reviewer_json="$(python3 -c "
import yaml, json, sys
with open('$YAML_FILE') as f:
    data = yaml.safe_load(f)
for name, cfg in data.get('reviewers', {}).items():
    obj = {'name': name, 'prompt': cfg.get('prompt', '')}
    ag = cfg.get('agentic', {})
    obj['extra_prompt'] = ag.get('extra_prompt', '')
    obj['allowed_tools'] = ag.get('allowed_tools', '')
    obj['max_turns'] = ag.get('max_turns', 10)
    print(json.dumps(obj))
")"

    local pids=()
    local names=()
    local outfiles=()

    while IFS= read -r line; do
        local name extra_prompt allowed_tools max_turns prompt
        name="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['name'])")"
        prompt="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['prompt'])")"
        extra_prompt="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['extra_prompt'])")"
        allowed_tools="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['allowed_tools'])")"
        max_turns="$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['max_turns'])")"

        local outfile="/tmp/claude-review-out.${name}.txt"
        outfiles+=("$outfile")
        names+=("$name")

        if [[ "$mode" == "agentic" ]]; then
            local full_prompt
            full_prompt="$(printf '%s\n\n%s\n\nHere is the diff to review:\n\n```diff\n%s\n```' \
                "$prompt" "$extra_prompt" "$(cat "$diff_file")")"
            (
                cd "$REPO_ROOT"
                claude -p "$full_prompt" \
                    --allowedTools "$allowed_tools" \
                    --max-turns "$max_turns" \
                    > "$outfile" 2>&1
            ) &
        else
            local full_prompt
            full_prompt="$(printf '%s\n\nHere is the diff to review:\n\n```diff\n%s\n```' \
                "$prompt" "$(cat "$diff_file")")"
            (
                claude -p "$full_prompt" > "$outfile" 2>&1
            ) &
        fi
        pids+=($!)
    done <<< "$reviewer_json"

    # Wait for all reviewers
    local failed=0
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "Warning: reviewer '${names[$i]}' exited with error" >&2
            failed=$((failed + 1))
        fi
    done

    # Print results
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              Claude PR Review Results (${mode})              "
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    for i in "${!names[@]}"; do
        local label="${names[$i]}"
        echo "━━━━━ 🔍 ${label} ━━━━━"
        echo ""
        if [[ -f "${outfiles[$i]}" ]]; then
            cat "${outfiles[$i]}"
        else
            echo "(no output)"
        fi
        echo ""
    done

    if [[ $failed -gt 0 ]]; then
        echo "⚠ ${failed} reviewer(s) exited with errors." >&2
    fi
}

# ── Main ────────────────────────────────────────────────────────────────────

acquire_diff "$@"
echo "Mode: ${MODE}"
echo "Diff: ${DIFF_FILE} ($(wc -l < "$DIFF_FILE") lines)"
echo "Running $(python3 -c "
import yaml
with open('$YAML_FILE') as f:
    data = yaml.safe_load(f)
print(len(data.get('reviewers', {})))
") reviewers in parallel..."
echo ""

run_reviewers "$DIFF_FILE" "$MODE"
