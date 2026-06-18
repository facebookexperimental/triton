#!/usr/bin/env bash
# Multi-agent debug helper for TLX/Triton investigations.
#
# Usage:
#   ./debug_helper.sh
#   ./debug_helper.sh --list-investigations
#   ./debug_helper.sh --run-subagent barrier_visualization <claude|codex> <ir-path> <out-dir> [context-file]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
SCRIPT_PATH="$REPO_ROOT/debug_helper.sh"
AGENT_TEMPLATE_DIR="$REPO_ROOT/.claude/debug_helper/agents"
SUBAGENT_TEMPLATE_DIR="$AGENT_TEMPLATE_DIR/subagents"
WRAPPER_AGENT_TEMPLATE="$AGENT_TEMPLATE_DIR/wrapper.md"

CLAUDE_SKIP_FLAG="--dangerously-skip-permissions"
CODEX_SKIP_FLAG="--dangerously-bypass-approvals-and-sandbox"

readonly CLAUDE_SKIP_FLAG
readonly CODEX_SKIP_FLAG

usage() {
    cat <<'EOF'
Usage:
  ./debug_helper.sh
  ./debug_helper.sh --list-investigations
  ./debug_helper.sh --run-subagent <investigation> <claude|codex> <ir-path> <out-dir> [context-file]

Supported investigations:
  Run ./debug_helper.sh --list-investigations
EOF
}

list_investigations() {
    if [[ ! -d "$SUBAGENT_TEMPLATE_DIR" ]]; then
        return 0
    fi

    find "$SUBAGENT_TEMPLATE_DIR" -maxdepth 1 -type f -name '*.md' \
        | while IFS= read -r template_path; do
            basename "$template_path" .md
        done \
        | sort
}

skip_flag_for_llm() {
    case "$1" in
        claude)
            echo "$CLAUDE_SKIP_FLAG"
            ;;
        codex)
            echo "$CODEX_SKIP_FLAG"
            ;;
        *)
            echo "unsupported LLM: $1" >&2
            return 1
            ;;
    esac
}

validate_llm() {
    local llm="$1"

    case "$llm" in
        claude|codex)
            if ! command -v "$llm" >/dev/null 2>&1; then
                echo "Error: '$llm' CLI was not found in PATH." >&2
                exit 1
            fi
            ;;
        *)
            echo "Error: expected 'claude' or 'codex', got '$llm'." >&2
            exit 1
            ;;
    esac
}

log_line() {
    local log_file="$1"
    shift
    mkdir -p "$(dirname "$log_file")"
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "$log_file"
}

render_template() {
    local template_path="$1"
    shift

    if [[ ! -f "$template_path" ]]; then
        echo "Error: missing agent template: $template_path" >&2
        exit 1
    fi

    python3 - "$template_path" "$@" <<'PY'
import sys

template_path = sys.argv[1]
items = sys.argv[2:]
if len(items) % 2 != 0:
    raise SystemExit("template replacements must be KEY VALUE pairs")

replacements = dict(zip(items[0::2], items[1::2]))
with open(template_path, encoding="utf-8") as f:
    text = f.read()

for key, value in replacements.items():
    text = text.replace("{{" + key + "}}", value)

sys.stdout.write(text)
PY
}

prompt_for_llm() {
    local choice=""

    while true; do
        read -r -p "Which LLM should debug_helper use? [claude/codex] " choice
        case "$choice" in
            claude|codex)
                echo "$choice"
                return 0
                ;;
            *)
                echo "Please enter 'claude' or 'codex'."
                ;;
        esac
    done
}

run_llm_print() {
    local llm="$1"
    local prompt_file="$2"
    local output_file="$3"
    local skip_flag
    skip_flag="$(skip_flag_for_llm "$llm")"

    case "$llm" in
        claude)
            (
                cd "$REPO_ROOT"
                claude "$skip_flag" -p "$(<"$prompt_file")"
            ) >"$output_file" 2>&1
            ;;
        codex)
            (
                cd "$REPO_ROOT"
                codex exec "$skip_flag" -C "$REPO_ROOT" "$(<"$prompt_file")"
            ) >"$output_file" 2>&1
            ;;
    esac
}

launch_wrapper_agent() {
    local llm="$1"
    local skip_flag
    local log_dir="${TMPDIR:-/tmp}/debug_helper"
    local launcher_log="$log_dir/launcher.log"
    local prompt_file="$log_dir/wrapper_prompt.md"

    skip_flag="$(skip_flag_for_llm "$llm")"
    mkdir -p "$log_dir"

    log_line "$launcher_log" "Selected LLM: $llm"
    log_line "$launcher_log" "Permission skipping enabled with flag: $skip_flag"
    log_line "$launcher_log" "Launching wrapper agent from: $REPO_ROOT"

    render_template "$WRAPPER_AGENT_TEMPLATE" \
        REPO_ROOT "$REPO_ROOT" \
        SCRIPT_PATH "$SCRIPT_PATH" \
        LLM "$llm" \
        SKIP_FLAG "$skip_flag" \
        >"$prompt_file"

    case "$llm" in
        claude)
            exec claude "$skip_flag" "$(<"$prompt_file")"
            ;;
        codex)
            exec codex "$skip_flag" -C "$REPO_ROOT" "$(<"$prompt_file")"
            ;;
    esac
}

json_status() {
    local status="$1"
    local reason="$2"
    local report_path="$3"
    local logs_path="$4"
    local suggested_next_modes="$5"
    local status_path="$6"

    python3 - "$status" "$reason" "$report_path" "$logs_path" "$suggested_next_modes" "$status_path" <<'PY'
import json
import sys

status, reason, report_path, logs_path, suggested, status_path = sys.argv[1:]
payload = {
    "status": status,
    "reason": reason,
    "report_path": report_path,
    "logs_path": logs_path,
    "suggested_next_modes": [item for item in suggested.split(",") if item],
}
with open(status_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY
}

valid_status_json() {
    local status_path="$1"

    python3 - "$status_path" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    sys.exit(1)

if payload.get("status") not in {"success", "failed", "needs_context"}:
    sys.exit(1)

for key in ("reason", "report_path", "logs_path", "suggested_next_modes"):
    if key not in payload:
        sys.exit(1)

if not isinstance(payload["suggested_next_modes"], list):
    sys.exit(1)
PY
}

subagent_template_path() {
    local investigation="$1"
    local template_path="$SUBAGENT_TEMPLATE_DIR/$investigation.md"

    if [[ ! "$investigation" =~ ^[A-Za-z0-9_][A-Za-z0-9_-]*$ ]]; then
        echo "Error: invalid investigation name '$investigation'." >&2
        exit 2
    fi

    if [[ ! -f "$template_path" ]]; then
        echo "Error: unsupported investigation '$investigation'." >&2
        exit 2
    fi

    echo "$template_path"
}

subagent_prompt() {
    local investigation="$1"
    local ir_path="$2"
    local subagent_dir="$3"
    local context_file="${4:-}"
    local report_path="$subagent_dir/report.md"
    local insights_path="$subagent_dir/insights.log"
    local status_path="$subagent_dir/status.json"
    local template_path
    template_path="$(subagent_template_path "$investigation")"

    render_template "$template_path" \
        INVESTIGATION "$investigation" \
        REPO_ROOT "$REPO_ROOT" \
        IR_PATH "$ir_path" \
        SUBAGENT_DIR "$subagent_dir" \
        REPORT_PATH "$report_path" \
        INSIGHTS_PATH "$insights_path" \
        STATUS_PATH "$status_path" \
        CONTEXT_FILE "${context_file:-<none>}"
}

run_template_subagent() {
    local subagent_name="$1"
    local llm="$2"
    local ir_path="$3"
    local out_dir="$4"
    local context_file="${5:-}"
    local subagent_dir="$out_dir/$subagent_name"
    local prompt_file="$subagent_dir/prompt.md"
    local llm_output="$subagent_dir/agent.log"
    local report_path="$subagent_dir/report.md"
    local insights_path="$subagent_dir/insights.log"
    local status_path="$subagent_dir/status.json"
    local skip_flag

    validate_llm "$llm"
    subagent_template_path "$subagent_name" >/dev/null
    skip_flag="$(skip_flag_for_llm "$llm")"
    mkdir -p "$subagent_dir"

    log_line "$out_dir/run.log" "Starting $subagent_name with $llm $skip_flag"
    if [[ -n "$context_file" ]]; then
        log_line "$out_dir/run.log" "$subagent_name retry context: $context_file"
    fi

    if [[ ! -e "$ir_path" ]]; then
        printf 'INSIGHT: IR input does not exist: %s\n' "$ir_path" >"$insights_path"
        json_status "needs_context" "IR input does not exist: $ir_path" "$report_path" "$llm_output" "provide-valid-ir-path" "$status_path"
        return 2
    fi

    if [[ -n "$context_file" && ! -e "$context_file" ]]; then
        printf 'INSIGHT: Retry context file does not exist: %s\n' "$context_file" >"$insights_path"
        json_status "needs_context" "Retry context file does not exist: $context_file" "$report_path" "$llm_output" "provide-valid-context-file,retry-without-context" "$status_path"
        return 2
    fi

    subagent_prompt "$subagent_name" "$ir_path" "$subagent_dir" "$context_file" >"$prompt_file"

    set +e
    run_llm_print "$llm" "$prompt_file" "$llm_output"
    local rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
        {
            printf 'INSIGHT: %s agent exited with code %s.\n' "$subagent_name" "$rc"
            printf 'INSIGHT: Review %s and retry with more context if the failure is input-specific.\n' "$llm_output"
        } >>"$insights_path"
        json_status "failed" "LLM subagent exited with code $rc" "$report_path" "$llm_output" "retry-with-prior-logs,provide-more-context" "$status_path"
        log_line "$out_dir/run.log" "$subagent_name failed with exit code $rc"
        return "$rc"
    fi

    if [[ ! -s "$status_path" ]]; then
        json_status "needs_context" "Subagent completed without writing status.json" "$report_path" "$llm_output" "retry-with-prior-logs,provide-more-context" "$status_path"
    elif ! valid_status_json "$status_path"; then
        json_status "needs_context" "Subagent wrote malformed status.json" "$report_path" "$llm_output" "retry-with-prior-logs,provide-more-context" "$status_path"
    fi

    if [[ ! -s "$insights_path" ]]; then
        printf 'INSIGHT: %s completed, but did not emit specific insights. Check the report and agent log.\n' "$subagent_name" >"$insights_path"
    fi

    if [[ ! -s "$report_path" ]]; then
        json_status "needs_context" "Subagent completed without writing report.md" "$report_path" "$llm_output" "retry-with-prior-logs,provide-more-context" "$status_path"
    fi

    log_line "$out_dir/run.log" "$subagent_name completed; status: $status_path"
}

run_subagent() {
    local investigation="${1:-}"
    local llm="${2:-}"
    local ir_path="${3:-}"
    local out_dir="${4:-}"
    local context_file="${5:-}"

    if [[ -z "$investigation" || -z "$llm" || -z "$ir_path" || -z "$out_dir" ]]; then
        usage >&2
        exit 2
    fi

    mkdir -p "$out_dir"

    run_template_subagent "$investigation" "$llm" "$ir_path" "$out_dir" "$context_file"
}

main() {
    case "${1:-}" in
        --help|-h)
            usage
            ;;
        --list-investigations)
            list_investigations
            ;;
        --run-subagent)
            shift
            run_subagent "$@"
            ;;
        "")
            local llm
            llm="$(prompt_for_llm)"
            validate_llm "$llm"
            launch_wrapper_agent "$llm"
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
