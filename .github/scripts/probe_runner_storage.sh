#!/usr/bin/env bash
# TEMPORARY (remove before landing): find out whether anything on a runner
# outlives the job, i.e. whether a persistent uv cache is possible.
#
# Prints the mounts other than the container's own overlay root, then, in each
# writable candidate directory, the markers earlier jobs left and a new one.
# A marker from a different run or job means that directory persists.
set -uo pipefail

echo "== runner: host=$(hostname) runner=${RUNNER_NAME:-?} node=${NODE_NAME:-${KUBERNETES_NODE_NAME:-?}} run=${GITHUB_RUN_ID:-?}/${GITHUB_RUN_ATTEMPT:-?} job=${GITHUB_JOB:-?}"

echo "== mounts (excluding overlay/proc/sys/cgroup/dev pseudo-filesystems)"
findmnt -rn -o TARGET,SOURCE,FSTYPE,SIZE 2>/dev/null \
  | awk '$3 !~ /^(overlay|proc|sysfs|cgroup2?|devpts|mqueue|tmpfs|devtmpfs|securityfs|debugfs|tracefs|bpf|pstore|configfs|fusectl|hugetlbfs|autofs|binfmt_misc|nsfs|rpc_pipefs)$/' \
  || grep -vE ' (overlay|proc|sysfs|cgroup2?|devpts|mqueue|tmpfs) ' /proc/mounts

echo "== disk usage of candidate directories"
CANDIDATES=(
  "${HOME:-/home/runner}/.cache"
  "${RUNNER_TOOL_CACHE:-/opt/hostedtoolcache}"
  "${RUNNER_WORKSPACE:-}"
  /workspace
  /mnt
  /data
  /tmp
)
for d in "${CANDIDATES[@]}"; do
  [ -n "${d}" ] && [ -d "${d}" ] && df -h "${d}" | tail -1 | sed "s|^|${d}: |"
done

echo "== persistence markers"
STAMP="$(date -u +%FT%TZ) run=${GITHUB_RUN_ID:-?}/${GITHUB_RUN_ATTEMPT:-?} job=${GITHUB_JOB:-?} host=$(hostname)"
for d in "${CANDIDATES[@]}"; do
  [ -n "${d}" ] || continue
  mkdir -p "${d}" 2>/dev/null || true
  [ -w "${d}" ] || { echo "${d}: not writable"; continue; }
  f="${d}/.ci-persist-probe"
  if [ -s "${f}" ]; then
    echo "${d}: PERSISTS, earlier markers:"
    sed 's/^/    /' "${f}"
  else
    echo "${d}: no earlier marker"
  fi
  echo "${STAMP}" >> "${f}"
done
exit 0
