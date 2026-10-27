#!/usr/bin/env bash
# Sample the pod's CPU and memory and GPU 0 once a second while a test
# step runs, then print a summary, to tell whether a step is bound by the GPU,
# the CPU or memory.
#
#   usage_monitor.sh start <prefix>   start sampling into <prefix>.csv
#   usage_monitor.sh stop <prefix>    stop sampling and print the summary
#
# CPU and memory come from the pod's cgroup (v2). The GPU comes from nvidia-smi
# on NVIDIA and rocm-smi on AMD; its columns stay empty when neither exists.
# rocm-smi is used rather than amd-smi because amd-smi refuses to run for users
# outside the render and video groups. stop appends the summary to
# $GITHUB_STEP_SUMMARY when it is set.
#
# Neither command may fail or hang the caller: both always exit 0, every SMI
# and Python call runs under timeout (a wedged GPU can hang the SMI tools), and
# the sampler exits by itself once the shell that started it is gone, in case
# stop never runs.
set -uo pipefail

cmd="${1:-}"
prefix="${2:-}"
[ -n "${prefix}" ] || cmd=usage
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Print GPU 0 as util,memory_mib,power_w. rocm-smi's
# CSV columns are looked up by header name, since their order varies.
rocm_gpu() {
  timeout 5 rocm-smi -d 0 --showuse --showmeminfo vram --showpower --csv 2>/dev/null | awk -F, '
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
    NR == 2 {
      printf "%s,%.0f,%s\n", $col["GPU use (%)"], $col["VRAM Total Used Memory (B)"] / 1048576,
        $col["Current Socket Graphics Package Power (W)"]
      exit
    }'
}

sample() {
  local parent="$1"
  echo "time,cpu_usage_usec,memory_bytes,gpu_util_pct,gpu_memory_mib,gpu_power_w"
  while kill -0 "${parent}" 2> /dev/null; do
    local t cpu mem g=",,"
    t="$(date +%s.%N)"
    cpu="$(awk '$1 == "usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null)"
    mem="$(cat /sys/fs/cgroup/memory.current 2>/dev/null)"
    if command -v nvidia-smi > /dev/null; then
      g="$(timeout 5 nvidia-smi --id=0 --format=csv,noheader,nounits \
        --query-gpu=utilization.gpu,memory.used,power.draw 2>/dev/null | tr -d ' ')"
    elif command -v rocm-smi > /dev/null; then
      g="$(rocm_gpu)"
    fi
    echo "${t},${cpu},${mem},${g}"
    sleep 1
  done
}

case "${cmd}" in
  start)
    sample "${PPID}" > "${prefix}.csv" 2> /dev/null &
    echo $! > "${prefix}.pid"
    ;;
  stop)
    # Kill only our own sampler, in case a stale pid file names a reused PID.
    pid="$(cat "${prefix}.pid" 2> /dev/null)"
    if [ -n "${pid}" ] && grep -q usage_monitor "/proc/${pid}/cmdline" 2> /dev/null; then
      kill "${pid}" 2> /dev/null
    fi
    rm -f "${prefix}.pid"
    if command -v nvidia-smi > /dev/null; then
      gpus="$(timeout 10 nvidia-smi --list-gpus 2> /dev/null | wc -l)"
    else
      gpus="$(timeout 10 rocm-smi --showid --csv 2> /dev/null | grep -c '^card')"
    fi
    timeout 60 python3 "${here}/usage_summary.py" "${prefix}.csv" \
      --cpus "$(nproc)" --gpus "${gpus}" \
      --memory-max "$(cat /sys/fs/cgroup/memory.max 2> /dev/null)" \
      --memory-peak "$(cat /sys/fs/cgroup/memory.peak 2> /dev/null)" \
      | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"
    ;;
  *)
    echo "usage: usage_monitor.sh start|stop <prefix>" >&2
    ;;
esac
exit 0
