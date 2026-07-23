#!/bin/bash
# SKC Phase B measurement driver — one worker process per point (binding
# isolation), FA_DISABLE_2CTA delivered via env before Python starts.
# Usage: run_phase_b.sh {m4|e3|bwd} [out.jsonl]
set -u
PKG="$(cd "$(dirname "$0")/.." && pwd)"
PY="${FA4_PYTHON:?set FA4_PYTHON to the FA4 venv python binary}"
OUT="${2:-$PKG/skc_cute/results_$1.jsonl}"
run() { # name dis2cta worker extra...
  local name=$1 dis=$2 worker=$3; shift 3
  local r
  r=$(timeout 600 env -u LD_LIBRARY_PATH FA_DISABLE_2CTA=$dis "$PY" \
      "$PKG/$worker" "$@" 2>/dev/null | tail -1)
  echo "{\"point\": \"$name\", \"result\": $r}" >> "$OUT"
  echo "$name -> $r"
}

case "$1" in
m4) # E2: three register candidates, fwd, 4 seqlens
  for S in 2048 4096 8192 16384; do
    for REGS in solver_liveness transfer_2cta_tuned upstream_default; do
      run "m4_fwd_${REGS}_S${S}" 1 bench/skc_cute_worker.py \
          --mode fwd --s $S --binding fwd_1cta --regs $REGS
    done
  done ;;
e3) # perturbation ordering at S=16384 (model-predicted monotonicity)
  S=16384
  for KV in 2 1; do
    run "e3_kv${KV}_S${S}" 1 bench/skc_cute_worker.py \
        --mode fwd --s $S --binding fwd_1cta --kv-clamp $KV
  done
  for SP in 64 32 0; do
    run "e3_splitp${SP}_S${S}" 1 bench/skc_cute_worker.py \
        --mode fwd --s $S --binding fwd_1cta --split-p $SP
  done
  run "e3_regs_sm224_S${S}" 1 bench/skc_cute_worker.py \
      --mode fwd --s $S --binding fwd_1cta --regs solver_liveness ;;
bwd) # M3/M4 bwd: stock, 1cta, identity, bound
  for S in 2048 4096 8192 16384; do
    run "bwd_fa4_2cta_S${S}" 0 bench/fa4_worker.py --mode bwd --s $S
    run "bwd_fa4_1cta_S${S}" 1 bench/fa4_worker.py --mode bwd --s $S
    run "bwd_skc_identity_S${S}" 1 bench/skc_cute_worker.py --mode bwd --s $S
    run "bwd_skc_bound_S${S}" 1 bench/skc_cute_worker.py --mode bwd --s $S \
        --binding bwd_1cta --regs transfer_2cta_tuned
  done ;;
*) echo "usage: $0 {m4|e3|bwd} [out.jsonl]"; exit 1 ;;
esac
