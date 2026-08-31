#!/bin/bash
set -euo pipefail

echo "Hello! (Facebook-only)"

usage() {
    echo "Usage: $0 [--buck]"
    echo "  --buck  Run LIT and Python tests through Buck using the current GPU."
}

use_buck="no"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --buck)
            use_buck="yes"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
triton_root="$(cd "$script_dir/../../.." && pwd)"
fbsource_root="$(cd "$script_dir/../../../../../../.." && pwd)"
cd "$triton_root" || exit 1

# Defaults for non-buck path under set -u: only populated by configure_buck().
buck_target_prefix=""
tutorial09_target=""
hopper_or_blackwell_fa_target=""
failed=0

configure_buck() {
    local gpu_index="${CUDA_VISIBLE_DEVICES:-0}"
    local gpu_name

    gpu_index="${gpu_index%%,*}"
    if ! gpu_name="$(nvidia-smi -i "$gpu_index" --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
        echo "Unable to query GPU $gpu_index with nvidia-smi." >&2
        exit 1
    fi

    case "$gpu_name" in
        *GB300*|*B300*)
            buck_gpu_family="blackwell"
            buck_nvcc_arch="b300a"
            buck_cuda_version="13.0"
            ;;
        *GB200*|*B200*)
            buck_gpu_family="blackwell"
            buck_nvcc_arch="b200a"
            buck_cuda_version="12.8"
            ;;
        *H200*)
            buck_gpu_family="hopper"
            buck_nvcc_arch="h200a"
            buck_cuda_version="12.8"
            ;;
        *H100*)
            buck_gpu_family="hopper"
            buck_nvcc_arch="h100a"
            buck_cuda_version="12.8"
            ;;
        *)
            echo "Unsupported GPU for Buck test execution: $gpu_name" >&2
            exit 1
            ;;
    esac

    buck_config=(
        "@mode/opt"
        "-m" "ovr_config//triton:beta"
        "-c" "fbcode.nvcc_arch=$buck_nvcc_arch"
        "-m" "ovr_config//third-party/cuda/constraints:$buck_cuda_version"
    )
    buck_target_prefix="fbsource//third-party/triton/beta/triton:"

    if [ "$buck_gpu_family" = "hopper" ]; then
        tutorial09_target="${buck_target_prefix}py_tutorial09_warp_specialization_hopper_test"
        hopper_or_blackwell_fa_target="${buck_target_prefix}py_fused_attention_ws_device_tma_hopper_test"
    else
        tutorial09_target="${buck_target_prefix}py_tutorial09_warp_specialization_blackwell_test"
        hopper_or_blackwell_fa_target="${buck_target_prefix}py_fused_attention_ws_device_tma_blackwell_test"
    fi

    echo "Running through Buck on $gpu_name ($buck_nvcc_arch, CUDA $buck_cuda_version)"
}

run_buck_target() {
    local target="$1"
    shift
    (
        cd "$fbsource_root/fbcode" || exit 1
        buck2 run "${buck_config[@]}" "$target" -- "$@"
    )
}

run_test() {
    local pytest_spec="$1"
    local buck_target="$2"
    shift 2

    if [ "$use_buck" = "yes" ]; then
        # fbcode.nvcc_arch in buck_config selects the local GPU; "$@" forwards
        # pytest -k filters to buck2 run.
        run_buck_target "$buck_target" "$@" || failed=1
    else
        pytest "$pytest_spec" || failed=1
    fi
}

if [ "$use_buck" = "yes" ]; then
    configure_buck
fi

get_cmake_build_dir() {
    local dirs=()
    local dir

    while IFS= read -r dir; do
        dirs+=("$dir")
    done < <(find build -mindepth 1 -maxdepth 1 -type d -name 'cmake.*' \
        -exec test -f '{}/CMakeCache.txt' ';' -print | sort)

    if [ "${#dirs[@]}" -eq 0 ]; then
        echo "No configured CMake build directory found under build/" >&2
        return 1
    fi

    if [ "${#dirs[@]}" -gt 1 ]; then
        echo "Warning: multiple configured CMake build directories found under build/:" >&2
        printf '  %s\n' "${dirs[@]}" >&2
        echo "Using ${dirs[0]}" >&2
    fi

    echo "${dirs[0]}"
}

# Run LIT
ask() {
    retval=""
    while true; do
        read -p "Run all LITs? {y|n}" yn
        case $yn in
            [Yy]* ) retval="yes"; break;;
            [Nn]* ) retval="no"; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
    echo "$retval"
}
if [ "$(ask)" == "yes" ]; then
    echo "Running LITs"
    if [ "$use_buck" = "yes" ]; then
        (
            cd "$fbsource_root/fbcode" || exit 1
            buck2 test "${buck_config[@]}" "${buck_target_prefix}all_lit_tests"
        ) || failed=1
    else
        cmake_build_dir="$(get_cmake_build_dir)" || exit 1
        pushd "$cmake_build_dir"
        lit test -a || failed=1
        popd
    fi
fi


# Run core triton unit tests
echo "Running core Triton python unit tests"
run_test python/test/unit/language/test_tutorial09_warp_specialization.py "$tutorial09_target"
run_test python/test/unit/language/test_autows_addmm.py "${buck_target_prefix}py_autows_addmm_blackwell_test"
run_test python/test/unit/language/test_autows_quantized_matmul.py "${buck_target_prefix}py_autows_quantized_matmul_blackwell_test"
run_test third_party/tlx/tutorials/testing/test_correctness_autows.py "${buck_target_prefix}py_autows_fa_correctness_blackwell_test"

echo "Verifying repeated high-grid 2-CTA CLC FA forward"
( export AUTOWS_FWD_CLC=1 AUTOWS_FWD_NUM_CTAS=2
  run_test third_party/tlx/tutorials/testing/test_correctness_autows.py::test_autows_fa_rescale_opt_clc_repeated_high_grid \
      "${buck_target_prefix}py_autows_fa_correctness_blackwell_test" \
      -k test_autows_fa_rescale_opt_clc_repeated_high_grid )

echo "Run autoWS tutorial kernels"
echo "Verifying correctness of FA tutorial kernels"
run_test third_party/tlx/tutorials/fused_attention_ws_device_tma.py \
    "${buck_target_prefix}py_autows_fa_compiler_dp_blackwell_test"

echo "Verifying correctness of HSTU cross-attention bwd (reduce_dq) autoWS kernel"
run_test third_party/tlx/tutorials/test_cross_attention_bwd_autows.py \
    "${buck_target_prefix}py_autows_cross_attention_bwd_blackwell_test"

echo "Verifying correctness of HSTU self-attention fwd autoWS kernels (including compiler DP)"
run_test third_party/tlx/tutorials/test_self_attention_autows.py \
    "${buck_target_prefix}py_autows_self_attention_blackwell_test"

echo "Verifying correctness of HSTU self-attention bwd (plain Triton vs TLX vs torch)"
run_test third_party/tlx/tutorials/test_self_attention_bwd.py \
    "${buck_target_prefix}py_self_attention_bwd_blackwell_test"

echo "run for Hopper or Blackwell"
run_test python/tutorials/fused-attention-ws-device-tma-hopper-or-blackwell.py \
    "$hopper_or_blackwell_fa_target"

echo "Verifying correctness of auto-TMA FA tutorial kernels"
run_test third_party/tlx/tutorials/fused_attention_ws_auto_tma.py \
    "${buck_target_prefix}py_auto_tma_fa_blackwell_test"
run_test third_party/tlx/tutorials/fused_attention_ws_auto_tma_dp.py \
    "${buck_target_prefix}py_auto_tma_fa_dp_blackwell_test"

exit $failed
