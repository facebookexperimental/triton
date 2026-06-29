#!/bin/bash

echo "Hello! (Facebook-only)"

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
    cmake_build_dir="$(get_cmake_build_dir)" || exit 1
    pushd "$cmake_build_dir"
    lit test -a
    popd
fi


# Run core triton unit tests
echo "Running core Triton python unit tests"
pytest python/test/unit/language/test_tutorial09_warp_specialization.py
pytest python/test/unit/language/test_autows_addmm.py
pytest third_party/tlx/tutorials/testing/test_correctness_autows.py

echo "Run autoWS tutorial kernels"
echo "Verifying correctness of FA tutorial kernels"
pytest third_party/tlx/tutorials/fused_attention_ws_device_tma.py

echo "run for Hopper or Blackwell"
pytest python/tutorials/fused-attention-ws-device-tma-hopper-or-blackwell.py
