#!/bin/bash

echo "Hello! (Facebook-only)"

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
    pushd build/cmake.linux-x86_64-cpython-3.13/
    lit test -a
    popd
fi


# Run core triton unit tests
echo "Running core Triton python unit tests"
pytest python/test/unit/language/test_tutorial09_warp_specialization.py
pytest python/test/unit/language/test_autows_addmm.py

echo "Run autoWS tutorial kernels"
echo "Verifying correctness of FA tutorial kernels"
TRITON_USE_META_PARTITION=1 TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1 pytest python/tutorials/fused-attention-ws-device-tma.py
TRITON_USE_META_PARTITION=1 TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1 python python/tutorials/test_tlx_bwd_from_fused_attention.py
