#!/bin/bash

ask() {
    retval=""
    while true; do
        read -p "$1" yn
        case $yn in
            [Yy]* ) retval="yes"; break;;
            [Nn]* ) retval="no"; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
    echo "$retval"
}

if [ "$(ask "Run all LITs? {y|n}")" == "yes" ]; then
    echo "Running LITs"
    pushd build/cmake.linux-x86_64-cpython-3.13/ || exit 1
    lit test -a
    popd || exit 1
fi

if [ "$(ask "Run all TLX core tests (unit + tlx.ops)? {y|n}")" == "yes" ]; then
    echo "Running TLX core tests"
    pytest python/test/unit/language/test_tlx_*.py python/test/unit/tlx_ops/
fi

if [ "$(ask "Verify correctness of TLX tutorial kernels? {y|n}")" == "yes" ]; then
    echo "Verifying correctness of TLX tutorial kernels"
    pytest third_party/tlx/tutorials/testing/test_correctness.py
fi
