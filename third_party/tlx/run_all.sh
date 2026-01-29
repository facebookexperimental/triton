#!/bin/bash

echo "Hello! (Facebook-only)"

# Build
ask() {
    retval=""
    while true; do
        read -p "Need to build triton in this script? {y|n}" yn
        case $yn in
            [Yy]* ) retval="yes"; break;;
            [Nn]* ) retval="no"; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
    echo "$retval"
}
if [ "$(ask)" == "yes" ]; then
    pip install -e . --no-build-isolation
fi

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
ask() {
    retval=""
    while true; do
        read -p "Run core Triton python unit tests? {y|n}" yn
        case $yn in
            [Yy]* ) retval="yes"; break;;
            [Nn]* ) retval="no"; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
    echo "$retval"
}
if [ "$(ask)" == "yes" ]; then
    echo "Running core Triton python unit tests"
    pytest python/test/unit/language/*.py
    pytest python/test/unit/runtime/*.py
    pytest python/test/unit/cuda/*.py
    pytest python/test/unit/tools/*.py
    pytest python/test/unit/instrumentation/*.py
    pytest python/test/unit/*.py
    pytest python/test/regression/*.py
    pytest python/test/backend/test_device_backend.py
fi


# Run TLX unit tests
ask() {
    retval=""
    while true; do
        read -p "Run all TLX unit tests? {y|n}" yn
        case $yn in
            [Yy]* ) retval="yes"; break;;
            [Nn]* ) retval="no"; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
    echo "$retval"
}
if [ "$(ask)" == "yes" ]; then
    echo "Running TLX Unit Tests"
    pytest python/test/unit/language/test_tlx.py
fi

echo "Run TLX tutorial kernels (correctness|performance|no)? {c|p|n}"
read user_choice

case $user_choice in
    c)
        echo "Verifying correctness of TLX tutorial kernels"
        pytest third_party/tlx/tutorials/*.py
        pytest third_party/tlx/tutorials/correctness_test.py
        ;;
    p)
        echo "Measuring performance of TLX tutorial kernels"
        for k in third_party/tlx/tutorials/*.py; do
            echo "Running $k"
            third_party/tlx/denoise.sh python $k
        done
        third_party/tlx/denoise.sh third_party/tlx/tutorials/blackwell_gemm_perf_test.py
        ;;
    n)
        break
        ;;
    *)
        echo "Invalid choice. "
        ;;
esac
