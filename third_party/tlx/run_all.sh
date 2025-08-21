#!/bin/bash

echo "Hello, $USER! (Facebook-only)"

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

# Run unit tests
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
    echo "Running TLX tutorial kernels"
    pytest python/test/unit/language/test_tlx.py
fi

echo "Run TLX tutorial kernels (correctness|performance|no)? {c|p|n}"
read user_choice

case $user_choice in
    c)
        echo "Verifying correctness of TLX tutorial kernels"
        for k in third_party/tlx/tutorials/*.py; do
            echo "Running $k"
            pytest $k
        done
        ;;
    p)
        echo "Measuring performance of TLX tutorial kernels"
        for k in third_party/tlx/tutorials/*.py; do
            echo "Running $k"
            python $k
        done
        ;;
    n)
        break
        ;;
    *)
        echo "Invalid choice. "
        ;;
esac

# if [ "$(ask)" == "correctness" ]; then
#     echo "Verifying correctness of TLX tutorial kernels"
#     for k in third_party/tlx/tutorials/*.py; do
#         echo "Running $k"
#         pytest $k
#     done
# elif [ $ans == "performance" ]; then
#     echo "Measuring performance of TLX tutorial kernels"
#     for k in third_party/tlx/tutorials/*.py; do
#         echo "Running $k"
#         python $k
#     done
# fi
