#!/bin/bash

set -o errexit -o pipefail -o noclobber -o nounset

cmake . -G Ninja -B build \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DSYCL_IMPLEMENTATION=SimSYCL \
    -DCMAKE_PREFIX_PATH=/sycl \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS=0 \
    -DSYCL_CTS_ENABLE_HALF_TESTS=0 \
    -DSYCL_CTS_ENABLE_FULL_CONFORMANCE=0 \
    -DSYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS=1 \
    $@
