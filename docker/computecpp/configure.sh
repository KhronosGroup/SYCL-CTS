#!/bin/bash

set -o errexit -o pipefail -o noclobber -o nounset

cmake . -G Ninja -B build \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DSYCL_IMPLEMENTATION=ComputeCpp \
    -DCMAKE_PREFIX_PATH=/sycl \
    -DCMAKE_BUILD_TYPE=Release \
    -DSYCL_CTS_ENABLE_FULL_CONFORMANCE=0 \
    -DSYCL_CTS_TEST_DEPRECATED_FEATURES=1 \
    -DSYCL_ENABLE_EXT_ONEAPI_SUB_GROUP_MASK_TESTS=0 \
    $@
