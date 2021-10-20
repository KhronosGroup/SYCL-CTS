#!/bin/bash

set -o errexit -o pipefail -o noclobber -o nounset

cmake . -G Ninja -B build \
    -DSYCL_IMPLEMENTATION=DPCPP \
    -DDPCPP_INSTALL_DIR=/sycl \
    -DCMAKE_CXX_COMPILER=/sycl/bin/clang++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DSYCL_CTS_ENABLE_FULL_CONFORMANCE=0 \
    -DSYCL_CTS_ENABLE_LEGACY_TESTS=1 \
    $@
