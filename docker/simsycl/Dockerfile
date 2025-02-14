# SimSYCL version (git revision) to install
ARG IMPL_VERSION

FROM khronosgroup/sycl-cts-ci:common

ARG IMPL_VERSION
RUN test -n "$IMPL_VERSION" || ( echo "Error: IMPL_VERSION is not set"; exit 1 )

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt install -y --no-install-recommends \
      libboost-context-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN git clone https://github.com/celerity/SimSYCL.git \
      --branch=master --single-branch --shallow-since=2024-12-01 \
      --recurse-submodules /tmp/SimSYCL && \
    cd /tmp/SimSYCL && \
    git checkout $IMPL_VERSION && \
    cmake /tmp/SimSYCL -G Ninja -B /tmp/build \
      -DSIMSYCL_ANNOTATE_SYCL_DEPRECATIONS=OFF \
      -DSIMSYCL_ENABLE_ASAN=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_INSTALL_PREFIX=/sycl && \
    cmake --build /tmp/build --target install && \
    rm -rf /tmp/SimSYCL /tmp/build

COPY configure.sh /scripts/
