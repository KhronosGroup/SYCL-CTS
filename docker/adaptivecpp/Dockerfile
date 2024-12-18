# AdaptiveCpp version (git revision) to install
ARG IMPL_VERSION

FROM khronosgroup/sycl-cts-ci:common

ARG IMPL_VERSION
RUN test -n "$IMPL_VERSION" || ( echo "Error: IMPL_VERSION is not set"; exit 1 )

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt install -y --no-install-recommends \
      libboost-context-dev \
      libboost-fiber-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git \
      --branch=develop --single-branch --shallow-since=2024-12-01 \
      --recurse-submodules /tmp/AdaptiveCpp && \
    cd /tmp/AdaptiveCpp && \
    git checkout $IMPL_VERSION && \
    cmake /tmp/AdaptiveCpp -G Ninja -B /tmp/build \
      -DWITH_SSCP_COMPILER=OFF \
      -DWITH_STDPAR_COMPILER=OFF \
      -DWITH_ACCELERATED_CPU=OFF \
      -DWITH_CUDA_BACKEND=OFF \
      -DWITH_ROCM_BACKEND=OFF \
      -DWITH_OPENCL_BACKEND=OFF \
      -DWITH_CPU_BACKEND=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/sycl && \
    cmake --build /tmp/build --target install && \
    rm -rf /tmp/AdaptiveCpp /tmp/build

COPY configure.sh /scripts/
