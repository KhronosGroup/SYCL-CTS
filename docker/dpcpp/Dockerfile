# DPC++ version (git revision) to install
ARG IMPL_VERSION

FROM khronosgroup/sycl-cts-ci:common

ARG IMPL_VERSION
RUN test -n "$IMPL_VERSION" || ( echo "Error: IMPL_VERSION is not set"; exit 1 )

RUN git clone https://github.com/intel/llvm.git \
      --branch=sycl --single-branch --shallow-since=2021-09-01 \
      --recurse-submodules /tmp/dpcpp && \
    cd /tmp/dpcpp && \
    git checkout $IMPL_VERSION

RUN python3 /tmp/dpcpp/buildbot/configure.py \
      --src-dir=/tmp/dpcpp \
      --obj-dir=/tmp/build \
      --build-type=Release \
      --cmake-opt=-DCMAKE_INSTALL_PREFIX=/sycl && \
    python3 /tmp/dpcpp/buildbot/compile.py \
      --src-dir=/tmp/dpcpp \
      --obj-dir=/tmp/build

COPY configure.sh /scripts/
