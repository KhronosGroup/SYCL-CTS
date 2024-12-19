# hipSYCL version (git revision) to install
ARG IMPL_VERSION

FROM khronosgroup/sycl-cts-ci:common

ARG IMPL_VERSION
RUN test -n "$IMPL_VERSION" || ( echo "Error: IMPL_VERSION is not set"; exit 1 )

RUN apt update && \
    apt install -y --no-install-recommends \
    lsb-release \
    software-properties-common \
    gnupg

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 && \
    apt update && \
    apt install -y --no-install-recommends \
    libclang-18-dev clang-tools-18 libomp-18-dev

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt install -y --no-install-recommends \
      libboost-context-dev \
      libboost-fiber-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN git clone https://github.com/illuhad/hipSYCL.git \
      --branch=develop --single-branch --shallow-since=2021-08-01 \
      --recurse-submodules /tmp/hipSYCL && \
    cd /tmp/hipSYCL && \
    git checkout $IMPL_VERSION && \
    cmake /tmp/hipSYCL -G Ninja -B /tmp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/sycl && \
    cmake --build /tmp/build --target install && \
    rm -rf /tmp/hipSYCL /tmp/build

COPY configure.sh /scripts/
