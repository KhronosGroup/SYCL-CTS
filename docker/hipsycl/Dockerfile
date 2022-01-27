# hipSYCL version (git revision) to install
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

RUN git clone https://github.com/illuhad/hipSYCL.git \
      --recurse-submodules /tmp/hipSYCL && \
    cd /tmp/hipSYCL && \
    git checkout $IMPL_VERSION && \
    cmake /tmp/hipSYCL -G Ninja -B /tmp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/sycl && \
    cmake --build /tmp/build --target install && \
    rm -rf /tmp/hipSYCL /tmp/build

COPY configure.sh /scripts/
