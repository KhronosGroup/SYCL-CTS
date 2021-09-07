# hipSYCL version (git revision) to install
ARG IMPL_VERSION

FROM khronosgroup/sycl-cts-ci:common

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
    git checkout $IMPL_VERSION

RUN cmake /tmp/hipSYCL -G Ninja -B /tmp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/sycl && \
    cmake --build /tmp/build --target install

COPY configure.sh /scripts/
