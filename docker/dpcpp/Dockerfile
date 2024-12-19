# DPC++ build version (git revision) to install
# Go to https://github.com/intel/llvm/pkgs/container/llvm%2Fsycl_ubuntu2204_nightly to see avilable docker image tags
ARG IMPL_VERSION

FROM ghcr.io/intel/llvm/sycl_ubuntu2204_nightly:no-drivers-$IMPL_VERSION

# Make sure that apt is executed with root previlegies
USER root

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt install -y --no-install-recommends \
      ccache && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

COPY configure.sh /scripts/
