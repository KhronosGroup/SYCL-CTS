FROM ubuntu:22.04

# DPC++ nightly to install
# Go to https://github.com/intel/llvm/releases to see avilable nightly builds.
ARG IMPL_VERSION

# Make sure that apt is executed with root previlegies
USER root

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt install -y --no-install-recommends wget ca-certificates build-essential \
      cmake ninja-build ccache git python3 python3-psutil python-is-python3 python3-pip \
      zstd ocl-icd-opencl-dev vim libffi-dev libva-dev libtool wget sudo zstd zip \
      unzip jq curl libhwloc-dev libzstd-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN mkdir -p /opt/sycl && \
    cd /opt/sycl && \
    wget https://github.com/intel/llvm/releases/download/nightly-${IMPL_VERSION}/sycl_linux.tar.gz && \
    tar xf sycl_linux.tar.gz && \
    rm sycl_linux.tar.gz

ENV PATH="/opt/sycl/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/sycl/lib:$LD_LIBRARY_PATH"

COPY configure.sh /scripts/
