FROM ubuntu:20.04

# We install OpenCL to enable compilation of interop tests (for SYCL
# implementations that support it).
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt install -y --no-install-recommends \
      build-essential \
      wget \
      python3 \
      git \
      ca-certificates \
      cmake \
      ninja-build \
      ccache \
      ocl-icd-opencl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*
