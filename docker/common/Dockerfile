FROM ubuntu:20.04

# The CTS currently requires OpenCL to be installed on the system,
# regardless of the SYCL implementation and/or backend used.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt install -y --no-install-recommends \
      build-essential \
      python3 \
      git \
      ca-certificates \
      cmake \
      ninja-build \
      ocl-icd-opencl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*
