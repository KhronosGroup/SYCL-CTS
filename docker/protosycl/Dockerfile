FROM khronosgroup/sycl-cts-ci:common

ARG IMPL_VERSION
RUN test -n "$IMPL_VERSION" || ( echo "Error: IMPL_VERSION is not set"; exit 1 )

# Install dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    lsb-release \
    g++-15 \
    rm -rf /var/lib/apt/lists/*

# Install LLVM 22.
RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 22 all && \
    rm llvm.sh

# Set clang-22 as the default clang version.
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-22 100
ENV CC=clang
ENV CXX=clang++

# Build ProtoSYCL.
RUN git clone https://github.com/0x12CC/ProtoSYCL.git && \
    cd ProtoSYCL && \
    git checkout $IMPL_VERSION && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja && \
    ninja

# Copy the configure script to the /scripts directory.
COPY configure.sh /scripts/
