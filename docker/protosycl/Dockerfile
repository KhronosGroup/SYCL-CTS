FROM khronosgroup/sycl-cts-ci:common

ARG IMPL_VERSION
RUN test -n "$IMPL_VERSION" || ( echo "Error: IMPL_VERSION is not set"; exit 1 )

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    apt install -y --no-install-recommends clang \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

ENV CC=clang
ENV CXX=clang++

COPY configure.sh /scripts/
