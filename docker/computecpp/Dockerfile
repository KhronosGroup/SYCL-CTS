# CoputeCpp version (major.minor.patch) to install
ARG IMPL_VERSION

# NOTE: This file is currently mostly a no-op because Codeplay
# does not yet offer a way of obtaining ComputeCpp binaries
# progammatically. To work around this, we rely on a manually
# created base container image instead.

FROM khronosgroup/sycl-cts-ci:ccpp-base-$IMPL_VERSION

ARG IMPL_VERSION
RUN test -n "$IMPL_VERSION" || ( echo "Error: IMPL_VERSION is not set"; exit 1 )

COPY configure.sh /scripts/
