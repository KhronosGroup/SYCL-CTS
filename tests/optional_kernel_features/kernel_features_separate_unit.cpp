/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides function in separate unit for tests of the optional kernel features
//  with SYCL_EXTERNAL function
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_EXTERNAL
namespace kernel_features_common {
template <typename T, sycl::aspect aspect>
[[sycl::device_has(aspect)]] SYCL_EXTERNAL void
use_feature_function_external_decorated();

template <>
SYCL_EXTERNAL void
use_feature_function_external_decorated<sycl::half, sycl::aspect::fp16>() {
  float temp = 42.25f;
  sycl::half feature(temp);
  feature += 42;
}

template <>
SYCL_EXTERNAL void
use_feature_function_external_decorated<double, sycl::aspect::fp64>() {
  float temp = 42.25f;
  double feature(temp);
  feature += 42;
}

using AtomicRefT =
    sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>;
template <>
SYCL_EXTERNAL void
use_feature_function_external_decorated<AtomicRefT, sycl::aspect::atomic64>() {
  unsigned long long temp = 42;
  AtomicRefT feature(temp);
  feature += 42;
}
}  // namespace kernel_features_common
#endif
