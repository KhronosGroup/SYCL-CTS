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
use_feature_function_external_decorated(const sycl::accessor<bool, 1>& acc);

#if SYCL_CTS_ENABLE_HALF_TESTS
template <>
SYCL_EXTERNAL void
use_feature_function_external_decorated<sycl::half, sycl::aspect::fp16>(
    const sycl::accessor<bool, 1>& acc) {
  float temp = 42.25f;
  sycl::half feature1(temp);
  sycl::half feature2(temp);
  feature1 += 42;
  acc[0] = (feature1 == feature2);
}
#endif

template <>
SYCL_EXTERNAL void
use_feature_function_external_decorated<double, sycl::aspect::fp64>(
    const sycl::accessor<bool, 1>& acc) {
  float temp = 42.25f;
  double feature1(temp);
  double feature2(temp);
  feature1 += 42;
  acc[0] = (feature1 == feature2);
}

using AtomicRefT =
    sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>;
template <>
SYCL_EXTERNAL void
use_feature_function_external_decorated<AtomicRefT, sycl::aspect::atomic64>(
    const sycl::accessor<bool, 1>& acc) {
  unsigned long long temp = 42;
  AtomicRefT feature1(temp);
  AtomicRefT feature2(temp);
  feature1 += 42;
  acc[0] = (feature1 == feature2);
}
}  // namespace kernel_features_common
#endif
