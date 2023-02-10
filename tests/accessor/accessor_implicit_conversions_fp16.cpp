/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor implicit conversions for the sycl::half type
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::accessor is implemented
// Issue link https://github.com/intel/llvm/issues/8298
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "accessor_implicit_conversions.h"

namespace accessor_implicit_conversions_fp16 {
using namespace accessor_implicit_conversions;
}  // namespace accessor_implicit_conversions_fp16

#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace accessor_implicit_conversions_fp16 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor implicit conversion. fp16 type",
 "[accessor][generic_accessor][conversion][fp16]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN("Device does not support half precision floating point operations");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  // TODO: implement factory functions for extending type packs and remove
  //       for_all_types/for_type_and_vectors/for_all_types_and_vectors/
  //       for_type_vectors_marray/for_all_types_vectors_marray/
  //       for_type_and_marrays/for_all_types_and_marrays/etc.
  // As result, it would be possible to use for_all_combinations once per test
  // with no wrappers at all, providing any combination of type coverage
  // Specifically for this use case:
  //  - there would be no need in run_implicit_conversion_test wrapper
  //  - we could easily provide coverage for any additional fp16 type with no
  //    need in further wrappers
  //  - we could easily enable additional dependant types coverage with no
  //    need in further wrappers
  // For example:
  //    enum class tcov : unsigned long long { ...
  //
  //    const auto types = get_fp16_types<tcov::scalar | tcov::marray |
  //                                      tcov::vector | tcov::variant>();
  //    for_all_combinations<test_conversion_within_command>(
  //        types, targets, dimensions);
  //
  for_type_vectors_marray<run_test_generic, sycl::half>("sycl::half");
#else
  run_test_generic<sycl::half>{}("sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("The sycl::local_accessor implicit conversion. fp16 type",
 "[accessor][local_accessor][conversion][fp16]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN("Device does not support half precision floating point operations");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_test_local, sycl::half>("sycl::half");
#else
  run_test_local<sycl::half>{}("sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("The sycl::host_accessor implicit conversion. fp16 type",
 "[accessor][host_accessor][conversion][fp16]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN("Device does not support half precision floating point operations");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_test_host, sycl::half>("sycl::half");
#else
  run_test_host<sycl::half>{}("sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

}  // namespace accessor_implicit_conversions_fp16
