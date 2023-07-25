/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor exceptions.
//
//  This test provides verifications that exception really has been thrown for
//  generic accessor, host_accessor and local_accessor with double type.
//
*******************************************************************************/
#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "accessor_exceptions.h"

using namespace accessor_exceptions_test;
using namespace accessor_tests_common;
#endif

namespace accessor_exceptions_test_fp64 {
using namespace sycl_cts;

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL, ComputeCpp)
("Generic sycl::accessor constructor exceptions. fp64 type", "[accessor]",
 test_combinations)({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_tests_with_types, double, generic_accessor,
                          TestType>("double");
#else
  run_tests_with_types<double, generic_accessor, TestType>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::local_accessor constructor exceptions. fp64 type", "[accessor]",
 test_combinations)({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_tests_with_types, double, local_accessor,
                          TestType>("double");
#else
  run_tests_with_types<double, local_accessor, TestType>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::host_accessor constructor exceptions. fp64 type", "[accessor]",
 test_combinations)({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_tests_with_types, double, host_accessor,
                          TestType>("double");
#else
  run_tests_with_types<double, host_accessor, TestType>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

}  // namespace accessor_exceptions_test_fp64
