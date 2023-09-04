/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor implicit conversions for the double type
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

#include "accessor_common.h"
#include "accessor_implicit_conversions.h"

namespace accessor_implicit_conversions_fp64 {
using namespace accessor_implicit_conversions;
}  // namespace accessor_implicit_conversions_fp64

#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace accessor_implicit_conversions_fp64 {

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("Generic sycl::accessor implicit conversion. fp64 type",
 "[accessor][generic_accessor][conversion][fp64]", generic_test_combinations)({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN("Device does not support double precision floating point operations");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_test_generic, double, TestType>("double");
#else
  run_test_generic<double, TestType>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("The sycl::local_accessor implicit conversion. fp64 type",
 "[accessor][local_accessor][conversion][fp64]", host_local_test_combinations)({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN("Device does not support double precision floating point operations");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_test_local, double, TestType>("double");
#else
  run_test_local<double, TestType>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

DISABLED_FOR_TEMPLATE_LIST_TEST_CASE(hipSYCL)
("The sycl::host_accessor implicit conversion. fp64 type",
 "[accessor][host_accessor][conversion][fp64]", host_local_test_combinations)({
  auto queue = sycl_cts::util::get_cts_object::queue();

  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN("Device does not support double precision floating point operations");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_test_host, double, TestType>("double");
#else
  run_test_host<double, TestType>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

}  // namespace accessor_implicit_conversions_fp64
