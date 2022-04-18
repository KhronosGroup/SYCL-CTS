/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor implicit conversions for core types
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "accessor_implicit_conversions.hpp"

namespace accessor_implicit_conversions_core {
using namespace accessor_implicit_conversions;
}  // namespace accessor_implicit_conversions_core

#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace accessor_implicit_conversions_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor implicit conversion. core types",
 "[accessor][generic_accessor][conversion][core]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  const auto types = get_lightweight_type_pack();
#else
  const auto types = get_full_conformance_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE

  for_all_types_vectors_marray<run_test_generic>(types);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("The sycl::local_accessor implicit conversion. core types",
 "[accessor][local_accessor][conversion][core]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  const auto types = get_lightweight_type_pack();
#else
  const auto types = get_full_conformance_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE

  for_all_types_vectors_marray<run_test_local>(types);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("The sycl::host_accessor implicit conversion. core types",
 "[accessor][host_accessor][conversion][core]")({
  auto queue = sycl_cts::util::get_cts_object::queue();

#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  const auto types = get_lightweight_type_pack();
#else
  const auto types = get_full_conformance_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE

  for_all_types_vectors_marray<run_test_host>(types);
});

}  // namespace accessor_implicit_conversions_core
