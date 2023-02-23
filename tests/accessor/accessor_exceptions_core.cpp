/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor exceptions.
//
//  This test provides verifications that exception really has been thrown for
//  generic accessor, host_accessor and local_accessor with generic types.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

// FIXME: re-enable when sycl::accessor is implemented
// Issue link https://github.com/intel/llvm/issues/8298
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP

#include "accessor_exceptions.h"

using namespace accessor_exceptions_test;
using namespace accessor_tests_common;
#endif

namespace accessor_exceptions_test_core {
using namespace sycl_cts;

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor constructor exceptions test. Core types.",
 "[accessor]")({
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_tests_with_types, generic_accessor>(types);
  for_all_device_copyable_std_containers<run_tests_with_types, generic_accessor>(types);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::local_accessor constructor exceptions test. Core types.",
 "[accessor]")({
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_tests_with_types, local_accessor>(types);
  for_all_device_copyable_std_containers<run_tests_with_types, local_accessor>(types);
});

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::host_accessor constructor exceptions test. Core types.", "[accessor]")({
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_tests_with_types, host_accessor>(types);
  for_all_device_copyable_std_containers<run_tests_with_types, host_accessor>(types);
});

}  // namespace accessor_exceptions_test_core
