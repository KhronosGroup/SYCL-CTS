/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor default values.
//
//  This test provides verifications that template parameters has default values
//  for generic accessor, host_accessor and local_accessor with generic types.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "accessor_common.h"

// FIXME: re-enable when sycl::accessor is implemented
// Issue link https://github.com/intel/llvm/issues/8298
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP

#include "accessor_default_values.h"

using namespace accessor_default_values_test;
using namespace accessor_tests_common;
#endif

namespace accessor_default_values_test_core {
using namespace sycl_cts;

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Accessors constructor default values test core types.", "[accessor]")({
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_tests>(types);
});

}  // namespace accessor_default_values_test_core
