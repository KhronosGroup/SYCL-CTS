/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for accessor default values.
//
//  This test provides verifications that template parameters has default values
//  for generic accessor, host_accessor and local_accessor with sycl::half type.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

// FIXME: re-enable when sycl::accessor is implemented
// Issue link https://github.com/intel/llvm/issues/8298
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP

#include "accessor_default_values.h"

using namespace accessor_default_values_test;
using namespace accessor_tests_common;
#endif

namespace accessor_exceptions_test_fp16 {
using namespace sycl_cts;

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Accessors constructor default values test fp16 types.", "[accessor]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN(
        "Device does not support half precision floating point operations. "
        "Skipping the test case.");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_tests, sycl::half>("sycl::half");
#else
  run_tests<sycl::half>{}("sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

}  // namespace accessor_exceptions_test_fp16
