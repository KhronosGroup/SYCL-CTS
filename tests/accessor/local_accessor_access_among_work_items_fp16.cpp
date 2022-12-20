/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for local_accessor.
//
//  This test provides verifications that local_accessor can access the memory
//  shared among work-items. For the sycl::half type.
//
*******************************************************************************/
#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP

#include "local_accessor_access_among_work_items.h"

using namespace local_accessor_access_among_work_items;
using namespace accessor_tests_common;
#endif

namespace local_accessor_access_among_work_items_fp16 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::local_accessor access among work items. fp16 type", "[accessor]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp16)) {
    WARN(
        "Device does not support half precision floating point operations. "
        "Skipping the test case.");
    return;
  }

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  for_type_vectors_marray<run_local_accessor_access_among_work_items_tests,
                          sycl::half>("sycl::half");
#else
  run_local_accessor_access_among_work_items_tests<sycl::half>{}("sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
});

}  // namespace local_accessor_access_among_work_items_fp16
