/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::local_accessor test for the sycl::half type
//
*******************************************************************************/
#include "../common/common.h"

// FIXME: re-enable when sycl::local_accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "local_accessor_constructors.h"
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace local_accessor_constructors_fp16 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::local_accessor constructors. fp16 type", "[accessor]")({
  using namespace local_accessor_constructors;
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp16)) {
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
    for_type_vectors_marray<run_local_constructors_test, sycl::half>(
        "sycl::half");
#else
    run_local_constructors_test<sycl::half>{}("sycl::half");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  } else {
    WARN("Device does not support half precision floating point operations");
    return;
  }
});
}  // namespace local_accessor_constructors_fp16
