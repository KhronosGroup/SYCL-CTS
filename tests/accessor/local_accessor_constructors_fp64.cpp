/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides sycl::local_accessor test for the double type
//
*******************************************************************************/
#include "../common/common.h"

// FIXME: re-enable when sycl::local_accessor is implemented
// Issue link https://github.com/intel/llvm/issues/8298
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
#include "accessor_common.h"
#include "local_accessor_constructors.h"
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace local_accessor_constructors_fp64 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::local_accessor constructors. fp64 type", "[accessor]")({
  using namespace local_accessor_constructors;
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp64)) {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
    for_type_vectors_marray<run_local_constructors_test, double>("double");
#else
    run_local_constructors_test<double>{}("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  } else {
    WARN("Device does not support double precision floating point operations");
    return;
  }
});
}  // namespace local_accessor_constructors_fp64
