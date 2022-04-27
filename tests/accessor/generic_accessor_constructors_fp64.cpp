/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor test for the double type
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "generic_accessor_constructors.hpp"
#endif

#include "../common/disabled_for_test_case.h"

namespace generic_accessor_constructors_fp64 {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor constructors. fp64 type", "[accessor]")({
  using namespace generic_accessor_constructors;
  
  auto queue = sycl_cts::util::get_cts_object::queue();
  if (queue.get_device().has(sycl::aspect::fp64)) {
#ifndef SYCL_CTS_ENABLE_FULL_CONFORMANCE
    run_generic_constructors_test<double>{}("double");
#else
    for_type_vectors_marray<run_generic_constructors_test, double>("double");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  } else {
    WARN("Device does not support double precision floating point operations");
    return;
  }
});
}  // namespace generic_accessor_constructors_fp64
