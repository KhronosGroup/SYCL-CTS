/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides host_accessor constructors test for generic types
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::host_accessor is implemented
// Issue links https://github.com/intel/llvm/issues/8298
// https://github.com/intel/llvm/issues/8299
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP && \
    !SYCL_CTS_COMPILING_WITH_DPCPP
#include "accessor_common.h"
#include "host_accessor_constructors.h"
#endif

#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

namespace host_accessor_constructors_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("sycl::host_accessor constructors. core types", "[accessor]")({
  using namespace host_accessor_constructors;
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_host_constructors_test>(types);
});

}  // namespace host_accessor_constructors_core
