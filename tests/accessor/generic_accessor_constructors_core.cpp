/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides generic sycl::accessor constructors test for generic types
//
*******************************************************************************/

#include "../common/common.h"

// FIXME: re-enable when sycl::accessor is implemented
#if !defined(__HIPSYCL__) && !defined(__COMPUTECPP__) && \
    !defined(__SYCL_COMPILER_VERSION)
#include "accessor_common.h"
#include "generic_accessor_constructors.h"
#endif

#include "../common/disabled_for_test_case.h"

namespace generic_accessor_constructors_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp, DPCPP)
("Generic sycl::accessor constructors. core types", "[accessor]")({
  using namespace generic_accessor_constructors;
  const auto types = get_conformance_type_pack();
  for_all_types_vectors_marray<run_generic_constructors_test>(types);
});

}  // namespace generic_accessor_constructors_core
