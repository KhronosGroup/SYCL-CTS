/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for local_accessor.
//
//  This test provides verifications that local_accessor can access the memory
//  shared among work-items. For generic types.
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

namespace local_accessor_access_among_work_items_core {

DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)
("sycl::local_accessor access among work items. core types", "[accessor]")({
  const auto types = get_conformance_type_pack();

  for_all_types_vectors_marray<
      run_local_accessor_access_among_work_items_tests>(types);
});

}  // namespace local_accessor_access_among_work_items_core
