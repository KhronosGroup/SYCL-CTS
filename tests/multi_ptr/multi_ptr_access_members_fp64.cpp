/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests multi_ptr access members for double type
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_access_members.h"

namespace multi_ptr_access_members_fp64 {

TEST_CASE("Access members. fp64 type", "[multi_ptr]") {
  using namespace multi_ptr_access_members;

  auto queue = sycl_cts::util::get_cts_object::queue();
  if (!queue.get_device().has(sycl::aspect::fp64)) {
    WARN(
        "Device does not support double precision floating point operations. "
        "Skipping the test case.");
    return;
  }
  check_multi_ptr_access_members_for_type<double>{}("double");
}

}  // namespace multi_ptr_access_members_fp64
