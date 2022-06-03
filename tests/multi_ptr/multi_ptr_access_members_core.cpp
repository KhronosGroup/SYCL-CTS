/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests multi_ptr access members for core types
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_access_members.h"

namespace multi_ptr_access_members_core {

TEST_CASE("Access members. core types", "[multi_ptr]") {
  using namespace multi_ptr_access_members;

  auto types = multi_ptr_common::get_types();
  auto composite_types = multi_ptr_common::get_composite_types();
  for_all_types<check_multi_ptr_access_members_for_type>(types);
  for_all_types<check_multi_ptr_access_members_for_type>(composite_types);
}

}  // namespace multi_ptr_access_members_core
