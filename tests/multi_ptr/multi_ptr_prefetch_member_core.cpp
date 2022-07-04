/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests multi_ptr prefetch member for core types
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_prefetch_member.h"

namespace multi_ptr_prefetch_member_core {

TEST_CASE("Prefetch member. core types", "[multi_ptr]") {
  using namespace multi_ptr_prefetch_member;
  auto types = multi_ptr_convert::get_types();
  auto composite_types = multi_ptr_convert::get_composite_types();
  for_all_types<check_multi_ptr_prefetch_for_type>(types);
  for_all_types<check_multi_ptr_prefetch_for_type>(composite_types);
}

}  // namespace multi_ptr_prefetch_member_core
