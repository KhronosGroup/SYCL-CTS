/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests multi_ptr convert assignment operators for core types
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_convert_assignment_ops.h"

namespace multi_ptr_convert_assignment_ops_core {

TEST_CASE("Convert assignment operators. core types", "[multi_ptr]") {
  using namespace multi_ptr_convert_assignment_ops;
  auto types = multi_ptr_convert::get_types();
  auto composite_types = multi_ptr_convert::get_composite_types();
  for_all_types<check_multi_ptr_convert_assign_for_type>(types);
  for_all_types<check_multi_ptr_convert_assign_for_type>(composite_types);
}

}  // namespace multi_ptr_convert_assignment_ops_core
