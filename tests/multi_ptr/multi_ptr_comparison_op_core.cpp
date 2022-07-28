/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests multi_ptr comparison operators for core types
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_comparison_op.h"
#include "multi_ptr_common.h"

namespace multi_ptr_comparison_op_core {

TEST_CASE("multi_ptr comparison operators. Core types", "[multi_ptr]") {
  using namespace multi_ptr_comparison_op;
  auto types = multi_ptr_convert::get_types();
  auto composite_types = multi_ptr_convert::get_composite_types();

  for_all_types<check_multi_ptr_comparison_op_for_type>(types);
  for_all_types<check_multi_ptr_comparison_op_for_type>(composite_types);
}

}  // namespace multi_ptr_comparison_op_core
