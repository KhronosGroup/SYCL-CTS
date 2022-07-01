/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr implicit conversions for core types
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_implicit_conversions.h"

namespace multi_ptr_implicit_conversions_core {

TEST_CASE("multi_ptr implicit conversions. Core types", "[multi_ptr]") {
  using namespace multi_ptr_implicit_conversions;
  auto types = multi_ptr_common::get_types();
  auto composite_types = multi_ptr_common::get_composite_types();
  for_all_types<check_multi_ptr_implicit_convert_for_type>(types);
  for_all_types<check_multi_ptr_implicit_convert_for_type>(composite_types);
}

}  // namespace multi_ptr_implicit_conversions_core
