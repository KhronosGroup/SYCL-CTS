/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr local_accessor constructor
//
*******************************************************************************/

#include "../common/type_coverage.h"
#include "multi_ptr_common.h"
#include "multi_ptr_local_accessor_constructor.h"

namespace multi_ptr_accessor_local_constructor_core {

TEST_CASE("constructor multi_ptr(local_accessor<T, dims>). core types",
          "[multi_ptr]") {
  using namespace multi_ptr_local_accessor_constructors;
  auto types = multi_ptr_common::get_types();
  auto composite_types = multi_ptr_common::get_composite_types();
  for_all_types<check_multi_ptr_local_accessor_cnstr_for_type>(types);
  for_all_types<check_multi_ptr_local_accessor_cnstr_for_type>(composite_types);
}

}  // namespace multi_ptr_accessor_local_constructor_core
