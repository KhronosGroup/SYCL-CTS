/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Type packs for device_global tests type coverage.
//
*******************************************************************************/

#ifndef SYCL_CTS_TEST_DEVICE_GLOBAL_TYPE_PACK_H
#define SYCL_CTS_TEST_DEVICE_GLOBAL_TYPE_PACK_H

#include "../../common/type_coverage.h"
#include "../../common/type_list.h"

namespace device_global_types {
inline auto get_types() {
  static const auto types =
      named_type_pack<int, bool, user_def_types::no_cnstr>{"int", "bool",
                                                           "no_cnstr"};
  return types;
}
}  // namespace device_global_types

#endif  // SYCL_CTS_TEST_DEVICE_GLOBAL_TYPE_PACK_H
