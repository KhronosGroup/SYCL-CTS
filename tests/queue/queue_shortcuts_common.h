/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#ifndef SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_COMMON_H
#define SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_COMMON_H

#include "../common/type_coverage.h"
#include "../common/type_list.h"

namespace queue_shortcuts_common {

inline auto get_types() {
  static const auto types =
      named_type_pack<char, short, int, long long, std::size_t, bool, float,
                      user_def_types::no_cnstr>::generate("char", "short",
                                                          "int", "long long",
                                                          "std::size_t", "bool",
                                                          "float",
                                                          "custom struct");
  static_assert(sycl::is_device_copyable_v<user_def_types::no_cnstr>);
  return types;
}

/** Performs std::iota with + 1 instead of ++ for compatibility with bool. */
template <typename ForwardIt, typename T>
constexpr void iota_comp(ForwardIt first, ForwardIt last, T value) {
  while (first != last) {
    *(first++) = value;
    value = value + 1;
  }
}

}  // namespace queue_shortcuts_common

#endif  // SYCL_CTS_QUEUE_QUEUE_SHORTCUTS_COMMON_H
