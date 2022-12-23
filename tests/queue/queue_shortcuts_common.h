/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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
