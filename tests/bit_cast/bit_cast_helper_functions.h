/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
//  Provides helper functions for the sycl::bit_cast test.
//
*******************************************************************************/
#ifndef SYCL_CTS_BIT_CAST_HELPER_FUNCTIONS_H
#define SYCL_CTS_BIT_CAST_HELPER_FUNCTIONS_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/type_list.h"

namespace bit_cast::tests::helper_functions {
constexpr size_t array_size = 2;

/**
 * @brief Factory function for getting primary type_pack with all generic types
 */
inline auto get_full_primary_type_pack() {
  static const auto types =
      named_type_pack<bool, char, signed char, unsigned char, short,
                      unsigned short, int, unsigned int, long, unsigned long,
                      long long, unsigned long long,
                      float>::generate("bool", "char", "signed char",
                                       "unsigned char", "short",
                                       "unsigned short", "int", "unsigned int",
                                       "long", "unsigned long", "long long",
                                       "unsigned long long", "float");
  return types;
}

/**
 * @brief Factory function for getting primary type_pack with generic types
 */
inline auto get_lightweight_primary_type_pack() {
  static const auto types =
      named_type_pack<int, float, bool>::generate("int", "float", "bool");
  return types;
}

/**
 * @brief Factory function for getting primary type_pack with types that depends
 * on full conformance mode enabling status
 * @return lightweight or full primary named_type_pack
 */
inline auto get_primary_type_pack() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return get_full_primary_type_pack();
#else
  return get_lightweight_primary_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
}

/**
 * @brief Factory function for getting derived type_pack for special primary
 * type
 */
template <typename PrimaryType>
inline auto get_derived_type_pack(const std::string& primary_type_name) {
  static const auto types =
      named_type_pack<PrimaryType, PrimaryType[array_size],
                      sycl::marray<PrimaryType, array_size>,
                      Base<PrimaryType>, Derived<PrimaryType>>::
          generate(primary_type_name,
                   primary_type_name + "[" + std::to_string(array_size) + "]",
                   "sycl::marray<" + primary_type_name + ", " +
                       std::to_string(array_size) + ">",
                   "Base<" + primary_type_name + ">",
                   "Derived<" + primary_type_name + ">");
  return types;
}

}  // namespace bit_cast::tests::helper_functions

#endif  // SYCL_CTS_BIT_CAST_HELPER_FUNCTIONS_H
