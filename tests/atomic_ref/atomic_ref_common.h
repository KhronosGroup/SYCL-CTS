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
//  Provides common functions for the sycl::atomic_ref tests.
//
*******************************************************************************/

#ifndef SYCL_CTS_ATOMIC_REF_COMMON_H
#define SYCL_CTS_ATOMIC_REF_COMMON_H

#include "../../util/accuracy.h"
#include "../common/common.h"
#include "../common/section_name_builder.h"
#include "../common/type_coverage.h"
#include "../common/type_list.h"

namespace atomic_ref::tests::common {
using namespace sycl_cts;

constexpr int expected_val = 42;
constexpr int changed_val = 1;

/**
 * @brief Function helps to get string section name that will contain template
 * parameters and function arguments
 *
 * @tparam Dimension Integer representing dimension
 * @param type_name String with name of the testing type
 * @param memory_order_name String with name of the testing memory_order
 * @param memory_scope_name String with name of the testing memory_scope
 * @param address_space String with name of the address_space
 * @param section_description String with human-readable description of the test
 * @return std::string String with name for section
 */
inline std::string get_section_name(const std::string& type_name,
                                    const std::string& memory_order_name,
                                    const std::string& memory_scope_name,
                                    const std::string& address_space_name,
                                    const std::string& section_description) {
  return section_name(section_description)
      .with("T", type_name)
      .with("memory_order", memory_order_name)
      .with("memory_scope", memory_scope_name)
      .with("address_space", address_space_name)
      .create();
}

/**
 * @brief Function helps to get string section name that will contain template
 * parameters and function arguments
 *
 * @tparam Dimension Integer representing dimension
 * @param type_name String with name of the testing type
 * @param memory_order_name String with name of the testing memory_order
 * @param memory_scope_name String with name of the testing memory_scope
 * @param address_space String with name of the address_space
 * @param memory_order sycl::memory_order which will be used as parameter of
 * atomic_ref method
 * @param momory_scope sycl::memory_scope which will be used as parameter of
 * atomic_ref method
 * @param section_description String with human-readable description of the test
 * @return std::string String with name for section
 */
inline std::string get_section_name(const std::string& type_name,
                                    const std::string& memory_order_name,
                                    const std::string& memory_scope_name,
                                    const std::string& address_space_name,
                                    const sycl::memory_order& memory_order,
                                    const sycl::memory_scope& memory_scope,
                                    const std::string& section_description) {
  return section_name(section_description)
      .with("T", type_name)
      .with("memory_order", memory_order_name)
      .with("memory_scope", memory_scope_name)
      .with("address_space", address_space_name)
      .with("memory_order arg", memory_order)
      .with("memory_scope arg", memory_scope)
      .create();
}

/**
 * @brief Factory function for getting type_pack with fp64 type
 */
inline auto get_atomic64_types() {
  static const auto types =
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
      named_type_pack<long long, unsigned long long>::generate(
          "long long", "unsigned long long");
#else
      named_type_pack<long long>::generate("long long");
#endif
  return types;
}

/**
 * @brief Factory function for getting type_pack with all generic types
 */
inline auto get_full_conformance_type_pack() {
  static const auto types =
      named_type_pack<int, unsigned int, long int, unsigned long int,
                      float>::generate("int", "unsigned int", "long int",
                                       "unsigned long int", "float");
  return types;
}

/**
 * @brief Factory function for getting type_pack with generic types
 */
inline auto get_lightweight_type_pack() {
  static const auto types =
      named_type_pack<int, float>::generate("int", "float");
  return types;
}

/**
 * @brief Factory function for getting type_pack with types that depends on full
 *        conformance mode enabling status
 * @return lightweight or full named_type_pack
 */
inline auto get_conformance_type_pack() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return get_full_conformance_type_pack();
#else
  return get_lightweight_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
}

/**
 * @brief Factory function for getting type_pack with all pointers types
 */
inline auto get_full_conformance_pointers_type_pack() {
  static const auto types =
      named_type_pack<int*, unsigned int*, long int*, unsigned long int*,
                      long long*, unsigned long long*, float*,
                      double*>::generate("int *", "unsigned int *",
                                         "long int *", "unsigned long int *",
                                         "long long *", "unsigned long long *",
                                         "float *", "double *");
  return types;
}

/**
 * @brief Factory function for getting type_pack with generic pointers types
 */
inline auto get_lightweight_pointers_type_pack() {
  static const auto types =
      named_type_pack<int*, float*>::generate("int *", "float *");
  return types;
}

/**
 * @brief Factory function for getting type_pack with pointers types that
 * depends on full conformance mode enabling status
 * @return lightweight or full named_type_pack for pointers types
 */
inline auto get_conformance_pointers_type_pack() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return get_full_conformance_pointers_type_pack();
#else
  return get_lightweight_pointers_type_pack();
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
}

/**
 * @brief Factory function for getting type_pack with memory_order values
 */
inline auto get_memory_orders() {
  static const auto memory_orders =
      value_pack<sycl::memory_order, sycl::memory_order::relaxed,
                 sycl::memory_order::acq_rel,
                 sycl::memory_order::seq_cst>::generate_named();
  return memory_orders;
}

/**
 * @brief Factory function for getting type_pack with memory_scope values
 */
inline auto get_memory_scopes() {
  static const auto memory_scopes =
      value_pack<sycl::memory_scope, sycl::memory_scope::work_item,
                 sycl::memory_scope::sub_group, sycl::memory_scope::work_group,
                 sycl::memory_scope::device,
                 sycl::memory_scope::system>::generate_named();
  return memory_scopes;
}

/**
 * @brief Factory function for getting type_pack with address_space values
 */
inline auto get_address_spaces() {
  static const auto address_spaces =
      value_pack<sycl::access::address_space,
                 sycl::access::address_space::global_space,
                 sycl::access::address_space::local_space,
                 sycl::access::address_space::generic_space>::generate_named();
  return address_spaces;
}

// FIXME: re-enable when sycl::info::device::atomic_memory_order_capabilities
// and sycl::info::device::atomic_memory_scope_capabilities are implemented in
// hipsycl
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

inline bool memory_order_is_supported(sycl::queue& q,
                                      sycl::memory_order order) {
  std::vector<sycl::memory_order> memory_orders_supported =
      q.get_device()
          .get_info<sycl::info::device::atomic_memory_order_capabilities>();
  auto it = std::find(memory_orders_supported.begin(),
                      memory_orders_supported.end(), order);
  return it != memory_orders_supported.end();
}

inline bool memory_scope_is_suppoted(sycl::queue& q, sycl::memory_scope scope) {
  std::vector<sycl::memory_scope> memory_scopes_supported =
      q.get_device()
          .get_info<sycl::info::device::atomic_memory_scope_capabilities>();
  auto it = std::find(memory_scopes_supported.begin(),
                      memory_scopes_supported.end(), scope);
  return it != memory_scopes_supported.end();
}

inline bool memory_order_and_scope_are_supported(sycl::queue& q,
                                                 sycl::memory_order order,
                                                 sycl::memory_scope scope) {
  return memory_order_is_supported(q, order) &&
         memory_scope_is_suppoted(q, scope);
}

inline bool memory_order_and_scope_are_not_supported(sycl::queue& q,
                                                     sycl::memory_order order,
                                                     sycl::memory_scope scope) {
  return !memory_order_and_scope_are_supported(q, order, scope);
}

#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL
/**
 * @brief Function to compare two floating point values
 */
template <typename T>
bool compare_floats(T actual, T expected) {
  const T difference = static_cast<T>(std::fabs(actual - expected));
  const T difference_expected = get_ulp_sycl(expected);

  return difference <= difference_expected;
}

inline bool device_has_not_aspect_atomic64() {
  auto queue = sycl_cts::util::get_cts_object::queue();
  return !queue.get_device().has(sycl::aspect::atomic64);
}

template <typename T, typename = std::enable_if_t<std::is_pointer_v<T>>>
inline bool is_64_bits_pointer() {
  constexpr uint32_t bytes_in_64_bits = 8;
  return sizeof(T) == bytes_in_64_bits;
}

}  // namespace atomic_ref::tests::common

#endif  // SYCL_CTS_ATOMIC_REF_COMMON_H
