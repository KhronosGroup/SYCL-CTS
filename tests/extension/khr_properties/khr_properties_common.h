/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Common helpers for the sycl_khr_properties extension tests.
//
//  The extension requires its APIs to be provided by <sycl/sycl.hpp> and,
//  if the implementation also implements sycl_khr_split_headers, by the
//  header files listed in its header file table. The tests that do not need a
//  device check the APIs at compile time. They include only the split headers
//  when those are available, so that a missing API in a split header is a
//  compilation error, and include <sycl/sycl.hpp> otherwise. This header does
//  not include any CTS helper that includes <sycl/sycl.hpp>.
//
//  Every test uses the queue properties, so this header also includes
//  <sycl/khr/properties.hpp> and the queue header. Each test file includes
//  the other split headers it needs when
//  SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS is 1.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_EXTENSION_KHR_PROPERTIES_COMMON_H
#define __SYCLCTS_TESTS_EXTENSION_KHR_PROPERTIES_COMMON_H

#if __has_include(<sycl/khr/split_headers/version.hpp>)
#include <sycl/khr/split_headers/version.hpp>
#endif

#ifdef SYCL_KHR_SPLIT_HEADERS
#define SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS 1
#include <sycl/khr/properties.hpp>
#include <sycl/khr/split_headers/queue.hpp>
#else
#define SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS 0
#include <sycl/sycl.hpp>
#endif

#include <catch2/catch_test_macros.hpp>

#include <type_traits>
#include <utility>

namespace khr_properties::tests {

namespace khr = sycl::khr;
namespace prop = sycl::khr::property;
namespace key = sycl::khr::property::key;

/** A properties list that contains only queue properties. */
using queue_props_t =
    decltype(khr::properties{prop::in_order{}, prop::enable_profiling{}});

/**
 * A type that is neither a property nor a properties list. Used to check that
 * the property-accepting overloads are removed from overload resolution.
 */
struct not_a_property {};

/**
 * Defines a detection trait `NAME<P>` that is `std::true_type` only if
 * `EXPR` is well-formed, where `P` is the type of the property (or properties
 * list) argument used inside `EXPR`. This is used to verify the
 * "Constraints:" clauses of the specification without writing code that must
 * fail to compile.
 */
#define KHR_PROPERTIES_DETECTOR(NAME, EXPR)                        \
  template <typename P, typename = void>                           \
  struct NAME : std::false_type {};                                \
  template <typename P>                                            \
  struct NAME<P, std::void_t<decltype(EXPR)>> : std::true_type {}; \
  template <typename P>                                            \
  inline constexpr bool NAME##_v = NAME<P>::value;

template <template <typename, typename> class Trait, typename T, typename Class,
          bool Expected>
void check_binary_trait() {
  using base_t = std::integral_constant<bool, Expected>;
  STATIC_CHECK(std::is_base_of_v<base_t, Trait<T, Class>>);
  STATIC_CHECK(Trait<T, Class>::value == Expected);
}

/**
 * Checks the property traits for a class or tag that no property defined by
 * the extension applies to: only the empty properties list applies to it.
 */
template <typename Class>
void check_class_without_properties() {
  check_binary_trait<khr::is_property_for, prop::in_order, Class, false>();
  check_binary_trait<khr::is_property_for, prop::enable_profiling, Class,
                     false>();
  check_binary_trait<khr::is_property_key_for, key::in_order, Class, false>();
  check_binary_trait<khr::is_property_key_for, key::enable_profiling, Class,
                     false>();
  check_binary_trait<khr::is_property_list_for, khr::empty_properties_t, Class,
                     true>();
  check_binary_trait<khr::is_property_list_for, queue_props_t, Class, false>();
}

}  // namespace khr_properties::tests

#endif  // __SYCLCTS_TESTS_EXTENSION_KHR_PROPERTIES_COMMON_H
