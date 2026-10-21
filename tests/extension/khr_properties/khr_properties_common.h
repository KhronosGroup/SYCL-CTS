/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Common helpers for the sycl_khr_properties extension tests.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_EXTENSION_KHR_PROPERTIES_COMMON_H
#define __SYCLCTS_TESTS_EXTENSION_KHR_PROPERTIES_COMMON_H

#include "../../../util/sycl_exceptions.h"
#include "../../common/common.h"
#include "../../common/disabled_for_test_case.h"

#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace khr_properties::tests {

namespace khr = sycl::khr;
namespace prop = sycl::khr::property;
namespace key = sycl::khr::property::key;
namespace tag = sycl::khr::property::tag;

/** The type of an empty properties list. */
using empty_props_t = khr::empty_properties_t;

/** A properties list that contains only queue properties. */
using queue_props_t =
    decltype(khr::properties{prop::in_order{}, prop::enable_profiling{}});

/** A properties list that contains only a reduction property. */
using reduction_props_t =
    decltype(khr::properties{prop::initialize_to_identity{}});

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

/** Returns true if `ptr` is aligned to `alignment` bytes. */
inline bool is_aligned(const void* ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

}  // namespace khr_properties::tests

#endif  // __SYCLCTS_TESTS_EXTENSION_KHR_PROPERTIES_COMMON_H
