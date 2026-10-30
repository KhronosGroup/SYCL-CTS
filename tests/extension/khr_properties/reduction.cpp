/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Compile-time tests for the reduction overloads added by the
//  sycl_khr_properties extension. The khr::property::initialize_to_identity
//  property is tested in properties.cpp, and the runtime effects in
//  runtime.cpp.
//
*******************************************************************************/

#include "khr_properties_common.h"

#if SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS
#include <sycl/khr/split_headers/functional.hpp>
#include <sycl/khr/split_headers/reduction.hpp>
#include <sycl/khr/split_headers/span.hpp>
#endif

namespace khr_properties::tests {

using reduction_props_t =
    decltype(khr::properties{prop::initialize_to_identity{}});

using plus_t = sycl::plus<int>;
using span2_t = sycl::span<int, 2>;
using dyn_span_t = sycl::span<int, sycl::dynamic_extent>;

// Detection traits for each overload. P is the type of the PropertyOrList
// argument.
KHR_PROPERTIES_DETECTOR(can_reduce_ptr,
                        sycl::reduction(std::declval<int*>(), plus_t{},
                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_span,
                        sycl::reduction(std::declval<span2_t>(), plus_t{},
                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_ptr_identity,
                        sycl::reduction(std::declval<int*>(), int{}, plus_t{},
                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_span_identity,
                        sycl::reduction(std::declval<span2_t>(), int{},
                                        plus_t{}, std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_dyn_span,
                        sycl::reduction(std::declval<dyn_span_t>(), plus_t{},
                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_dyn_span_identity,
                        sycl::reduction(std::declval<dyn_span_t>(), int{},
                                        plus_t{}, std::declval<P>()))

template <typename P>
void check_reduction_constraints(bool expected) {
  CHECK(can_reduce_ptr_v<P> == expected);
  CHECK(can_reduce_span_v<P> == expected);
  CHECK(can_reduce_ptr_identity_v<P> == expected);
  CHECK(can_reduce_span_identity_v<P> == expected);
}

TEST_CASE("the new reduction overloads are constrained on PropertyOrList",
          "[khr_properties][reduction]") {
  SECTION("reduction properties and lists are accepted") {
    check_reduction_constraints<prop::initialize_to_identity>(true);
    check_reduction_constraints<reduction_props_t>(true);
    check_reduction_constraints<khr::empty_properties_t>(true);
  }
  SECTION("queue properties are rejected") {
    check_reduction_constraints<prop::in_order>(false);
    check_reduction_constraints<prop::enable_profiling>(false);
    check_reduction_constraints<queue_props_t>(false);
  }
  SECTION("lists with a property of another class are rejected") {
    using mixed_t = decltype(khr::properties{prop::in_order{},
                                             prop::initialize_to_identity{}});
    check_reduction_constraints<mixed_t>(false);
  }
  SECTION("types that are neither properties nor lists are rejected") {
    check_reduction_constraints<not_a_property>(false);
    check_reduction_constraints<key::initialize_to_identity>(false);
  }
}

TEST_CASE(
    "the new span reduction overloads are not available for "
    "sycl::dynamic_extent",
    "[khr_properties][reduction]") {
  STATIC_CHECK_FALSE(can_reduce_dyn_span_v<prop::initialize_to_identity>);
  STATIC_CHECK_FALSE(can_reduce_dyn_span_v<reduction_props_t>);
  STATIC_CHECK_FALSE(can_reduce_dyn_span_v<khr::empty_properties_t>);
  STATIC_CHECK_FALSE(
      can_reduce_dyn_span_identity_v<prop::initialize_to_identity>);
  STATIC_CHECK_FALSE(can_reduce_dyn_span_identity_v<reduction_props_t>);
  STATIC_CHECK_FALSE(can_reduce_dyn_span_identity_v<khr::empty_properties_t>);
}

}  // namespace khr_properties::tests
