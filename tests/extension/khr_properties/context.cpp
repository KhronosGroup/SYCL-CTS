/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Compile-time tests for the context constructors added by the
//  sycl_khr_properties extension. The runtime effects are tested in
//  runtime.cpp.
//
//  The extension does not define any property that can be used with the
//  context class, and applications cannot define their own properties, so the
//  only PropertyOrList argument that satisfies the constraints is the empty
//  properties list. Properties of other classes are used to check that the
//  constraints remove the constructors from overload resolution.
//
*******************************************************************************/

#include "khr_properties_common.h"

#if SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS
#include <sycl/khr/split_headers/context.hpp>
#include <sycl/khr/split_headers/device.hpp>
#include <sycl/khr/split_headers/exception.hpp>
#include <sycl/khr/split_headers/platform.hpp>
#endif

#include <vector>

namespace khr_properties::tests {

using ctx_dev_t = const sycl::device&;
using ctx_plt_t = const sycl::platform&;
using ctx_devs_t = const std::vector<sycl::device>&;
using ctx_ah_t = sycl::async_handler;

template <typename PropertyOrList>
void check_context_constraints(bool expected) {
  using sycl::context;
  using dev_t = const sycl::device&;
  using plt_t = const sycl::platform&;
  using devs_t = const std::vector<sycl::device>&;
  using ah_t = sycl::async_handler;
  using p_t = PropertyOrList;

  CHECK(std::is_constructible_v<context, p_t> == expected);
  CHECK(std::is_constructible_v<context, ah_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, dev_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, dev_t, ah_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, plt_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, plt_t, ah_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, devs_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, devs_t, ah_t, p_t> == expected);
}

TEST_CASE("the new context constructors are constrained on PropertyOrList",
          "[khr_properties][context]") {
  SECTION("the empty properties list is accepted") {
    check_context_constraints<khr::empty_properties_t>(true);
  }
  SECTION("queue properties are rejected") {
    check_context_constraints<prop::in_order>(false);
    check_context_constraints<prop::enable_profiling>(false);
  }
  SECTION("reduction properties are rejected") {
    check_context_constraints<prop::initialize_to_identity>(false);
  }
  SECTION("properties lists with properties of other classes are rejected") {
    check_context_constraints<queue_props_t>(false);
    check_context_constraints<reduction_props_t>(false);
    check_context_constraints<mixed_props_t>(false);
  }
  SECTION("types that are neither properties nor lists are rejected") {
    check_context_constraints<not_a_property>(false);
  }
  SECTION("the new constructors are explicit") {
    using sycl::context;
    using p_t = khr::empty_properties_t;
    STATIC_CHECK_FALSE(std::is_convertible_v<p_t, context>);
    STATIC_CHECK_FALSE(is_implicitly_constructible_v<context, p_t>);
    STATIC_CHECK_FALSE(is_implicitly_constructible_v<context, ctx_ah_t, p_t>);
    STATIC_CHECK_FALSE(is_implicitly_constructible_v<context, ctx_dev_t, p_t>);
    STATIC_CHECK_FALSE(
        is_implicitly_constructible_v<context, ctx_dev_t, ctx_ah_t, p_t>);
    STATIC_CHECK_FALSE(is_implicitly_constructible_v<context, ctx_plt_t, p_t>);
    STATIC_CHECK_FALSE(
        is_implicitly_constructible_v<context, ctx_plt_t, ctx_ah_t, p_t>);
    STATIC_CHECK_FALSE(is_implicitly_constructible_v<context, ctx_devs_t, p_t>);
    STATIC_CHECK_FALSE(
        is_implicitly_constructible_v<context, ctx_devs_t, ctx_ah_t, p_t>);
  }
}

TEST_CASE("only the empty properties list applies to the context class",
          "[khr_properties][context][traits]") {
  check_class_without_properties<sycl::context>();
}

}  // namespace khr_properties::tests
