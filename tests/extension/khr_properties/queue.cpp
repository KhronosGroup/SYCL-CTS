/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Compile-time tests for the queue additions of the sycl_khr_properties
//  extension: the new queue constructors, queue::is_in_order and
//  queue::khr_is_profiling_enabled. The khr::property::in_order and
//  khr::property::enable_profiling properties are tested in properties.cpp,
//  and the runtime effects in runtime.cpp.
//
*******************************************************************************/

#include "khr_properties_common.h"

#if SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS
#include <sycl/khr/split_headers/context.hpp>
#include <sycl/khr/split_headers/device.hpp>
#include <sycl/khr/split_headers/exception.hpp>
#include <sycl/khr/split_headers/reduction.hpp>
#endif

namespace khr_properties::tests {

using reduction_props_t =
    decltype(khr::properties{prop::initialize_to_identity{}});

TEST_CASE("the new queue constructors are constrained on PropertyOrList",
          "[khr_properties][queue]") {
  using sycl::queue;
  using dev_t = const sycl::device&;
  using ctx_t = const sycl::context&;
  using ah_t = const sycl::async_handler&;
  using sel_t = decltype(sycl::default_selector_v);

  SECTION("accepted properties and properties lists") {
    STATIC_CHECK(std::is_constructible_v<queue, prop::in_order>);
    STATIC_CHECK(std::is_constructible_v<queue, prop::enable_profiling>);
    STATIC_CHECK(std::is_constructible_v<queue, khr::empty_properties_t>);
    STATIC_CHECK(std::is_constructible_v<queue, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, sel_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, sel_t, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, dev_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, dev_t, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, ctx_t, sel_t, queue_props_t>);
    STATIC_CHECK(
        std::is_constructible_v<queue, ctx_t, sel_t, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, ctx_t, dev_t, queue_props_t>);
    STATIC_CHECK(
        std::is_constructible_v<queue, ctx_t, dev_t, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, ctx_t, dev_t, ah_t,
                                         prop::enable_profiling>);
  }

  SECTION("properties that do not apply to queue are rejected") {
    using p_t = prop::initialize_to_identity;
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ah_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, sel_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, sel_t, ah_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, dev_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, dev_t, ah_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ctx_t, sel_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ctx_t, sel_t, ah_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ctx_t, dev_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ctx_t, dev_t, ah_t, p_t>);
  }

  SECTION("properties lists that do not apply to queue are rejected") {
    using mixed_t = decltype(khr::properties{prop::in_order{},
                                             prop::initialize_to_identity{}});
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, reduction_props_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, mixed_t>);
    STATIC_CHECK_FALSE(
        std::is_constructible_v<queue, dev_t, reduction_props_t>);
    STATIC_CHECK_FALSE(
        std::is_constructible_v<queue, ctx_t, dev_t, ah_t, mixed_t>);
  }

  SECTION("types that are neither properties nor lists are rejected") {
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, not_a_property>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, key::in_order>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, dev_t, not_a_property>);
  }

  SECTION("the DeviceSelector must be a device selector") {
    STATIC_CHECK_FALSE(
        std::is_constructible_v<queue, not_a_property, queue_props_t>);
    STATIC_CHECK_FALSE(
        std::is_constructible_v<queue, ctx_t, not_a_property, queue_props_t>);
  }

  SECTION("the new constructors are explicit") {
    STATIC_CHECK_FALSE(std::is_convertible_v<prop::in_order, queue>);
    STATIC_CHECK_FALSE(std::is_convertible_v<prop::enable_profiling, queue>);
    STATIC_CHECK_FALSE(std::is_convertible_v<queue_props_t, queue>);
    STATIC_CHECK_FALSE(std::is_convertible_v<khr::empty_properties_t, queue>);
  }
}

TEST_CASE(
    "queue::is_in_order and khr_is_profiling_enabled have the specified "
    "signatures",
    "[khr_properties][queue]") {
  using q_t = const sycl::queue&;
  STATIC_CHECK(
      std::is_same_v<decltype(std::declval<q_t>().is_in_order()), bool>);
  STATIC_CHECK(
      std::is_same_v<decltype(std::declval<q_t>().khr_is_profiling_enabled()),
                     bool>);
}

}  // namespace khr_properties::tests
