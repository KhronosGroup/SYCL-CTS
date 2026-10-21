/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Checks that <sycl/khr/split_headers/queue.hpp>
//  provides the APIs listed for it in the header file table of the
//  sycl_khr_properties extension. These tests only apply if the
//  implementation also implements sycl_khr_split_headers, and this file
//  deliberately does not include <sycl/sycl.hpp>.
//
*******************************************************************************/

#include <catch2/catch_test_macros.hpp>

#if SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS

#include <sycl/khr/split_headers/queue.hpp>

#include <type_traits>
#include <utility>

namespace khr_properties::tests {

TEST_CASE(
    "the queue header defines the queue properties, their keys, the new "
    "queue constructors and member functions",
    "[khr_properties][headers]") {
  using sycl::queue;
  using in_order_t = sycl::khr::property::in_order;
  using profiling_t = sycl::khr::property::enable_profiling;
  STATIC_CHECK(std::is_trivially_copyable_v<in_order_t>);
  STATIC_CHECK(std::is_trivially_copyable_v<profiling_t>);
  STATIC_CHECK(
      std::is_trivially_copyable_v<sycl::khr::property::key::in_order>);
  STATIC_CHECK(
      std::is_trivially_copyable_v<sycl::khr::property::key::enable_profiling>);

  using dev_t = const sycl::device&;
  using ctx_t = const sycl::context&;
  using ah_t = const sycl::async_handler&;
  STATIC_CHECK(std::is_constructible_v<queue, in_order_t>);
  STATIC_CHECK(std::is_constructible_v<queue, ah_t, in_order_t>);
  STATIC_CHECK(std::is_constructible_v<queue, dev_t, profiling_t>);
  STATIC_CHECK(std::is_constructible_v<queue, dev_t, ah_t, profiling_t>);
  STATIC_CHECK(std::is_constructible_v<queue, ctx_t, dev_t, in_order_t>);
  STATIC_CHECK(std::is_constructible_v<queue, ctx_t, dev_t, ah_t, in_order_t>);

  STATIC_CHECK(
      std::is_same_v<decltype(std::declval<const queue&>().is_in_order()),
                     bool>);
  STATIC_CHECK(std::is_same_v<decltype(std::declval<const queue&>()
                                           .khr_is_profiling_enabled()),
                              bool>);
}

}  // namespace khr_properties::tests

#endif  // SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS
