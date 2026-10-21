/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Checks that <sycl/khr/split_headers/reduction.hpp>
//  provides the APIs listed for it in the header file table of the
//  sycl_khr_properties extension. These tests only apply if the
//  implementation also implements sycl_khr_split_headers, and this file
//  deliberately does not include <sycl/sycl.hpp>.
//
*******************************************************************************/

#include <catch2/catch_test_macros.hpp>

#if SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS

#include <sycl/khr/split_headers/reduction.hpp>
#include <sycl/khr/split_headers/span.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace khr_properties::tests {

struct header_test_combiner {
  int operator()(const int& a, const int& b) const { return a + b; }
};

TEST_CASE(
    "the reduction header defines the reduction tag, the "
    "initialize_to_identity property and key, and the new reduction overloads",
    "[khr_properties][headers]") {
  using prop_t = sycl::khr::property::initialize_to_identity;
  using tag_t = sycl::khr::property::tag::reduction;
  STATIC_CHECK(std::is_trivially_copyable_v<prop_t>);
  STATIC_CHECK(std::is_trivially_copyable_v<
               sycl::khr::property::key::initialize_to_identity>);
  (void)static_cast<tag_t*>(nullptr);

  using span_t = sycl::span<int, 2>;
  using ptr_red_t = decltype(sycl::reduction(std::declval<int*>(),
                                             header_test_combiner{}, prop_t{}));
  using ptr_id_red_t = decltype(sycl::reduction(
      std::declval<int*>(), int{}, header_test_combiner{}, prop_t{}));
  using span_red_t = decltype(sycl::reduction(
      std::declval<span_t>(), header_test_combiner{}, prop_t{}));
  using span_id_red_t = decltype(sycl::reduction(
      std::declval<span_t>(), int{}, header_test_combiner{}, prop_t{}));
  STATIC_CHECK_FALSE(std::is_void_v<ptr_red_t>);
  STATIC_CHECK_FALSE(std::is_void_v<ptr_id_red_t>);
  STATIC_CHECK_FALSE(std::is_void_v<span_red_t>);
  STATIC_CHECK_FALSE(std::is_void_v<span_id_red_t>);
}

}  // namespace khr_properties::tests

#endif  // SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS
