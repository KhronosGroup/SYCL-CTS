/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Checks that <sycl/khr/split_headers/usm.hpp>
//  provides the APIs listed for it in the header file table of the
//  sycl_khr_properties extension. These tests only apply if the
//  implementation also implements sycl_khr_split_headers, and this file
//  deliberately does not include <sycl/sycl.hpp>.
//
*******************************************************************************/

#include <catch2/catch_test_macros.hpp>

#if SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS

#include <sycl/khr/properties.hpp>
#include <sycl/khr/split_headers/usm.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace khr_properties::tests {

TEST_CASE(
    "the usm header defines the USM tags, usm_allocator constructors and "
    "allocation function overloads",
    "[khr_properties][headers]") {
  using e_t = sycl::khr::empty_properties_t;
  using sz = std::size_t;
  using q_t = const sycl::queue&;
  using d_t = const sycl::device&;
  using c_t = const sycl::context&;

  // The tag types are declared.
  using device_tag_t = sycl::khr::property::tag::usm_device;
  using host_tag_t = sycl::khr::property::tag::usm_host;
  using shared_tag_t = sycl::khr::property::tag::usm_shared;
  STATIC_CHECK(sycl::khr::is_property_list_for_v<e_t, device_tag_t>);
  STATIC_CHECK(sycl::khr::is_property_list_for_v<e_t, host_tag_t>);
  STATIC_CHECK(sycl::khr::is_property_list_for_v<e_t, shared_tag_t>);

  STATIC_CHECK(std::is_same_v<decltype(sycl::malloc_device(
                                  sz{}, std::declval<q_t>(), e_t{})),
                              void*>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::aligned_alloc_device<int>(
                                  sz{}, sz{}, std::declval<d_t>(),
                                  std::declval<c_t>(), e_t{})),
                              int*>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::malloc_host<int>(
                                  sz{}, std::declval<c_t>(), e_t{})),
                              int*>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::aligned_alloc_host(
                                  sz{}, sz{}, std::declval<q_t>(), e_t{})),
                              void*>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::malloc_shared<int>(
                                  sz{}, std::declval<d_t>(),
                                  std::declval<c_t>(), e_t{})),
                              int*>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::aligned_alloc_shared(
                                  sz{}, sz{}, std::declval<q_t>(), e_t{})),
                              void*>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::malloc(sz{}, std::declval<q_t>(),
                                           sycl::usm::alloc::device, e_t{})),
                     void*>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::aligned_alloc<int>(
                         sz{}, sz{}, std::declval<d_t>(), std::declval<c_t>(),
                         sycl::usm::alloc::shared, e_t{})),
                     int*>);

  using host_alloc_t = sycl::usm_allocator<int, sycl::usm::alloc::host>;
  using shared_alloc_t = sycl::usm_allocator<int, sycl::usm::alloc::shared>;
  STATIC_CHECK(std::is_constructible_v<host_alloc_t, c_t, d_t, e_t>);
  STATIC_CHECK(std::is_constructible_v<shared_alloc_t, q_t, e_t>);
}

}  // namespace khr_properties::tests

#endif  // SYCL_CTS_ENABLE_KHR_SPLIT_HEADERS_TESTS
