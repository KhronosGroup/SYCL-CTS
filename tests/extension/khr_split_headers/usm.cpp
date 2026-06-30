/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2026 The Khronos Group Inc.
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
*******************************************************************************/

#include "util.h"
#include <catch2/catch_test_macros.hpp>
#include <sycl/khr/split_headers/usm.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>

// These tests verify that <sycl/khr/split_headers/usm.hpp> provides the USM
// definitions from the SYCL specification: the usm::alloc enumeration, the
// usm_allocator class, the allocation free functions (malloc_device,
// malloc_host, malloc_shared and their aligned_alloc and untyped variants) and
// the memory pointer query functions (get_pointer_type, get_pointer_device).
// We check that each symbol is present and has the documented return type,
// using the queue-based overloads so the checks remain compile-time only.

namespace khr_split_headers::tests {

TEST_CASE("the usm header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][usm]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the usm header defines the usm::alloc enumeration",
          "[khr_split_headers][usm]") {
  STATIC_REQUIRE(std::is_enum_v<sycl::usm::alloc>);
}

TEST_CASE("the usm header defines the usm_allocator class template",
          "[khr_split_headers][usm]") {
  using allocator_t = sycl::usm_allocator<int, sycl::usm::alloc::host>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<allocator_t>);
}

// Untyped allocation functions return void*.
TEST_CASE("the usm header defines the untyped malloc_device function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::malloc_device(std::declval<std::size_t>(),
                                                std::declval<sycl::queue>()));
  STATIC_REQUIRE(std::is_same_v<return_t, void*>);
}

TEST_CASE("the usm header defines the untyped malloc_host function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::malloc_host(std::declval<std::size_t>(),
                                              std::declval<sycl::queue>()));
  STATIC_REQUIRE(std::is_same_v<return_t, void*>);
}

TEST_CASE("the usm header defines the untyped malloc_shared function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::malloc_shared(std::declval<std::size_t>(),
                                                std::declval<sycl::queue>()));
  STATIC_REQUIRE(std::is_same_v<return_t, void*>);
}

TEST_CASE("the usm header defines the untyped aligned_alloc_device function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::aligned_alloc_device(
      std::declval<std::size_t>(), std::declval<std::size_t>(),
      std::declval<sycl::queue>()));
  STATIC_REQUIRE(std::is_same_v<return_t, void*>);
}

TEST_CASE("the usm header defines the untyped aligned_alloc_host function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::aligned_alloc_host(
      std::declval<std::size_t>(), std::declval<std::size_t>(),
      std::declval<sycl::queue>()));
  STATIC_REQUIRE(std::is_same_v<return_t, void*>);
}

TEST_CASE("the usm header defines the untyped aligned_alloc_shared function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::aligned_alloc_shared(
      std::declval<std::size_t>(), std::declval<std::size_t>(),
      std::declval<sycl::queue>()));
  STATIC_REQUIRE(std::is_same_v<return_t, void*>);
}

TEST_CASE("the usm header defines the untyped malloc function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::malloc(std::declval<std::size_t>(),
                                         std::declval<sycl::queue>(),
                                         std::declval<sycl::usm::alloc>()));
  STATIC_REQUIRE(std::is_same_v<return_t, void*>);
}

TEST_CASE("the usm header defines the untyped aligned_alloc function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::aligned_alloc(
      std::declval<std::size_t>(), std::declval<std::size_t>(),
      std::declval<sycl::queue>(), std::declval<sycl::usm::alloc>()));
  STATIC_REQUIRE(std::is_same_v<return_t, void*>);
}

// Typed allocation functions return T*.
TEST_CASE("the usm header defines the typed malloc_device function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::malloc_device<int>(
      std::declval<std::size_t>(), std::declval<sycl::queue>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int*>);
}

TEST_CASE("the usm header defines the typed malloc_shared function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::malloc_shared<int>(
      std::declval<std::size_t>(), std::declval<sycl::queue>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int*>);
}

TEST_CASE("the usm header defines the free function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::free(std::declval<void*>(),
                                       std::declval<sycl::queue>()));
  STATIC_REQUIRE(std::is_void_v<return_t>);
}

TEST_CASE("the usm header defines the get_pointer_type query function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::get_pointer_type(
      std::declval<const void*>(), std::declval<sycl::context>()));
  STATIC_REQUIRE(std::is_same_v<return_t, sycl::usm::alloc>);
}

TEST_CASE("the usm header defines the get_pointer_device query function",
          "[khr_split_headers][usm]") {
  using return_t = decltype(sycl::get_pointer_device(
      std::declval<const void*>(), std::declval<sycl::context>()));
  STATIC_REQUIRE(std::is_same_v<return_t, sycl::device>);
}

}  // namespace khr_split_headers::tests
