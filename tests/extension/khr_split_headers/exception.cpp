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
#include <sycl/khr/split_headers/exception.hpp>
#include <system_error>
#include <type_traits>
#include <utility>

// These tests verify that <sycl/khr/split_headers/exception.hpp> provides the
// error handling APIs from the SYCL specification: the async_handler alias, the
// exception and exception_list classes, the errc enumeration, the
// make_error_code and sycl_category free functions, and the
// std::is_error_code_enum specialization for sycl::errc.

namespace khr_split_headers::tests {

TEST_CASE("the exception header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][exception]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the exception header defines the exception class",
          "[khr_split_headers][exception]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::exception>);
}

TEST_CASE("the exception header defines the exception_list class",
          "[khr_split_headers][exception]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::exception_list>);
}

TEST_CASE("the exception header defines the errc enumeration",
          "[khr_split_headers][exception]") {
  STATIC_REQUIRE(std::is_enum_v<sycl::errc>);
}

TEST_CASE("the exception header defines the async_handler alias",
          "[khr_split_headers][exception]") {
  // async_handler is a callable accepting an exception_list.
  STATIC_REQUIRE(
      std::is_invocable_v<sycl::async_handler, sycl::exception_list>);
}

TEST_CASE("the exception header defines the make_error_code function",
          "[khr_split_headers][exception]") {
  using return_t = decltype(sycl::make_error_code(std::declval<sycl::errc>()));
  STATIC_REQUIRE(std::is_same_v<return_t, std::error_code>);
}

TEST_CASE("the exception header defines the sycl_category function",
          "[khr_split_headers][exception]") {
  using return_t = decltype(sycl::sycl_category());
  STATIC_REQUIRE(std::is_same_v<return_t, const std::error_category&>);
}

TEST_CASE(
    "the exception header specializes std::is_error_code_enum for sycl::errc",
    "[khr_split_headers][exception]") {
  STATIC_REQUIRE(std::is_error_code_enum<sycl::errc>::value);
}

}  // namespace khr_split_headers::tests
