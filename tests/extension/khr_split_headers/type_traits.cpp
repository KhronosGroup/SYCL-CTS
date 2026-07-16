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
// functional.hpp provides sycl::plus, used below as the binary operator for the
// known_identity / has_known_identity traits.
#include <sycl/khr/split_headers/functional.hpp>
#include <sycl/khr/split_headers/type_traits.hpp>
#include <type_traits>

// These tests verify that <sycl/khr/split_headers/type_traits.hpp> provides the
// type traits from the SYCL specification: is_property / is_property_of (and
// their _v forms), is_group / is_group_v, is_device_copyable / _v,
// any_device_has / all_devices_have (and their _v forms), remove_decoration /
// remove_decoration_t, and known_identity / has_known_identity (and their _v
// forms). We check that each trait template is present and produces a value of
// the documented type using simple standalone types.

namespace khr_split_headers::tests {

TEST_CASE("the type_traits header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][type_traits]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

// NOTE: The spec lists is_property, is_property_of, is_group and
// is_device_copyable (with their _v forms) as part of this header. The current
// DPC++ build only includes <sycl/device_aspect_traits.hpp> here, which does
// not provide these traits, so the tests below fail to compile until the
// implementation exposes them through
// <sycl/khr/split_headers/type_traits.hpp>.
TEST_CASE("the type_traits header defines the is_property trait",
          "[khr_split_headers][type_traits]") {
  // A type that is not a property yields a well-formed false value.
  STATIC_REQUIRE(
      std::is_same_v<decltype(sycl::is_property<int>::value), const bool>);
  STATIC_REQUIRE(sycl::is_property_v<int> == false);
}

TEST_CASE("the type_traits header defines the is_property_of trait",
          "[khr_split_headers][type_traits]") {
  STATIC_REQUIRE(std::is_same_v<decltype(sycl::is_property_of<int, int>::value),
                                const bool>);
  STATIC_REQUIRE(sycl::is_property_of_v<int, int> == false);
}

TEST_CASE("the type_traits header defines the is_group trait",
          "[khr_split_headers][type_traits]") {
  STATIC_REQUIRE(
      std::is_same_v<decltype(sycl::is_group<int>::value), const bool>);
  STATIC_REQUIRE(sycl::is_group_v<int> == false);
}

TEST_CASE("the type_traits header defines the is_device_copyable trait",
          "[khr_split_headers][type_traits]") {
  STATIC_REQUIRE(sycl::is_device_copyable_v<int> == true);
  STATIC_REQUIRE(std::is_same_v<decltype(sycl::is_device_copyable<int>::value),
                                const bool>);
}

TEST_CASE("the type_traits header defines the any_device_has trait",
          "[khr_split_headers][type_traits]") {
  STATIC_REQUIRE(
      std::is_same_v<decltype(sycl::any_device_has<sycl::aspect::cpu>::value),
                     const bool>);
  constexpr bool v = sycl::any_device_has_v<sycl::aspect::cpu>;
  (void)v;
  SUCCEED();
}

TEST_CASE("the type_traits header defines the all_devices_have trait",
          "[khr_split_headers][type_traits]") {
  STATIC_REQUIRE(
      std::is_same_v<decltype(sycl::all_devices_have<sycl::aspect::cpu>::value),
                     const bool>);
  constexpr bool v = sycl::all_devices_have_v<sycl::aspect::cpu>;
  (void)v;
  SUCCEED();
}

TEST_CASE("the type_traits header defines the remove_decoration trait",
          "[khr_split_headers][type_traits]") {
  // For a non-decorated pointer the trait yields the same pointer type.
  STATIC_REQUIRE(std::is_same_v<sycl::remove_decoration<int*>::type, int*>);
  STATIC_REQUIRE(std::is_same_v<sycl::remove_decoration_t<int*>, int*>);
}

TEST_CASE("the type_traits header defines the known_identity trait",
          "[khr_split_headers][type_traits]") {
  // plus has a known identity of 0 for arithmetic accumulator types.
  STATIC_REQUIRE(sycl::known_identity_v<sycl::plus<int>, int> == 0);
  STATIC_REQUIRE(std::is_same_v<
                 decltype(sycl::known_identity<sycl::plus<int>, int>::value),
                 const int>);
}

TEST_CASE("the type_traits header defines the has_known_identity trait",
          "[khr_split_headers][type_traits]") {
  STATIC_REQUIRE(sycl::has_known_identity_v<sycl::plus<int>, int> == true);
  STATIC_REQUIRE(
      std::is_same_v<
          decltype(sycl::has_known_identity<sycl::plus<int>, int>::value),
          const bool>);
}

}  // namespace khr_split_headers::tests
