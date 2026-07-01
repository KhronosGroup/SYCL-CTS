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
#include <sycl/khr/split_headers/multi_ptr.hpp>
#include <type_traits>

// These tests verify that <sycl/khr/split_headers/multi_ptr.hpp> provides the
// multi_ptr class, the access::address_space and access::decorated
// enumerations, and the explicit pointer aliases from the SYCL specification.
// For each alias we check that it names the expected multi_ptr specialization,
// i.e. the correct address space and decoration. The default-decorated aliases
// (global_ptr, etc.) default to access::decorated::legacy; the raw_* aliases
// use access::decorated::no and the decorated_* aliases use
// access::decorated::yes.

namespace khr_split_headers::tests {

namespace as = sycl::access;
using space = as::address_space;
using deco = as::decorated;

template <space Space, deco Decorated>
using mp = sycl::multi_ptr<int, Space, Decorated>;

TEST_CASE("the multi_ptr header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][multi_ptr]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the multi_ptr header defines the address_space enumeration",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_enum_v<as::address_space>);
}

TEST_CASE("the multi_ptr header defines the decorated enumeration",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_enum_v<as::decorated>);
}

TEST_CASE("the multi_ptr header defines the multi_ptr class template",
          "[khr_split_headers][multi_ptr]") {
  using ptr_t = mp<space::global_space, deco::no>;
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<ptr_t>);
}

// NOTE: The spec lists global_ptr, raw_*, decorated_* and the other aliases as
// part of this header. The current DPC++ build only includes
// <sycl/multi_ptr.hpp> here, not <sycl/pointers.hpp> where the aliases are
// defined, so the tests below fail to compile until the implementation exposes
// the aliases through <sycl/khr/split_headers/multi_ptr.hpp>.
// --- Default-decorated aliases (decorated::legacy) ---
TEST_CASE("the multi_ptr header defines the global_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::global_ptr<int>,
                                mp<space::global_space, deco::legacy>>);
}

TEST_CASE("the multi_ptr header defines the local_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::local_ptr<int>,
                                mp<space::local_space, deco::legacy>>);
}

TEST_CASE("the multi_ptr header defines the constant_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::constant_ptr<int>,
                                mp<space::constant_space, deco::legacy>>);
}

TEST_CASE("the multi_ptr header defines the private_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::private_ptr<int>,
                                mp<space::private_space, deco::legacy>>);
}

TEST_CASE("the multi_ptr header defines the generic_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::generic_ptr<int>,
                                mp<space::generic_space, deco::legacy>>);
}

// --- raw_* aliases (decorated::no) ---
TEST_CASE("the multi_ptr header defines the raw_global_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::raw_global_ptr<int>,
                                mp<space::global_space, deco::no>>);
}

TEST_CASE("the multi_ptr header defines the raw_local_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::raw_local_ptr<int>,
                                mp<space::local_space, deco::no>>);
}

TEST_CASE("the multi_ptr header defines the raw_private_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::raw_private_ptr<int>,
                                mp<space::private_space, deco::no>>);
}

TEST_CASE("the multi_ptr header defines the raw_generic_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::raw_generic_ptr<int>,
                                mp<space::generic_space, deco::no>>);
}

// --- decorated_* aliases (decorated::yes) ---
TEST_CASE("the multi_ptr header defines the decorated_global_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::decorated_global_ptr<int>,
                                mp<space::global_space, deco::yes>>);
}

TEST_CASE("the multi_ptr header defines the decorated_local_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::decorated_local_ptr<int>,
                                mp<space::local_space, deco::yes>>);
}

TEST_CASE("the multi_ptr header defines the decorated_private_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::decorated_private_ptr<int>,
                                mp<space::private_space, deco::yes>>);
}

TEST_CASE("the multi_ptr header defines the decorated_generic_ptr alias",
          "[khr_split_headers][multi_ptr]") {
  STATIC_REQUIRE(std::is_same_v<sycl::decorated_generic_ptr<int>,
                                mp<space::generic_space, deco::yes>>);
}

}  // namespace khr_split_headers::tests
