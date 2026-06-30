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
// group_algorithms.hpp provides the algorithms; groups.hpp provides the group
// type used as their first argument, and functional.hpp provides the SYCL
// binary operators (e.g. sycl::plus) used by the reduce and scan algorithms.
#include <sycl/khr/split_headers/functional.hpp>
#include <sycl/khr/split_headers/group_algorithms.hpp>
#include <sycl/khr/split_headers/groups.hpp>
#include <type_traits>
#include <utility>

// These tests verify that <sycl/khr/split_headers/group_algorithms.hpp>
// provides the group algorithm functions from the SYCL specification:
// group_broadcast; the any/all/none_of_group and joint_* predicate functions;
// reduce_over_group and joint_reduce; the shift/permute/select group functions;
// and the exclusive/inclusive scan functions. The algorithms are device
// functions, but the checks below only query their return types in an
// unevaluated decltype context using a sycl::group as the group argument, so no
// kernel is launched. The predicate functions return bool; the value-producing
// functions return the work-item value type T.

namespace khr_split_headers::tests {

using group_t = sycl::group<1>;
// shift_group_left/right, permute_group_by_xor and select_from_group are only
// available for the sub_group.
using sub_group_t = sycl::sub_group;
using op_t = sycl::plus<int>;

// A simple unary predicate for the *_of_group / joint_*_of predicate forms.
struct int_predicate {
  bool operator()(int) const;
};

TEST_CASE("the group_algorithms header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][group_algorithms]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the group_algorithms header defines group_broadcast",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::group_broadcast(std::declval<group_t>(),
                                                  std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

// --- Predicate functions (return bool) ---
TEST_CASE("the group_algorithms header defines any_of_group",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::any_of_group(std::declval<group_t>(),
                                               std::declval<bool>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the group_algorithms header defines all_of_group",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::all_of_group(std::declval<group_t>(),
                                               std::declval<bool>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the group_algorithms header defines none_of_group",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::none_of_group(std::declval<group_t>(),
                                                std::declval<bool>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the group_algorithms header defines joint_any_of",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::joint_any_of(
      std::declval<group_t>(), std::declval<int*>(), std::declval<int*>(),
      std::declval<int_predicate>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the group_algorithms header defines joint_all_of",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::joint_all_of(
      std::declval<group_t>(), std::declval<int*>(), std::declval<int*>(),
      std::declval<int_predicate>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

TEST_CASE("the group_algorithms header defines joint_none_of",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::joint_none_of(
      std::declval<group_t>(), std::declval<int*>(), std::declval<int*>(),
      std::declval<int_predicate>()));
  STATIC_REQUIRE(std::is_same_v<return_t, bool>);
}

// --- Reductions (return T) ---
TEST_CASE("the group_algorithms header defines reduce_over_group",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::reduce_over_group(
      std::declval<group_t>(), std::declval<int>(), std::declval<op_t>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the group_algorithms header defines joint_reduce",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::joint_reduce(
      std::declval<group_t>(), std::declval<int*>(), std::declval<int*>(),
      std::declval<op_t>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

// --- Communication functions (return T) ---
TEST_CASE("the group_algorithms header defines shift_group_left",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::shift_group_left(std::declval<sub_group_t>(),
                                                   std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the group_algorithms header defines shift_group_right",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::shift_group_right(std::declval<sub_group_t>(),
                                                    std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the group_algorithms header defines permute_group_by_xor",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::permute_group_by_xor(
      std::declval<sub_group_t>(), std::declval<int>(),
      std::declval<sub_group_t::linear_id_type>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the group_algorithms header defines select_from_group",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::select_from_group(
      std::declval<sub_group_t>(), std::declval<int>(),
      std::declval<sub_group_t::id_type>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

// --- Scans (return T) ---
TEST_CASE("the group_algorithms header defines exclusive_scan_over_group",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::exclusive_scan_over_group(
      std::declval<group_t>(), std::declval<int>(), std::declval<op_t>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the group_algorithms header defines inclusive_scan_over_group",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::inclusive_scan_over_group(
      std::declval<group_t>(), std::declval<int>(), std::declval<op_t>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

TEST_CASE("the group_algorithms header defines joint_exclusive_scan",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::joint_exclusive_scan(
      std::declval<group_t>(), std::declval<int*>(), std::declval<int*>(),
      std::declval<int*>(), std::declval<op_t>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int*>);
}

TEST_CASE("the group_algorithms header defines joint_inclusive_scan",
          "[khr_split_headers][group_algorithms]") {
  using return_t = decltype(sycl::joint_inclusive_scan(
      std::declval<group_t>(), std::declval<int*>(), std::declval<int*>(),
      std::declval<int*>(), std::declval<op_t>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int*>);
}

}  // namespace khr_split_headers::tests
