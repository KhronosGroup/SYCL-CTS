/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "util.h"
#include <catch2/catch_test_macros.hpp>
#include <sycl/khr/split_headers/functional.hpp>
#include <type_traits>
#include <utility>

// These tests verify that <sycl/khr/split_headers/functional.hpp> provides the
// function objects from the SYCL specification: plus, multiplies, bit_and,
// bit_or, bit_xor, logical_and, logical_or, minimum and maximum. Each is a
// class template with a default template argument of void, providing
// T operator()(const T&, const T&) const, and is additionally specialized for
// void as a transparent function object. We check that the int specialization
// is complete and callable returning T, and that the transparent (void) form is
// callable.

namespace khr_split_headers::tests {

// Check that FO<int> is complete, FO<int>{}(int, int) returns int, and the
// transparent FO<void> form is callable.
#define CHECK_FUNCTION_OBJECT(FO)                                        \
  TEST_CASE("the functional header defines the " #FO " function object", \
            "[khr_split_headers][functional]") {                         \
    STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::FO<int>>);        \
    using ret_t = decltype(std::declval<sycl::FO<int>>()(                \
        std::declval<int>(), std::declval<int>()));                      \
    STATIC_REQUIRE(std::is_same_v<ret_t, int>);                          \
    STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::FO<>>);           \
    STATIC_REQUIRE(std::is_invocable_v<sycl::FO<>, int, int>);           \
  }

TEST_CASE("the functional header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][functional]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

CHECK_FUNCTION_OBJECT(plus)
CHECK_FUNCTION_OBJECT(multiplies)
CHECK_FUNCTION_OBJECT(bit_and)
CHECK_FUNCTION_OBJECT(bit_or)
CHECK_FUNCTION_OBJECT(bit_xor)
CHECK_FUNCTION_OBJECT(minimum)
CHECK_FUNCTION_OBJECT(maximum)

#undef CHECK_FUNCTION_OBJECT

// logical_and and logical_or implement && and ||, which yield bool rather than
// the operand type, so they are checked separately.
#define CHECK_LOGICAL_FUNCTION_OBJECT(FO)                                \
  TEST_CASE("the functional header defines the " #FO " function object", \
            "[khr_split_headers][functional]") {                         \
    STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::FO<int>>);        \
    using ret_t = decltype(std::declval<sycl::FO<int>>()(                \
        std::declval<int>(), std::declval<int>()));                      \
    STATIC_REQUIRE(std::is_same_v<ret_t, bool>);                         \
    STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::FO<>>);           \
    STATIC_REQUIRE(std::is_invocable_v<sycl::FO<>, int, int>);           \
  }

CHECK_LOGICAL_FUNCTION_OBJECT(logical_and)
CHECK_LOGICAL_FUNCTION_OBJECT(logical_or)

#undef CHECK_LOGICAL_FUNCTION_OBJECT

TEST_CASE("the functional header default template argument is void",
          "[khr_split_headers][functional]") {
  STATIC_REQUIRE(std::is_same_v<sycl::plus<>, sycl::plus<void>>);
  STATIC_REQUIRE(std::is_same_v<sycl::maximum<>, sycl::maximum<void>>);
}

}  // namespace khr_split_headers::tests
