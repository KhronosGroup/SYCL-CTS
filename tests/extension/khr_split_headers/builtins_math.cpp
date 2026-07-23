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
#include <cstdint>
#include <sycl/khr/split_headers/builtins_math.hpp>
#include <type_traits>
#include <utility>

// These tests verify that <sycl/khr/split_headers/builtins_math.hpp> provides
// the math, native-precision and half-precision functions from the SYCL
// specification. We use scalar float inputs so the header can be validated
// standalone, without pulling in vec/marray headers. For scalar floating point
// inputs each function returns the floating point argument type, except ilogb
// (returns int). Pointer-argument functions take raw pointers.

namespace khr_split_headers::tests {

TEST_CASE("the builtins_math header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][builtins_math]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

// Helper: assert f(float...) returns float.
#define CHECK_UNARY_FLOAT(FN)                                        \
  TEST_CASE("the builtins_math header defines the " #FN " function", \
            "[khr_split_headers][builtins_math]") {                  \
    using return_t = decltype(sycl::FN(std::declval<float>()));      \
    STATIC_REQUIRE(std::is_same_v<return_t, float>);                 \
  }

#define CHECK_BINARY_FLOAT(FN)                                            \
  TEST_CASE("the builtins_math header defines the " #FN " function",      \
            "[khr_split_headers][builtins_math]") {                       \
    using return_t =                                                      \
        decltype(sycl::FN(std::declval<float>(), std::declval<float>())); \
    STATIC_REQUIRE(std::is_same_v<return_t, float>);                      \
  }

// --- Unary float -> float math functions ---
CHECK_UNARY_FLOAT(acos)
CHECK_UNARY_FLOAT(acosh)
CHECK_UNARY_FLOAT(acospi)
CHECK_UNARY_FLOAT(asin)
CHECK_UNARY_FLOAT(asinh)
CHECK_UNARY_FLOAT(asinpi)
CHECK_UNARY_FLOAT(atan)
CHECK_UNARY_FLOAT(atanh)
CHECK_UNARY_FLOAT(atanpi)
CHECK_UNARY_FLOAT(cbrt)
CHECK_UNARY_FLOAT(ceil)
CHECK_UNARY_FLOAT(cos)
CHECK_UNARY_FLOAT(cosh)
CHECK_UNARY_FLOAT(cospi)
CHECK_UNARY_FLOAT(erfc)
CHECK_UNARY_FLOAT(erf)
CHECK_UNARY_FLOAT(exp)
CHECK_UNARY_FLOAT(exp2)
CHECK_UNARY_FLOAT(exp10)
CHECK_UNARY_FLOAT(expm1)
CHECK_UNARY_FLOAT(fabs)
CHECK_UNARY_FLOAT(floor)
CHECK_UNARY_FLOAT(lgamma)
CHECK_UNARY_FLOAT(log)
CHECK_UNARY_FLOAT(log2)
CHECK_UNARY_FLOAT(log10)
CHECK_UNARY_FLOAT(log1p)
CHECK_UNARY_FLOAT(logb)
CHECK_UNARY_FLOAT(rint)
CHECK_UNARY_FLOAT(round)
CHECK_UNARY_FLOAT(rsqrt)
CHECK_UNARY_FLOAT(sin)
CHECK_UNARY_FLOAT(sinh)
CHECK_UNARY_FLOAT(sinpi)
CHECK_UNARY_FLOAT(sqrt)
CHECK_UNARY_FLOAT(tan)
CHECK_UNARY_FLOAT(tanh)
CHECK_UNARY_FLOAT(tanpi)
CHECK_UNARY_FLOAT(tgamma)
CHECK_UNARY_FLOAT(trunc)

// --- Binary (float, float) -> float math functions ---
CHECK_BINARY_FLOAT(atan2)
CHECK_BINARY_FLOAT(atan2pi)
CHECK_BINARY_FLOAT(copysign)
CHECK_BINARY_FLOAT(fdim)
CHECK_BINARY_FLOAT(fmax)
CHECK_BINARY_FLOAT(fmin)
CHECK_BINARY_FLOAT(fmod)
CHECK_BINARY_FLOAT(hypot)
CHECK_BINARY_FLOAT(maxmag)
CHECK_BINARY_FLOAT(minmag)
CHECK_BINARY_FLOAT(nextafter)
CHECK_BINARY_FLOAT(pow)
CHECK_BINARY_FLOAT(powr)
CHECK_BINARY_FLOAT(remainder)

#undef CHECK_UNARY_FLOAT
#undef CHECK_BINARY_FLOAT

// --- Ternary (float, float, float) -> float ---
TEST_CASE("the builtins_math header defines the fma function",
          "[khr_split_headers][builtins_math]") {
  using return_t = decltype(sycl::fma(
      std::declval<float>(), std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the mad function",
          "[khr_split_headers][builtins_math]") {
  using return_t = decltype(sycl::mad(
      std::declval<float>(), std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

// --- Functions with an integer argument ---
TEST_CASE("the builtins_math header defines the ldexp function",
          "[khr_split_headers][builtins_math]") {
  using return_t =
      decltype(sycl::ldexp(std::declval<float>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the pown function",
          "[khr_split_headers][builtins_math]") {
  using return_t =
      decltype(sycl::pown(std::declval<float>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the rootn function",
          "[khr_split_headers][builtins_math]") {
  using return_t =
      decltype(sycl::rootn(std::declval<float>(), std::declval<int>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

// --- Functions returning an integer ---
TEST_CASE("the builtins_math header defines the ilogb function",
          "[khr_split_headers][builtins_math]") {
  using return_t = decltype(sycl::ilogb(std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, int>);
}

// --- nan: unsigned bit-pattern argument ---
TEST_CASE("the builtins_math header defines the nan function",
          "[khr_split_headers][builtins_math]") {
  using return_t = decltype(sycl::nan(std::declval<std::uint32_t>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

// --- Functions taking a pointer output argument ---
TEST_CASE("the builtins_math header defines the frexp function",
          "[khr_split_headers][builtins_math]") {
  using return_t =
      decltype(sycl::frexp(std::declval<float>(), std::declval<int*>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the modf function",
          "[khr_split_headers][builtins_math]") {
  using return_t =
      decltype(sycl::modf(std::declval<float>(), std::declval<float*>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the sincos function",
          "[khr_split_headers][builtins_math]") {
  using return_t =
      decltype(sycl::sincos(std::declval<float>(), std::declval<float*>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the fract function",
          "[khr_split_headers][builtins_math]") {
  using return_t =
      decltype(sycl::fract(std::declval<float>(), std::declval<float*>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the lgamma_r function",
          "[khr_split_headers][builtins_math]") {
  using return_t =
      decltype(sycl::lgamma_r(std::declval<float>(), std::declval<int*>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the remquo function",
          "[khr_split_headers][builtins_math]") {
  using return_t = decltype(sycl::remquo(
      std::declval<float>(), std::declval<float>(), std::declval<int*>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

// --- double scalar overloads (representative sample) ---
TEST_CASE("the builtins_math header provides functions for double scalars",
          "[khr_split_headers][builtins_math]") {
  using sqrt_t = decltype(sycl::sqrt(std::declval<double>()));
  using pow_t =
      decltype(sycl::pow(std::declval<double>(), std::declval<double>()));
  using ilogb_t = decltype(sycl::ilogb(std::declval<double>()));
  STATIC_REQUIRE(std::is_same_v<sqrt_t, double>);
  STATIC_REQUIRE(std::is_same_v<pow_t, double>);
  STATIC_REQUIRE(std::is_same_v<ilogb_t, int>);
}

// --- native:: variants ---
#define CHECK_NATIVE_UNARY(FN)                                               \
  TEST_CASE("the builtins_math header defines the native::" #FN " function", \
            "[khr_split_headers][builtins_math]") {                          \
    using return_t = decltype(sycl::native::FN(std::declval<float>()));      \
    STATIC_REQUIRE(std::is_same_v<return_t, float>);                         \
  }

CHECK_NATIVE_UNARY(cos)
CHECK_NATIVE_UNARY(exp)
CHECK_NATIVE_UNARY(exp2)
CHECK_NATIVE_UNARY(exp10)
CHECK_NATIVE_UNARY(log)
CHECK_NATIVE_UNARY(log2)
CHECK_NATIVE_UNARY(log10)
CHECK_NATIVE_UNARY(recip)
CHECK_NATIVE_UNARY(rsqrt)
CHECK_NATIVE_UNARY(sin)
CHECK_NATIVE_UNARY(sqrt)
CHECK_NATIVE_UNARY(tan)

#undef CHECK_NATIVE_UNARY

TEST_CASE("the builtins_math header defines the native::divide function",
          "[khr_split_headers][builtins_math]") {
  using return_t = decltype(sycl::native::divide(std::declval<float>(),
                                                 std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the native::powr function",
          "[khr_split_headers][builtins_math]") {
  using return_t = decltype(sycl::native::powr(std::declval<float>(),
                                               std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

// --- half_precision:: variants ---
#define CHECK_HALF_UNARY(FN)                                            \
  TEST_CASE("the builtins_math header defines the half_precision::" #FN \
            " function",                                                \
            "[khr_split_headers][builtins_math]") {                     \
    using return_t =                                                    \
        decltype(sycl::half_precision::FN(std::declval<float>()));      \
    STATIC_REQUIRE(std::is_same_v<return_t, float>);                    \
  }

CHECK_HALF_UNARY(cos)
CHECK_HALF_UNARY(exp)
CHECK_HALF_UNARY(exp2)
CHECK_HALF_UNARY(exp10)
CHECK_HALF_UNARY(log)
CHECK_HALF_UNARY(log2)
CHECK_HALF_UNARY(log10)
CHECK_HALF_UNARY(recip)
CHECK_HALF_UNARY(rsqrt)
CHECK_HALF_UNARY(sin)
CHECK_HALF_UNARY(sqrt)
CHECK_HALF_UNARY(tan)

#undef CHECK_HALF_UNARY

TEST_CASE(
    "the builtins_math header defines the half_precision::divide function",
    "[khr_split_headers][builtins_math]") {
  using return_t = decltype(sycl::half_precision::divide(
      std::declval<float>(), std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

TEST_CASE("the builtins_math header defines the half_precision::powr function",
          "[khr_split_headers][builtins_math]") {
  using return_t = decltype(sycl::half_precision::powr(std::declval<float>(),
                                                       std::declval<float>()));
  STATIC_REQUIRE(std::is_same_v<return_t, float>);
}

}  // namespace khr_split_headers::tests
