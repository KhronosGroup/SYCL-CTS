/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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

#include "../common/common.h"
#include "../common/get_group_range.h"
#include "../common/once_per_unit.h"
#include "type_coverage.h"

#include <catch2/catch_template_test_macros.hpp>

/*
 * FIXME: hipSYCL does not implement size member function of sycl::vec
 * As a result the following implementation is not working
 * Workaround is presented below

template<typename T, typename U>
bool equal(const T& a, const U& b)
{
  return value_operations::are_equal(a, b);
}
*/
template <typename T, typename U>
bool equal_impl(const T& a, const U& b) {
  return a == b;
}

template <typename T, int N>
bool equal_impl(const sycl::vec<T, N>& a, const sycl::vec<T, N>& b) {
  bool res = true;
  for (int i = 0; i < N; ++i) {
    res &= (a[i] == b[i]);
  }
  return res;
}

template <typename T, int N, typename U>
bool equal_impl(const sycl::vec<T, N>& a, const U& b) {
  bool res = true;
  for (int i = 0; i < N; ++i) {
    res &= (a[i] == b);
  }
  return res;
}

// FIXME: hipSYCL has not implemented sycl::marray type yet
//        The warning is printed from group_shift.cpp and group_permute.cpp
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
bool equal_impl(const sycl::marray<T, N>& a, const sycl::marray<T, N>& b) {
  bool res = true;
  for (int i = 0; i < N; ++i) {
    res &= (a[i] == b[i]);
  }
  return res;
}

template <typename T, size_t N, typename U>
bool equal_impl(const sycl::marray<T, N>& a, const U& b) {
  bool res = true;
  for (size_t i = 0; i < N; ++i) {
    res &= (a[i] == b);
  }
  return res;
}
#endif

template <typename T, typename U>
bool equal(const T& a, const U& b) {
  return equal_impl(a, b);
}

/*
 * FIXME: hipSYCL cannot construct vector from 1 value
 * As a result the helper below is needed
 */
template <typename T, typename U>
T splat_init(const U init) {
  T res;
  value_operations::assign(res, init);
  return res;
}

namespace util {
/** @brief A user-defined class with several scalar member variables, default
 *         constructor and some overloaded operators.
 */
struct custom_type {
  int m_int_field{};
  bool m_bool_field{};

  custom_type() = default;

  custom_type(int value) {
    m_int_field = value;
    m_bool_field = value;
  }

  friend bool operator==(const custom_type& c_t_l, const custom_type& c_t_r) {
    return c_t_l.m_int_field == c_t_r.m_int_field &&
           c_t_l.m_bool_field == c_t_r.m_bool_field;
  }

  operator int() const { return m_int_field; }

  void operator=(int value) {
    m_int_field = value;
    m_bool_field = value;
  }
};

/**
 * @brief Provides limit for exact integer number representation by different
 * types
 *
 */
template <typename T>
inline constexpr uint64_t exact_max = std::numeric_limits<T>::max();

template <>
inline constexpr uint64_t exact_max<sycl::half> = 1ull << 11;
template <>
inline constexpr uint64_t exact_max<float> = 1ull << 24;
template <>
inline constexpr uint64_t exact_max<double> = 1ull << 53;

}  // namespace util

// Cartesian product of type lists
// adapted from Patrick Fromberg, https://stackoverflow.com/a/19611856
template <typename... T>
struct concatenation;

template <template <typename...> class R, typename... As, typename... Bs>
struct concatenation<R<As...>, R<Bs...>> {
  using type = R<As..., Bs...>;
};

template <template <typename...> class R, typename... As, typename... Bs>
struct concatenation<R<As...>, Bs...> {
  using type = R<As..., Bs...>;
};

template <typename... Ts>
struct product_helper;

template <template <typename...> class R, typename... Ts>
struct product_helper<R<Ts...>> {  // stop condition
  using type = R<Ts...>;
};

template <template <typename...> class R, typename... Ts>
struct product_helper<R<R<>>, Ts...> {  // catches first empty tuple
  using type = R<>;
};

template <template <typename...> class R, typename... Ts, typename... Rests>
struct product_helper<R<Ts...>, R<>,
                      Rests...> {  // catches any empty tuple except first
  using type = R<>;
};

template <template <typename...> class R, typename... X, typename H,
          typename... Rests>
struct product_helper<R<X...>, R<H>, Rests...> {
  using type1 = R<typename concatenation<X, R<H>>::type...>;
  using type = typename product_helper<type1, Rests...>::type;
};

template <template <typename...> class R, typename... X,
          template <typename...> class Head, typename T, typename... Ts,
          typename... Rests>
struct product_helper<R<X...>, Head<T, Ts...>, Rests...> {
  using type1 = R<typename concatenation<X, R<T>>::type...>;
  using type2 = typename product_helper<R<X...>, R<Ts...>>::type;
  using type3 = typename concatenation<type1, type2>::type;
  using type = typename product_helper<type3, Rests...>::type;
};

template <template <typename...> class R, typename... Ts>
struct product;

template <template <typename...> class R>
struct product<R> {  // no input, R specifies the return type
  using type = R<>;
};

template <template <typename...> class R, template <typename...> class Head,
          typename... Ts, typename... Tail>
struct product<R, Head<Ts...>, Tail...> {  // R is the return type, Head<A...>
                                           // is the first input list
  using type = typename product_helper<R<R<Ts>...>, Tail...>::type;
};

/// Type lists for tests
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE

using FundamentalTypes =
    std::tuple<size_t, float, char, signed char, unsigned char, short int,
               unsigned short int, int, unsigned int, long int,
               unsigned long int, long long int, unsigned long long int>;
using Types = unnamed_type_pack<size_t, float, char, signed char, unsigned char,
                                short int, unsigned short int, int,
                                unsigned int, long int, unsigned long int,
                                long long int, unsigned long long int>;

#else

using FundamentalTypes = std::tuple<float, char, int, unsigned long long int>;
using Types = unnamed_type_pack<float, char, int, unsigned long long int>;

#endif

// FIXME: hipSYCL has not implemented sycl::marray type yet
//        The warning is printed from group_shift.cpp and group_permute.cpp
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
using ExtendedTypes =
    concatenation<FundamentalTypes,
                  std::tuple<bool, sycl::vec<unsigned int, 4>,
                             sycl::vec<long long int, 2>>>::type;
#else
using ExtendedTypes = concatenation<
    FundamentalTypes,
    std::tuple<bool, sycl::vec<unsigned int, 4>, sycl::vec<long long int, 2>,
               sycl::marray<float, 5>, sycl::marray<short int, 7>>>::type;
#endif

using CustomTypes = concatenation<ExtendedTypes, util::custom_type>::type;

template <typename T>
inline auto get_op_types() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  static const auto types = []() {
    if constexpr (std::is_floating_point_v<T> ||
                  std::is_same_v<std::remove_cv_t<T>, sycl::half>) {
      // Bitwise operations are not defined for floating point types.
      return named_type_pack<sycl::plus<T>, sycl::multiplies<T>,
                             sycl::logical_and<T>, sycl::logical_or<T>,
                             sycl::minimum<T>,
                             sycl::maximum<T>>::generate("plus", "multiplies",
                                                         "logical_and",
                                                         "logical_or",
                                                         "minimum", "maximum");
    } else {
      return named_type_pack<
          sycl::plus<T>, sycl::multiplies<T>, sycl::bit_and<T>, sycl::bit_or<T>,
          sycl::bit_xor<T>, sycl::logical_and<T>, sycl::logical_or<T>,
          sycl::minimum<T>, sycl::maximum<T>>::generate("plus", "multiplies",
                                                        "bit_and", "bit_or",
                                                        "bit_xor",
                                                        "logical_and",
                                                        "logical_or", "minimum",
                                                        "maximum");
    }
  }();
#else
  static const auto types =
      named_type_pack<sycl::plus<T>, sycl::maximum<T>>::generate("plus",
                                                                 "maximum");
#endif
  return types;
}
