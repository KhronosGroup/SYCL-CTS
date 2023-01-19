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
  for (int i = 0; i < N; ++i) res &= (a[i] == b[i]);
  return res;
}

template <typename T, int N, typename U>
bool equal_impl(const sycl::vec<T, N>& a, const U& b) {
  bool res = true;
  for (int i = 0; i < N; ++i) res &= (a[i] == b);
  return res;
}

template <typename T, typename U>
bool equal(const T& a, const U& b) {
  return equal_impl(a, b);
}

/*
 * FIXME: hipSYCL cannot construct vector from 1 value
 * As a result the helper below is needed
 */
template <typename T, typename U>
T init_helper(const U init) {
  T res;
  value_operations::assign(res, init);
  return res;
}

namespace util {
/**
 * Since \p sycl::range by standard provides no default constructor,
 * this function returns a \p sycl::range with 1 for each dimension. */
template <int Dimensions>
sycl::range<Dimensions> get_default_range();

template <>
inline sycl::range<1> get_default_range() {
  return sycl::range<1>{1};
}

template <>
inline sycl::range<2> get_default_range() {
  return sycl::range<2>{1, 1};
}

template <>
inline sycl::range<3> get_default_range() {
  return sycl::range<3>{1, 1, 1};
}

/**
 * @brief Provides range for maximal size work-group
 *        supported by device of a given queue.
 *        Multidimentional work-group range is made
 *        as hyper-cybic as possible
 * @tparam Dimensions Dimension to use for group instance
 */
template <int Dimensions>
sycl::range<Dimensions> work_group_range(sycl::queue queue) {
  // query device for work-group sizes
  size_t max_work_item_sizes[Dimensions];
  {
    // FIXME: hipSYCL and ComputeCPP do not implement
    //        sycl::info::device::max_work_item_sizes<3> property
#if defined(SYCL_CTS_COMPILING_WITH_HIPSYCL) || \
    defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP)
    sycl::id<3> sizes =
        queue.get_device().get_info<sycl::info::device::max_work_item_sizes>();
#else
    sycl::id<3> sizes =
        queue.get_device()
            .get_info<sycl::info::device::max_work_item_sizes<3>>();
#endif
    for (int i = 0; i < Dimensions; ++i) {
      max_work_item_sizes[i] = sizes.get(i);
    }
  }
  size_t max_work_group_size =
      queue.get_device().get_info<sycl::info::device::max_work_group_size>();

  // make work-group size as much square/cubic as possible
  size_t work_group_sizes[Dimensions] = {
      std::min(max_work_item_sizes[0], max_work_group_size)};
  if constexpr (Dimensions > 1) {
    size_t rest_work_group_size = max_work_group_size;
    for (int cur_D = Dimensions; cur_D > 1; --cur_D) {
      size_t pref_size = pow(rest_work_group_size, 1. / cur_D) + 1;
      pref_size = std::min(pref_size, max_work_item_sizes[cur_D - 1]);
      // in the worst case of prime rest_work_group_size pref_size comes to 1
      while (rest_work_group_size % pref_size != 0) {
        --pref_size;
      }
      work_group_sizes[cur_D - 1] = pref_size;
      rest_work_group_size /= pref_size;
    }
    work_group_sizes[0] =
        std::min(max_work_item_sizes[0], rest_work_group_size);
  }

  sycl::range<Dimensions> work_group_range = get_default_range<Dimensions>();
  for (int i = 0; i < Dimensions; ++i)
    work_group_range[i] = work_group_sizes[i];

  return work_group_range;
}

/**
 * @brief Provides group size pretty printing
 * @tparam D Dimension of group instance
 */
template <int D>
std::string work_group_print(const sycl::range<D>& work_group_range) {
  std::string res("{ " + std::to_string(work_group_range[0]));
  for (int i = 1; i < D; ++i) res += ", " + std::to_string(work_group_range[i]);
  res += " }";
  return res;
}

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
static constexpr uint64_t exact_max = std::numeric_limits<T>::max();

template <>
static constexpr uint64_t exact_max<sycl::half> = 1ull << 11;
template <>
static constexpr uint64_t exact_max<float> = 1ull << 24;
template <>
static constexpr uint64_t exact_max<double> = 1ull << 53;

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
