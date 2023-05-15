/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef __SYCLCTS_UTIL_MATH_REFERENCE_H
#define __SYCLCTS_UTIL_MATH_REFERENCE_H

#include <sycl/sycl.hpp>

#include "./../oclmath/reference_math.h"
#include "./math_helper.h"
#include <cmath>

// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL

#define MAKE_VEC_AND_MARRAY_VERSIONS(func)              \
  template <typename T, int N>                          \
  sycl::vec<T, N> func(sycl::vec<T, N> a) {             \
    return sycl_cts::math::run_func_on_vector<T, T, N>( \
        [](T x) { return func(x); }, a);                \
  }                                                     \
  template <typename T, size_t N>                       \
  sycl::marray<T, N> func(sycl::marray<T, N> a) {       \
    return sycl_cts::math::run_func_on_marray<T, T, N>( \
        [](T x) { return func(x); }, a);                \
  }

#define MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(func)                        \
  template <typename T, int N>                                          \
  sycl::vec<T, N> func(sycl::vec<T, N> a, sycl::vec<T, N> b) {          \
    return sycl_cts::math::run_func_on_vector<T, T, N>(                 \
        [](T x, T y) { return func(x, y); }, a, b);                     \
  }                                                                     \
  template <typename T, size_t N>                                       \
  sycl::marray<T, N> func(sycl::marray<T, N> a, sycl::marray<T, N> b) { \
    return sycl_cts::math::run_func_on_marray<T, T, N>(                 \
        [](T x, T y) { return func(x, y); }, a, b);                     \
  }

#define MAKE_VEC_AND_MARRAY_VERSIONS_3ARGS(func)                      \
  template <typename T, int N>                                        \
  sycl::vec<T, N> func(sycl::vec<T, N> a, sycl::vec<T, N> b,          \
                       sycl::vec<T, N> c) {                           \
    return sycl_cts::math::run_func_on_vector<T, T, N>(               \
        [](T x, T y, T z) { return func(x, y, z); }, a, b, c);        \
  }                                                                   \
  template <typename T, size_t N>                                     \
  sycl::marray<T, N> func(sycl::marray<T, N> a, sycl::marray<T, N> b, \
                          sycl::marray<T, N> c) {                     \
    return sycl_cts::math::run_func_on_marray<T, T, N>(               \
        [](T x, T y, T z) { return func(x, y, z); }, a, b, c);        \
  }

#define MAKE_VEC_AND_MARRAY_VERSIONS_WITH_SCALAR(func)  \
  template <typename T, int N>                          \
  sycl::vec<T, N> func(sycl::vec<T, N> a, T b) {        \
    return sycl_cts::math::run_func_on_vector<T, T, N>( \
        [](T x, T y) { return func(x, y); }, a, b);     \
  }                                                     \
  template <typename T, size_t N>                       \
  sycl::marray<T, N> func(sycl::marray<T, N> a, T b) {  \
    return sycl_cts::math::run_func_on_marray<T, T, N>( \
        [](T x, T y) { return func(x, y); }, a, b);     \
  }

#else  // definitions without marray for hipSYCL

#define MAKE_VEC_AND_MARRAY_VERSIONS(func)              \
  template <typename T, int N>                          \
  sycl::vec<T, N> func(sycl::vec<T, N> a) {             \
    return sycl_cts::math::run_func_on_vector<T, T, N>( \
        [](T x) { return func(x); }, a);                \
  }

#define MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(func)               \
  template <typename T, int N>                                 \
  sycl::vec<T, N> func(sycl::vec<T, N> a, sycl::vec<T, N> b) { \
    return sycl_cts::math::run_func_on_vector<T, T, N>(        \
        [](T x, T y) { return func(x, y); }, a, b);            \
  }

#define MAKE_VEC_AND_MARRAY_VERSIONS_3ARGS(func)               \
  template <typename T, int N>                                 \
  sycl::vec<T, N> func(sycl::vec<T, N> a, sycl::vec<T, N> b,   \
                       sycl::vec<T, N> c) {                    \
    return sycl_cts::math::run_func_on_vector<T, T, N>(        \
        [](T x, T y, T z) { return func(x, y, z); }, a, b, c); \
  }

#define MAKE_VEC_AND_MARRAY_VERSIONS_WITH_SCALAR(func)  \
  template <typename T, int N>                          \
  sycl::vec<T, N> func(sycl::vec<T, N> a, T b) {        \
    return sycl_cts::math::run_func_on_vector<T, T, N>( \
        [](T x, T y) { return func(x, y); }, a, b);     \
  }

#endif

namespace reference {
/* two argument relational reference */
template <typename T>
auto isequal(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher<std::equal_to>(a, b);
}

template <typename T>
auto isnotequal(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher<std::not_equal_to>(a, b);
}

template <typename T>
auto isgreater(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher<std::greater>(a, b);
}

template <typename T>
auto isgreaterequal(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher<std::greater_equal>(a, b);
}

template <typename T>
auto isless(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher<std::less>(a, b);
}

template <typename T>
auto islessequal(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher<std::less_equal>(a, b);
}

auto constexpr islessgreater_func = [](const auto &x, const auto &y) {
  return (x < y) || (x > y);
};
template <typename T>
auto islessgreater(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher(islessgreater_func, a, b);
}

auto constexpr isordered_func = [](const auto &x, const auto &y) {
  return (x == x) && (y == y);
};
template <typename T>
auto isordered(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher(isordered_func, a, b);
}

auto constexpr isunordered_func = [](const auto &x, const auto &y) {
  return !((x == x) && (y == y));
};
template <typename T>
auto isunordered(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher(isunordered_func, a, b);
}

/* one argument relational reference */
auto constexpr isfinite_func = [](const auto &x) { return std::isfinite(x); };
template <typename T>
auto isfinite(T a) {
  return sycl_cts::math::rel_func_dispatcher(isfinite_func, a);
}

auto constexpr isinf_func = [](const auto &x) { return std::isinf(x); };
template <typename T>
auto isinf(T a) {
  return sycl_cts::math::rel_func_dispatcher(isinf_func, a);
}

auto constexpr isnan_func = [](const auto &x) { return std::isnan(x); };
template <typename T>
auto isnan(T a) {
  return sycl_cts::math::rel_func_dispatcher(isnan_func, a);
}

auto constexpr isnormal_func = [](const auto &x) { return std::isnormal(x); };
template <typename T>
auto isnormal(T a) {
  return sycl_cts::math::rel_func_dispatcher(isnormal_func, a);
}

auto constexpr signbit_func = [](const auto &x) { return std::signbit(x); };
template <typename T>
auto signbit(T a) {
  return sycl_cts::math::rel_func_dispatcher(signbit_func, a);
}

template <typename T>
bool any(T x) {
  return sycl_cts::math::if_msb_set(x);
}
template <typename T, int N>
int any(sycl::vec<T, N> a) {
  for (int i = 0; i < N; i++) {
    if (any(getElement(a, i)) == 1) return true;
  }
  return false;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
bool any(sycl::marray<T, N> a) {
  for (size_t i = 0; i < N; i++) {
    if (any(a[i]) == 1) return true;
  }
  return false;
}
#endif

template <typename T>
bool all(T x) {
  return sycl_cts::math::if_msb_set(x);
}
template <typename T, int N>
int all(sycl::vec<T, N> a) {
  for (int i = 0; i < N; i++) {
    if (all(getElement(a, i)) == 0) return false;
  }
  return true;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
bool all(sycl::marray<T, N> a) {
  for (size_t i = 0; i < N; i++) {
    if (all(a[i]) == 0) return false;
  }
  return true;
}
#endif

template <typename T>
T bitselect(T a, T b, T c) {
  return (c & b) | (~c & a);
}
sycl::half bitselect(sycl::half a, sycl::half b, sycl::half c);
float bitselect(float a, float b, float c);
double bitselect(double a, double b, double c);
MAKE_VEC_AND_MARRAY_VERSIONS_3ARGS(bitselect)

template <typename T>
T select(T a, T b, bool c) {
  return c ? b : a;
}
template <typename T, typename K, int N>
sycl::vec<T, N> select(sycl::vec<T, N> a, sycl::vec<T, N> b,
                       sycl::vec<K, N> c) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    if (any(getElement<K, N>(c, i)) == 1)
      setElement<T, N>(res, i, getElement(b, i));
    else
      setElement<T, N>(res, i, getElement(a, i));
  }
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> select(sycl::marray<T, N> a, sycl::marray<T, N> b,
                          sycl::marray<bool, N> c) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = c[i] ? b[i] : a[i];
  }
  return res;
}
#endif

/* absolute value */
template <typename T>
T abs(T x) {
  return x < 0 ? -x : x;
}
MAKE_VEC_AND_MARRAY_VERSIONS(abs)

/* absolute difference */
template <typename T>
auto abs_diff(T a, T b) {
  using R = typename std::make_unsigned<T>::type;
  R h = (a > b) ? a : b;
  R l = (a <= b) ? a : b;
  return h - l;
}
template <typename T, int N, typename R = typename std::make_unsigned<T>::type>
sycl::vec<R, N> abs_diff(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<R, T, N>(
      [](T x, T y) { return abs_diff(x, y); }, a, b);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N,
          typename R = typename std::make_unsigned<T>::type>
sycl::marray<R, N> abs_diff(sycl::marray<T, N> a, sycl::marray<T, N> b) {
  return sycl_cts::math::run_func_on_marray<R, T, N>(
      [](T x, T y) { return abs_diff(x, y); }, a, b);
}
#endif

/* add with saturation */
template <typename T>
T add_sat(T a, T b) {
  if (std::is_unsigned<T>::value) {
    T res = a + b;
    if (res < a) res = -1;
    return res;
  } else {
    typedef typename std::make_unsigned<T>::type U;
    T r = T(U(a) + U(b));
    if (b > 0) {
      if (r < a) return std::numeric_limits<T>::max();
    } else {
      if (r > a) return std::numeric_limits<T>::min();
    }
    return r;
  }
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(add_sat)

/* half add */
template <typename T>
T hadd(T a, T b) {
  if (std::is_unsigned<T>::value) return (a >> 1) + (b >> 1) + ((a & b) & 0x1);
  return (a >> 1) + (b >> 1) + (a & b & 1);
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(hadd)

/* round up half add */
template <typename T>
T rhadd(T a, T b) {
  return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(rhadd)

/* clamp */
template <typename T>
sycl_cts::resultRef<T> clamp(T v, T minv, T maxv) {
  if (minv > maxv) return sycl_cts::resultRef<T>(T(), true);
  return (v < minv) ? minv : ((v > maxv) ? maxv : v);
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> clamp(sycl::vec<T, N> a, sycl::vec<T, N> b,
                                           sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return clamp(x, y, z); }, a, b, c);
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> clamp(sycl::vec<T, N> a, T b, T c) {
  sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    sycl_cts::resultRef<T> element = clamp(getElement(a, i), b, c);
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::vec<T, N>>(res, undefined);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> clamp(sycl::marray<T, N> a,
                                              sycl::marray<T, N> b,
                                              sycl::marray<T, N> c) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y, T z) { return clamp(x, y, z); }, a, b, c);
}
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> clamp(sycl::marray<T, N> a, T b, T c) {
  sycl::marray<T, N> res;
  std::map<int, bool> undefined;
  for (size_t i = 0; i < N; i++) {
    sycl_cts::resultRef<T> element = clamp(a[i], b, c);
    if (element.undefined.empty())
      res[i] = element.res;
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::marray<T, N>>(res, undefined);
}
#endif

/* count leading zeros */
template <typename T>
T clz(T x) {
  int lz = 0;
  for (int i = 0; i < sycl_cts::math::num_bits(x); i++)
    if (x & (1ull << i))
      lz = 0;
    else
      lz++;
  return static_cast<T>(lz);
}
MAKE_VEC_AND_MARRAY_VERSIONS(clz)

/* count trailing zeros */
template <typename T>
T ctz(T x) {
  const int bit_size = sycl_cts::math::num_bits(x);

  int tz = 0;
  for (int i = 0; i < bit_size; i++)
    if (x & (1ull << (bit_size - i - 1)))
      tz = 0;
    else
      tz++;
  return static_cast<T>(tz);
}
MAKE_VEC_AND_MARRAY_VERSIONS(ctz)

// mad_hi is after mul_hi

/* multiply add saturate */
unsigned char mad_sat(unsigned char, unsigned char, unsigned char);
unsigned short mad_sat(unsigned short, unsigned short, unsigned short);
unsigned int mad_sat(unsigned int, unsigned int, unsigned int);
unsigned long mad_sat(unsigned long, unsigned long, unsigned long);
unsigned long long mad_sat(unsigned long long, unsigned long long,
                           unsigned long long);
char mad_sat(char, char, char);
signed char mad_sat(signed char, signed char, signed char);
short mad_sat(short, short, short);
int mad_sat(int, int, int);
long mad_sat(long, long, long);
long long mad_sat(long long, long long, long long);

MAKE_VEC_AND_MARRAY_VERSIONS_3ARGS(mad_sat)

/* maximum value */
template <typename T>
sycl_cts::resultRef<T> max(T a, T b) {
  if constexpr (std::is_integral_v<T>)
    return (a < b) ? b : a;
  else if (std::isfinite(a) && std::isfinite(b))
    return (a < b) ? b : a;
  return sycl_cts::resultRef<T>(T(), true);
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> max(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return max(x, y); }, a, b);
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> max(sycl::vec<T, N> a, T b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return max(x, y); }, a, b);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> max(sycl::marray<T, N> a,
                                            sycl::marray<T, N> b) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y) { return max(x, y); }, a, b);
}
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> max(sycl::marray<T, N> a, T b) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y) { return max(x, y); }, a, b);
}
#endif

/* minimum value */
template <typename T>
sycl_cts::resultRef<T> min(T a, T b) {
  if constexpr (std::is_integral_v<T>)
    return (b < a) ? b : a;
  else if (std::isfinite(a) && std::isfinite(b))
    return (b < a) ? b : a;
  return sycl_cts::resultRef<T>(T(), true);
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> min(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return min(x, y); }, a, b);
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> min(sycl::vec<T, N> a, T b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return min(x, y); }, a, b);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> min(sycl::marray<T, N> a,
                                            sycl::marray<T, N> b) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y) { return min(x, y); }, a, b);
}
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> min(sycl::marray<T, N> a, T b) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y) { return min(x, y); }, a, b);
}
#endif

/* multiply and return high part */
unsigned char mul_hi(unsigned char, unsigned char);
unsigned short mul_hi(unsigned short, unsigned short);
unsigned int mul_hi(unsigned int, unsigned int);
unsigned long mul_hi(unsigned long, unsigned long);
unsigned long long mul_hi(unsigned long long, unsigned long long);
char mul_hi(char, char);
signed char mul_hi(signed char, signed char);
short mul_hi(short, short);
int mul_hi(int, int);
long mul_hi(long, long);
long long mul_hi(long long, long long);
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(mul_hi)

/* multiply add, get high part */
template <typename T>
T mad_hi(T x, T y, T z) {
  return mul_hi(x, y) + z;
}
MAKE_VEC_AND_MARRAY_VERSIONS_3ARGS(mad_hi)

/* bitwise rotate */
template <typename T>
T rotate(T v, T i) {
  if (std::is_unsigned<T>::value) {
    i = i % sycl_cts::math::num_bits(v);
    if (i == 0) return v;
    size_t nBits = sycl_cts::math::num_bits(v) - size_t(i);
    return T((v << i) | ((v >> nBits)));
  }
  typedef typename std::make_unsigned<T>::type R;
  R i_mod = R(i) % sycl_cts::math::num_bits(v);
  if (i_mod == 0) return v;
  T mask = T((T(1) << i_mod) - T(1));
  size_t nBits = sycl_cts::math::num_bits(v) - size_t(i_mod);
  return T((v << i_mod) | ((v >> nBits) & mask));
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(rotate)

/* substract with saturation */
template <typename T>
T sub_sat(T x, T y) {
  if (std::is_unsigned<T>::value) return x <= y ? 0 : x - y;

  const T max_val = std::numeric_limits<T>::max();
  const T min_val = std::numeric_limits<T>::min();
  if (x > 0) {
    if (y > 0) {
      return x - y;
    } else  // x > 0, y <= 0
    {
      return (x - max_val) > y ? max_val : x - y;
    }
  } else  // x <= 0
  {
    if (y > 0) {
      return (x - min_val) < y ? min_val : x - y;
    } else  // x <= 0, y <= 0
    {
      return x - y;
    }
  }
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(sub_sat)

/* upsample */
uint16_t upsample(uint8_t h, uint8_t l);
uint32_t upsample(uint16_t h, uint16_t l);
uint64_t upsample(uint32_t h, uint32_t l);
int16_t upsample(int8_t h, uint8_t l);
int32_t upsample(int16_t h, uint16_t l);
int64_t upsample(int32_t h, uint32_t l);

template <typename T>
struct upsample_t;

template <>
struct upsample_t<uint8_t> {
  using type = uint16_t;
};

template <>
struct upsample_t<uint16_t> {
  using type = uint32_t;
};

template <>
struct upsample_t<uint32_t> {
  using type = uint64_t;
};

template <>
struct upsample_t<int8_t> {
  using type = int16_t;
};

template <>
struct upsample_t<int16_t> {
  using type = int32_t;
};

template <>
struct upsample_t<int32_t> {
  using type = int64_t;
};

template <typename T, int N>
sycl::vec<typename upsample_t<T>::type, N> upsample(
    sycl::vec<T, N> a, sycl::vec<typename std::make_unsigned<T>::type, N> b) {
  return sycl_cts::math::run_func_on_vector<typename upsample_t<T>::type, T, N>(
      [](T x, T y) { return upsample(x, y); }, a, b);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<typename upsample_t<T>::type, N> upsample(
    sycl::marray<T, N> a,
    sycl::marray<typename std::make_unsigned<T>::type, N> b) {
  return sycl_cts::math::run_func_on_marray<typename upsample_t<T>::type, T, N>(
      [](T x, T y) { return upsample(x, y); }, a, b);
}
#endif

/* return number of non zero bits in x */
template <typename T>
T popcount(T x) {
  int lz = 0;
  for (int i = 0; i < sycl_cts::math::num_bits(x); i++)
    if (x & (1ull << i)) lz++;
  return lz;
}
MAKE_VEC_AND_MARRAY_VERSIONS(popcount)

/* fast multiply add 24bits */
sycl_cts::resultRef<int32_t> mad24(int32_t x, int32_t y, int32_t z);
sycl_cts::resultRef<uint32_t> mad24(uint32_t x, uint32_t y, uint32_t z);

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> mad24(sycl::vec<T, N> a, sycl::vec<T, N> b,
                                           sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return mad24(x, y, z); }, a, b, c);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> mad24(sycl::marray<T, N> a,
                                              sycl::marray<T, N> b,
                                              sycl::marray<T, N> c) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y, T z) { return mad24(x, y, z); }, a, b, c);
}
#endif

/* fast multiply 24bits */
sycl_cts::resultRef<int32_t> mul24(int32_t x, int32_t y);
sycl_cts::resultRef<uint32_t> mul24(uint32_t x, uint32_t y);

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> mul24(sycl::vec<T, N> a,
                                           sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return mul24(x, y); }, a, b);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> mul24(sycl::marray<T, N> a,
                                              sycl::marray<T, N> b) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y) { return mul24(x, y); }, a, b);
}
#endif

// Common functions

// clamp is in Integer functions

/* degrees */
sycl::half degrees(sycl::half);
float degrees(float a);
double degrees(double a);
MAKE_VEC_AND_MARRAY_VERSIONS(degrees)

// max and min are in Integer functions

/* mix */
sycl_cts::resultRef<sycl::half> mix(const sycl::half a, const sycl::half b,
                                    const sycl::half c);
sycl_cts::resultRef<float> mix(const float a, const float b, const float c);
sycl_cts::resultRef<double> mix(const double a, const double b, const double c);

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> mix(sycl::vec<T, N> a, sycl::vec<T, N> b,
                                         sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return mix(x, y, z); }, a, b, c);
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> mix(sycl::vec<T, N> a, sycl::vec<T, N> b,
                                         T c) {
  sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    sycl_cts::resultRef<T> element = mix(getElement(a, i), getElement(b, i), c);
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::vec<T, N>>(res, undefined);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> mix(sycl::marray<T, N> a,
                                            sycl::marray<T, N> b,
                                            sycl::marray<T, N> c) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y, T z) { return mix(x, y, z); }, a, b, c);
}
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> mix(sycl::marray<T, N> a,
                                            sycl::marray<T, N> b, T c) {
  sycl::marray<T, N> res;
  std::map<int, bool> undefined;
  for (size_t i = 0; i < N; i++) {
    sycl_cts::resultRef<T> element = mix(a[i], b[i], c);
    if (element.undefined.empty())
      res[i] = element.res;
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::marray<T, N>>(res, undefined);
}
#endif

/* radians */
sycl::half radians(sycl::half);
float radians(float a);
double radians(double a);
MAKE_VEC_AND_MARRAY_VERSIONS(radians)

/* step */
sycl::half step(sycl::half a, sycl::half b);
float step(float a, float b);
double step(double a, double b);
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(step)

template <typename T, int N>
sycl::vec<T, N> step(T a, sycl::vec<T, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, step(a, getElement(b, i)));
  }
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> step(T a, sycl::marray<T, N> b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = step(a, b[i]);
  }
  return res;
}
#endif

/* smoothstep */
sycl_cts::resultRef<sycl::half> smoothstep(sycl::half a, sycl::half b,
                                           sycl::half c);
sycl_cts::resultRef<float> smoothstep(float a, float b, float c);
sycl_cts::resultRef<double> smoothstep(double a, double b, double c);

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> smoothstep(sycl::vec<T, N> a,
                                                sycl::vec<T, N> b,
                                                sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return smoothstep(x, y, z); }, a, b, c);
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> smoothstep(T a, T b, sycl::vec<T, N> c) {
  sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    sycl_cts::resultRef<T> element = smoothstep(a, b, getElement(c, i));
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::vec<T, N>>(res, undefined);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> smoothstep(sycl::marray<T, N> a,
                                                   sycl::marray<T, N> b,
                                                   sycl::marray<T, N> c) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y, T z) { return smoothstep(x, y, z); }, a, b, c);
}
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> smoothstep(T a, T b,
                                                   sycl::marray<T, N> c) {
  sycl::marray<T, N> res;
  std::map<int, bool> undefined;
  for (size_t i = 0; i < N; i++) {
    sycl_cts::resultRef<T> element = smoothstep(a, b, c[i]);
    if (element.undefined.empty())
      res[i] = element.res;
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::marray<T, N>>(res, undefined);
}
#endif

/* sign */
sycl::half sign(sycl::half a);
float sign(float a);
double sign(double a);
MAKE_VEC_AND_MARRAY_VERSIONS(sign)

// Math Functions

template <typename T>
struct higher_accuracy;

template <>
struct higher_accuracy<sycl::half> {
  using type = float;
};
template <>
struct higher_accuracy<float> {
  using type = double;
};
template <>
struct higher_accuracy<double> {
  using type = long double;
};

template <typename T, int N>
struct higher_accuracy<sycl::vec<T, N>> {
  using type = sycl::vec<typename higher_accuracy<T>::type, N>;
};
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
struct higher_accuracy<sycl::marray<T, N>> {
  using type = sycl::marray<typename higher_accuracy<T>::type, N>;
};
#endif

template <typename T>
T acos(T a) {
  return std::acos(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(acos)

template <typename T>
T acosh(T a) {
  return std::acosh(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(acosh)

sycl::half acospi(sycl::half a);
float acospi(float a);
double acospi(double a);
MAKE_VEC_AND_MARRAY_VERSIONS(acospi)

template <typename T>
T asin(T a) {
  return std::asin(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(asin)

template <typename T>
T asinh(T a) {
  return std::asinh(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(asinh)

sycl::half asinpi(sycl::half a);
float asinpi(float a);
double asinpi(double a);
MAKE_VEC_AND_MARRAY_VERSIONS(asinpi)

template <typename T>
T atan(T a) {
  return std::atan(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(atan)

template <typename T>
T atan2(T a, T b) {
  return std::atan2(static_cast<typename higher_accuracy<T>::type>(a), b);
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(atan2)

template <typename T>
T atanh(T a) {
  return std::atanh(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(atanh)

sycl::half atanpi(sycl::half a);
float atanpi(float a);
double atanpi(double a);
MAKE_VEC_AND_MARRAY_VERSIONS(atanpi)

sycl::half atan2pi(sycl::half a, sycl::half b);
float atan2pi(float a, float b);
double atan2pi(double a, double b);
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(atan2pi)

template <typename T>
T cbrt(T a) {
  return std::cbrt(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(cbrt)

using std::ceil;
MAKE_VEC_AND_MARRAY_VERSIONS(ceil)

using std::copysign;
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(copysign)

template <typename T>
T cos(T a) {
  return std::cos(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(cos)

template <typename T>
T cosh(T a) {
  return std::cosh(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(cosh)

sycl::half cospi(sycl::half a);
float cospi(float a);
double cospi(double a);
MAKE_VEC_AND_MARRAY_VERSIONS(cospi)

template <typename T>
T erfc(T a) {
  return std::erfc(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(erfc)

template <typename T>
T erf(T a) {
  return std::erf(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(erf)

template <typename T>
T exp(T a) {
  return std::exp(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(exp)

template <typename T>
T exp2(T a) {
  return std::exp2(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(exp2)

template <typename T>
T exp10(T a) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(10),
                  static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(exp10)

template <typename T>
T expm1(T a) {
  return std::expm1(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(expm1)

using std::fabs;
MAKE_VEC_AND_MARRAY_VERSIONS(fabs)

using std::fdim;
sycl::half fdim(sycl::half a, sycl::half b);
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(fdim)

using std::floor;
MAKE_VEC_AND_MARRAY_VERSIONS(floor)

sycl::half fma(sycl::half a, sycl::half b, sycl::half c);
float fma(float a, float b, float c);
double fma(double a, double b, double c);

MAKE_VEC_AND_MARRAY_VERSIONS_3ARGS(fma)

using std::fmax;
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(fmax)
MAKE_VEC_AND_MARRAY_VERSIONS_WITH_SCALAR(fmax)

using std::fmin;
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(fmin)
MAKE_VEC_AND_MARRAY_VERSIONS_WITH_SCALAR(fmin)

using std::fmod;
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(fmod)

sycl::half fract(sycl::half a, sycl::half *b);
float fract(float a, float *b);
double fract(double a, double *b);

template <typename T, int N>
sycl::vec<T, N> fract(sycl::vec<T, N> a, sycl::vec<T, N> *b) {
  sycl::vec<T, N> res;
  sycl::vec<T, N> resPtr;
  for (int i = 0; i < N; i++) {
    T value;
    setElement<T, N>(res, i, fract(getElement(a, i), &value));
    setElement<T, N>(resPtr, i, value);
  }
  *b = resPtr;
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> fract(sycl::marray<T, N> a, sycl::marray<T, N> *b) {
  sycl::marray<T, N> res;
  sycl::marray<T, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    T value;
    res[i] = fract(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}
#endif

using std::frexp;
template <typename T, int N>
sycl::vec<T, N> frexp(sycl::vec<T, N> a, sycl::vec<int, N> *b) {
  sycl::vec<T, N> res;
  sycl::vec<int, N> resPtr;
  for (int i = 0; i < N; i++) {
    int value;
    setElement<T, N>(res, i, frexp(getElement(a, i), &value));
    setElement<int, N>(resPtr, i, value);
  }
  *b = resPtr;
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> frexp(sycl::marray<T, N> a, sycl::marray<int, N> *b) {
  sycl::marray<T, N> res;
  sycl::marray<int, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    int value;
    res[i] = frexp(a[i], &value);
    ;
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}
#endif

template <typename T>
T hypot(T a, T b) {
  return std::hypot(static_cast<typename higher_accuracy<T>::type>(a), b);
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(hypot)

using std::ilogb;
template <typename T, int N>
sycl::vec<int, N> ilogb(sycl::vec<T, N> a) {
  sycl::vec<int, N> res;
  for (int i = 0; i < N; i++) {
    setElement<int, N>(res, i, ilogb(getElement<T, N>(a, i)));
  }
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<int, N> ilogb(sycl::marray<T, N> a) {
  sycl::marray<int, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = ilogb(a[i]);
  }
  return res;
}
#endif

using std::ldexp;
template <typename T, int N>
sycl::vec<T, N> ldexp(sycl::vec<T, N> a, sycl::vec<int, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i,
                     ldexp(getElement<T, N>(a, i), getElement<int, N>(b, i)));
  }
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> ldexp(sycl::marray<T, N> a, sycl::marray<int, N> b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = ldexp(a[i], b[i]);
  }
  return res;
}
#endif
template <typename T, int N>
sycl::vec<T, N> ldexp(sycl::vec<T, N> a, int b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, ldexp(getElement<T, N>(a, i), b));
  }
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> ldexp(sycl::marray<T, N> a, int b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = ldexp(a[i], b);
  }
  return res;
}
#endif

using std::lgamma;
MAKE_VEC_AND_MARRAY_VERSIONS(lgamma)

template <typename T>
T lgamma_r(T a, int *b) {
  *b = (std::tgamma(a) > 0) ? 1 : -1;
  return std::lgamma(a);
}
template <typename T, int N>
sycl::vec<T, N> lgamma_r(sycl::vec<T, N> a, sycl::vec<int, N> *b) {
  sycl::vec<T, N> res;
  sycl::vec<int, N> resPtr;
  for (int i = 0; i < N; i++) {
    int value;
    setElement<T, N>(res, i, lgamma_r(getElement(a, i), &value));
    setElement<int, N>(resPtr, i, value);
  }
  *b = resPtr;
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> lgamma_r(sycl::marray<T, N> a, sycl::marray<int, N> *b) {
  sycl::marray<T, N> res;
  sycl::marray<int, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    int value;
    res[i] = lgamma_r(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}
#endif

template <typename T>
T log(T a) {
  return std::log(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(log)

template <typename T>
T log2(T a) {
  return std::log2(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(log2)

template <typename T>
T log10(T a) {
  return std::log10(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(log10)

template <typename T>
T log1p(T a) {
  return std::log1p(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(log1p)

using std::logb;
MAKE_VEC_AND_MARRAY_VERSIONS(logb)

template <typename T>
T mad(T a, T b, T c) {
  return a * b + c;
}
MAKE_VEC_AND_MARRAY_VERSIONS_3ARGS(mad)

template <typename T>
T maxmag(T a, T b) {
  if (fabs(a) > fabs(b))
    return a;
  else if (fabs(b) > fabs(a))
    return b;
  return fmax(a, b);
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(maxmag)

template <typename T>
T minmag(T a, T b) {
  if (fabs(a) < fabs(b))
    return a;
  else if (fabs(b) < fabs(a))
    return b;
  return fmin(a, b);
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(minmag)

using std::modf;
sycl::half modf(sycl::half a, sycl::half *b);
template <typename T, int N>
sycl::vec<T, N> modf(sycl::vec<T, N> a, sycl::vec<T, N> *b) {
  sycl::vec<T, N> res;
  sycl::vec<T, N> resPtr;
  for (int i = 0; i < N; i++) {
    T value;
    setElement<T, N>(res, i, modf(getElement(a, i), &value));
    setElement<T, N>(resPtr, i, value);
  }
  *b = resPtr;
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> modf(sycl::marray<T, N> a, sycl::marray<T, N> *b) {
  sycl::marray<T, N> res;
  sycl::marray<T, N> resPtr;
  for (int i = 0; i < N; i++) {
    T value;
    res[i] = modf(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}
#endif

float nan(unsigned int a);
double nan(unsigned long a);
double nan(unsigned long long a);
template <int N>
sycl::vec<float, N> nan(sycl::vec<unsigned int, N> a) {
  return sycl_cts::math::run_func_on_vector<float, unsigned int, N>(
      [](unsigned int x) { return nan(x); }, a);
}
template <typename T, int N>
std::enable_if_t<std::is_same_v<unsigned long, T> ||
                     std::is_same_v<unsigned long long, T>,
                 sycl::vec<double, N>>
nan(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<double, T, N>(
      [](T x) { return nan(x); }, a);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <size_t N>
sycl::marray<float, N> nan(sycl::marray<unsigned int, N> a) {
  return sycl_cts::math::run_func_on_marray<float, unsigned int, N>(
      [](unsigned int x) { return nan(x); }, a);
}
template <typename T, size_t N>
std::enable_if_t<std::is_same_v<unsigned long, T> ||
                     std::is_same_v<unsigned long long, T>,
                 sycl::marray<double, N>>
nan(sycl::marray<T, N> a) {
  return sycl_cts::math::run_func_on_marray<double, T, N>(
      [](T x) { return nan(x); }, a);
}
#endif

using std::nextafter;
sycl::half nextafter(sycl::half a, sycl::half b);
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(nextafter)

template <typename T>
T pow(T a, T b) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
                  static_cast<typename higher_accuracy<T>::type>(b));
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(pow)

template <typename T>
T pown(T a, int b) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
                  static_cast<typename higher_accuracy<T>::type>(b));
}
template <typename T, int N>
sycl::vec<T, N> pown(sycl::vec<T, N> a, sycl::vec<int, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i,
                     pown(getElement<T, N>(a, i), getElement<int, N>(b, i)));
  }
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> pown(sycl::marray<T, N> a, sycl::marray<int, N> b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = pown(a[i], b[i]);
  }
  return res;
}
#endif

template <typename T>
sycl_cts::resultRef<T> powr(T a, T b) {
  if (a < 0) return sycl_cts::resultRef<T>(T(), true);
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
                  static_cast<typename higher_accuracy<T>::type>(b));
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> powr(sycl::vec<T, N> a,
                                          sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return powr(x, y); }, a, b);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> powr(sycl::marray<T, N> a,
                                             sycl::marray<T, N> b) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y) { return powr(x, y); }, a, b);
}
#endif

using std::remainder;
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(remainder)

template <typename T>
T remquo(T x, T y, int *quo) {
  return reference_remquol(x, y, quo);
}

template <typename T, int N>
sycl::vec<T, N> remquo(sycl::vec<T, N> a, sycl::vec<T, N> b,
                       sycl::vec<int, N> *c) {
  sycl::vec<T, N> res;
  sycl::vec<int, N> resPtr;
  for (int i = 0; i < N; i++) {
    int value;
    setElement<T, N>(res, i,
                     remquo(getElement(a, i), getElement(b, i), &value));
    setElement<int, N>(resPtr, i, value);
  }
  *c = resPtr;
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> remquo(sycl::marray<T, N> a, sycl::marray<T, N> b,
                          sycl::marray<int, N> *c) {
  sycl::marray<T, N> res;
  sycl::marray<int, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    int value;
    res[i] = remquo(a[i], b[i], &value);
    resPtr[i] = value;
  }
  *c = resPtr;
  return res;
}
#endif

using std::rint;
MAKE_VEC_AND_MARRAY_VERSIONS(rint)

template <typename T>
T rootn(T a, int b) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
                  static_cast<typename higher_accuracy<T>::type>(1.0 / b));
}
template <typename T, int N>
sycl::vec<T, N> rootn(sycl::vec<T, N> a, sycl::vec<int, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i,
                     rootn(getElement<T, N>(a, i), getElement<int, N>(b, i)));
  }
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> rootn(sycl::marray<T, N> a, sycl::marray<int, N> b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = rootn(a[i], b[i]);
  }
  return res;
}
#endif

using std::round;
MAKE_VEC_AND_MARRAY_VERSIONS(round)

template <typename T>
T rsqrt(T a) {
  return 1 / std::sqrt(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(rsqrt)

template <typename T>
T sincos(T a, T *b) {
  *b = std::cos(static_cast<typename higher_accuracy<T>::type>(a));
  return std::sin(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N>
sycl::vec<T, N> sincos(sycl::vec<T, N> a, sycl::vec<T, N> *b) {
  sycl::vec<T, N> res;
  sycl::vec<T, N> resPtr;
  for (int i = 0; i < N; i++) {
    T value;
    setElement<T, N>(res, i, sincos(getElement(a, i), &value));
    setElement<T, N>(resPtr, i, value);
  }
  *b = resPtr;
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> sincos(sycl::marray<T, N> a, sycl::marray<T, N> *b) {
  sycl::marray<T, N> res;
  sycl::marray<T, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    T value;
    res[i] = sincos(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}
#endif

template <typename T>
T sin(T a) {
  return std::sin(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(sin)

template <typename T>
T sinh(T a) {
  return std::sinh(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(sinh)

sycl::half sinpi(sycl::half a);
float sinpi(float a);
double sinpi(double a);
MAKE_VEC_AND_MARRAY_VERSIONS(sinpi)

template <typename T>
T sqrt(T a) {
  return std::sqrt(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(sqrt)

template <typename T>
T tan(T a) {
  return std::tan(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(tan)

template <typename T>
T tanh(T a) {
  return std::tanh(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(tanh)

sycl::half tanpi(sycl::half a);
float tanpi(float a);
double tanpi(double a);
MAKE_VEC_AND_MARRAY_VERSIONS(tanpi)

template <typename T>
T tgamma(T a) {
  return std::tgamma(static_cast<typename higher_accuracy<T>::type>(a));
}
MAKE_VEC_AND_MARRAY_VERSIONS(tgamma)

using std::trunc;
MAKE_VEC_AND_MARRAY_VERSIONS(trunc)

template <typename T>
T recip(T a) {
  return 1.0 / a;
}
MAKE_VEC_AND_MARRAY_VERSIONS(recip)

template <typename T>
T divide(T a, T b) {
  return a / b;
}
MAKE_VEC_AND_MARRAY_VERSIONS_2ARGS(divide)

// Geometric functions

sycl::float4 cross(sycl::float4 p0, sycl::float4 p1);
sycl::float3 cross(sycl::float3 p0, sycl::float3 p1);
sycl::double4 cross(sycl::double4 p0, sycl::double4 p1);
sycl::double3 cross(sycl::double3 p0, sycl::double3 p1);

// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
sycl::mfloat4 cross(sycl::mfloat4 p0, sycl::mfloat4 p1);
sycl::mfloat3 cross(sycl::mfloat3 p0, sycl::mfloat3 p1);
sycl::mdouble4 cross(sycl::mdouble4 p0, sycl::mdouble4 p1);
sycl::mdouble3 cross(sycl::mdouble3 p0, sycl::mdouble3 p1);
#endif

template <typename T>
T dot(T p0, T p1) {
  return p0 * p1;
}
template <typename T, int N>
T dot(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  T res = 0;
  for (int i = 0; i < N; i++)
    res += getElement<T, N>(a, i) * getElement<T, N>(b, i);
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
T dot(sycl::marray<T, N> a, sycl::marray<T, N> b) {
  T res = 0;
  for (size_t i = 0; i < N; i++) res += a[i] * b[i];
  return res;
}
#endif

template <typename T>
auto length(T p) {
  return sqrt(reference::dot(p, p));
}

template <typename T>
auto distance(T p0, T p1) {
  return reference::length(p0 - p1);
}

template <typename T>
T normalize(T p) {
  if (p < 0) return -1;
  return 1;
}
template <typename T, int N>
sycl::vec<T, N> normalize(sycl::vec<T, N> a) {
  sycl::vec<T, N> res;
  T len_a = reference::length(a);
  if (len_a == 0) return sycl::vec<T, N>(0);
  for (int i = 0; i < N; i++)
    setElement<T, N>(res, i, getElement<T, N>(a, i) / len_a);
  return res;
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> normalize(sycl::marray<T, N> a) {
  sycl::marray<T, N> res;
  T len_a = reference::length(a);
  if (len_a == 0) return sycl::marray<T, N>(0);
  for (size_t i = 0; i < N; i++) res[i] = a[i] / len_a;
  return res;
}
#endif

sycl::half fast_dot(float p0);
sycl::half fast_dot(sycl::float2 p0);
sycl::half fast_dot(sycl::float3 p0);
sycl::half fast_dot(sycl::float4 p0);
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
sycl::half fast_dot(sycl::mfloat2 p0);
sycl::half fast_dot(sycl::mfloat3 p0);
sycl::half fast_dot(sycl::mfloat4 p0);
#endif

template <typename T>
float fast_length(T p0) {
  return sqrt(fast_dot(p0));
}

template <typename T>
float fast_distance(T p0, T p1) {
  return reference::fast_length(p0 - p1);
}

template <typename T>
T fast_normalize(T p0) {
  return p0 * rsqrt(fast_dot(p0));
}

}  // namespace reference

#endif  // __SYCLCTS_UTIL_MATH_REFERENCE_H
