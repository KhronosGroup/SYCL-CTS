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

auto constexpr islessgreater_func = [](const auto& x, const auto& y) {
  return (x < y) || (x > y);
};
template <typename T>
auto islessgreater(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher(islessgreater_func, a, b);
}

auto constexpr isordered_func = [](const auto& x, const auto& y) {
  return (x == x) && (y == y);
};
template <typename T>
auto isordered(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher(isordered_func, a, b);
}

auto constexpr isunordered_func = [](const auto& x, const auto& y) {
  return !((x == x) && (y == y));
};
template <typename T>
auto isunordered(T a, T b) {
  return sycl_cts::math::rel_func_dispatcher(isunordered_func, a, b);
}

/* one argument relational reference */
auto constexpr isfinite_func = [](const auto& x) { return std::isfinite(x); };
template <typename T>
auto isfinite(T a) {
  return sycl_cts::math::rel_func_dispatcher(isfinite_func, a);
}

auto constexpr isinf_func = [](const auto& x) { return std::isinf(x); };
template <typename T>
auto isinf(T a) {
  return sycl_cts::math::rel_func_dispatcher(isinf_func, a);
}

auto constexpr isnan_func = [](const auto& x) { return std::isnan(x); };
template <typename T>
auto isnan(T a) {
  return sycl_cts::math::rel_func_dispatcher(isnan_func, a);
}

auto constexpr isnormal_func = [](const auto& x) { return std::isnormal(x); };
template <typename T>
auto isnormal(T a) {
  return sycl_cts::math::rel_func_dispatcher(isnormal_func, a);
}

auto constexpr signbit_func = [](const auto& x) { return std::signbit(x); };
template <typename T>
auto signbit(T a) {
  return sycl_cts::math::rel_func_dispatcher(signbit_func, a);
}

template <typename T>
bool any(T x) {
  return sycl_cts::math::if_msb_set(x);
}

template <typename T>
bool all(T x) {
  return sycl_cts::math::if_msb_set(x);
}

template <typename T>
T bitselect(T a, T b, T c) {
  return (c & b) | (~c & a);
}
float bitselect(float a, float b, float c);

template <typename T>
T select(T a, T b, bool c) {
  return c ? b : a;
}

/* absolute value */
template <typename T>
sycl_cts::resultRef<T> abs(T x) {
  using U = std::make_unsigned_t<T>;
  T result = x < 0 ? T(-U(x)) : x;
  return result < 0 ? sycl_cts::resultRef<T>(0, true) : result;
}

/* absolute difference */
template <typename T>
sycl_cts::resultRef<T> abs_diff(T a, T b) {
  using U = std::make_unsigned_t<T>;
  T h = (a > b) ? a : b;
  T l = (a <= b) ? a : b;
  // Using two's-complement and that unsigned integer underflow is defined as
  // modulo 2^n we get the result by computing the distance based on signed
  // comparison.
  U result = static_cast<U>(h) - static_cast<U>(l);
  return result > std::numeric_limits<T>::max()
             ? sycl_cts::resultRef<T>(0, true)
             : T(result);
}

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

/* half add */
template <typename T>
T hadd(T a, T b) {
  if (std::is_unsigned<T>::value) return (a >> 1) + (b >> 1) + ((a & b) & 0x1);
  return (a >> 1) + (b >> 1) + (a & b & 1);
}

/* round up half add */
template <typename T>
T rhadd(T a, T b) {
  return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

/* clamp */
template <typename T>
sycl_cts::resultRef<T> clamp(T v, T minv, T maxv) {
  if (minv > maxv) return sycl_cts::resultRef<T>(T(), true);
  return (v < minv) ? minv : ((v > maxv) ? maxv : v);
}

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

/* maximum value */
template <typename T>
sycl_cts::resultRef<T> max(T a, T b) {
  if constexpr (std::is_integral_v<T>)
    return (a < b) ? b : a;
  else if (std::isfinite(a) && std::isfinite(b))
    return (a < b) ? b : a;
  return sycl_cts::resultRef<T>(T(), true);
}

/* minimum value */
template <typename T>
sycl_cts::resultRef<T> min(T a, T b) {
  if constexpr (std::is_integral_v<T>)
    return (b < a) ? b : a;
  else if (std::isfinite(a) && std::isfinite(b))
    return (b < a) ? b : a;
  return sycl_cts::resultRef<T>(T(), true);
}

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

/* multiply add, get high part */
template <typename T>
T mad_hi(T x, T y, T z) {
  return mul_hi(x, y) + z;
}

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

/* return number of non zero bits in x */
template <typename T>
T popcount(T x) {
  int lz = 0;
  for (int i = 0; i < sycl_cts::math::num_bits(x); i++)
    if (x & (1ull << i)) lz++;
  return lz;
}

/* fast multiply add 24bits */
sycl_cts::resultRef<int32_t> mad24(int32_t x, int32_t y, int32_t z);
sycl_cts::resultRef<uint32_t> mad24(uint32_t x, uint32_t y, uint32_t z);

/* fast multiply 24bits */
sycl_cts::resultRef<int32_t> mul24(int32_t x, int32_t y);
sycl_cts::resultRef<uint32_t> mul24(uint32_t x, uint32_t y);

// Common functions

float degrees(float a);
sycl_cts::resultRef<float> mix(const float a, const float b, const float c);
float radians(float a);
float step(float a, float b);
sycl_cts::resultRef<float> smoothstep(float a, float b, float c);
float sign(float a);

// Math Functions

template <typename T>
struct higher_accuracy;

template <>
struct higher_accuracy<float> {
  using type = double;
};

template <typename T>
T acos(T a) {
  return std::acos(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T acosh(T a) {
  return std::acosh(static_cast<typename higher_accuracy<T>::type>(a));
}

float acospi(float a);

template <typename T>
T asin(T a) {
  return std::asin(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T asinh(T a) {
  return std::asinh(static_cast<typename higher_accuracy<T>::type>(a));
}

float asinpi(float a);

template <typename T>
T atan(T a) {
  return std::atan(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T atan2(T a, T b) {
  return std::atan2(static_cast<typename higher_accuracy<T>::type>(a), b);
}

template <typename T>
T atanh(T a) {
  return std::atanh(static_cast<typename higher_accuracy<T>::type>(a));
}

float atanpi(float a);

float atan2pi(float a, float b);

template <typename T>
T cbrt(T a) {
  return std::cbrt(static_cast<typename higher_accuracy<T>::type>(a));
}

using std::ceil;
using std::copysign;

template <typename T>
T cos(T a) {
  return std::cos(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T cosh(T a) {
  return std::cosh(static_cast<typename higher_accuracy<T>::type>(a));
}

float cospi(float a);

template <typename T>
T erfc(T a) {
  return std::erfc(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T erf(T a) {
  return std::erf(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T exp(T a) {
  return std::exp(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T exp2(T a) {
  return std::exp2(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T exp10(T a) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(10),
                  static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T expm1(T a) {
  return std::expm1(static_cast<typename higher_accuracy<T>::type>(a));
}

using std::fabs;
using std::fdim;
using std::floor;
float fma(float a, float b, float c);
using std::fmax;
using std::fmin;
using std::fmod;
float fract(float a, float* b);
using std::frexp;

template <typename T>
T hypot(T a, T b) {
  return std::hypot(static_cast<typename higher_accuracy<T>::type>(a), b);
}

using std::ilogb;
using std::ldexp;
using std::lgamma;

template <typename T>
T lgamma_r(T a, int* b) {
  *b = (std::tgamma(a) > 0) ? 1 : -1;
  return std::lgamma(a);
}

template <typename T>
T log(T a) {
  return std::log(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T log2(T a) {
  return std::log2(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T log10(T a) {
  return std::log10(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T log1p(T a) {
  return std::log1p(static_cast<typename higher_accuracy<T>::type>(a));
}

using std::logb;

template <typename T>
T mad(T a, T b, T c) {
  return a * b + c;
}

template <typename T>
T maxmag(T a, T b) {
  if (fabs(a) > fabs(b))
    return a;
  else if (fabs(b) > fabs(a))
    return b;
  return fmax(a, b);
}

template <typename T>
T minmag(T a, T b) {
  if (fabs(a) < fabs(b))
    return a;
  else if (fabs(b) < fabs(a))
    return b;
  return fmin(a, b);
}

using std::modf;
float nan(unsigned int a);
using std::nextafter;

template <typename T>
T pow(T a, T b) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
                  static_cast<typename higher_accuracy<T>::type>(b));
}

template <typename T>
T pown(T a, int b) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
                  static_cast<typename higher_accuracy<T>::type>(b));
}

template <typename T>
sycl_cts::resultRef<T> powr(T a, T b) {
  if (a < 0) return sycl_cts::resultRef<T>(T(), true);
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
                  static_cast<typename higher_accuracy<T>::type>(b));
}

using std::remainder;

template <typename T>
T remquo(T x, T y, int* quo) {
  return reference_remquol(x, y, quo);
}

using std::rint;

template <typename T>
T rootn(T a, int b) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
                  static_cast<typename higher_accuracy<T>::type>(1.0 / b));
}

using std::round;

template <typename T>
T rsqrt(T a) {
  return 1 / std::sqrt(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T sincos(T a, T* b) {
  *b = std::cos(static_cast<typename higher_accuracy<T>::type>(a));
  return std::sin(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T sin(T a) {
  return std::sin(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T sinh(T a) {
  return std::sinh(static_cast<typename higher_accuracy<T>::type>(a));
}

float sinpi(float a);

template <typename T>
T sqrt(T a) {
  return std::sqrt(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T tan(T a) {
  return std::tan(static_cast<typename higher_accuracy<T>::type>(a));
}

template <typename T>
T tanh(T a) {
  return std::tanh(static_cast<typename higher_accuracy<T>::type>(a));
}

float tanpi(float a);

template <typename T>
T tgamma(T a) {
  return std::tgamma(static_cast<typename higher_accuracy<T>::type>(a));
}

using std::trunc;

template <typename T>
T recip(T a) {
  return 1.0 / a;
}

template <typename T>
T divide(T a, T b) {
  return a / b;
}

// Geometric functions

sycl::float4 cross(sycl::float4 p0, sycl::float4 p1);
sycl::float3 cross(sycl::float3 p0, sycl::float3 p1);

sycl::mfloat4 cross(sycl::mfloat4 p0, sycl::mfloat4 p1);
sycl::mfloat3 cross(sycl::mfloat3 p0, sycl::mfloat3 p1);

template <typename T>
T dot(T p0, T p1) {
  return p0 * p1;
}

template <typename T>
T normalize(T p) {
  if (p < 0) return -1;
  return 1;
}

#if SYCL_CTS_ENABLE_HALF_TESTS

template <>
struct higher_accuracy<sycl::half> {
  using type = float;
};

sycl::half bitselect(sycl::half a, sycl::half b, sycl::half c);
sycl::half degrees(sycl::half);
sycl_cts::resultRef<sycl::half> mix(const sycl::half a, const sycl::half b,
                                    const sycl::half c);
sycl::half radians(sycl::half);
sycl::half step(sycl::half a, sycl::half b);
sycl_cts::resultRef<sycl::half> smoothstep(sycl::half a, sycl::half b,
                                           sycl::half c);
sycl::half sign(sycl::half a);
sycl::half acospi(sycl::half a);
sycl::half asinpi(sycl::half a);
sycl::half atanpi(sycl::half a);
sycl::half atan2pi(sycl::half a, sycl::half b);
sycl::half cospi(sycl::half a);
sycl::half fdim(sycl::half a, sycl::half b);
sycl::half fma(sycl::half a, sycl::half b, sycl::half c);
sycl::half fract(sycl::half a, sycl::half* b);
sycl::half modf(sycl::half a, sycl::half* b);
sycl::half nan(unsigned short a);

template <int N>
sycl::vec<sycl::half, N> nan(sycl::vec<unsigned short, N> a) {
  return sycl_cts::math::run_func_on_vector<sycl::half, unsigned short, N>(
      [](unsigned short x) { return nan(x); }, a);
}

template <size_t N>
sycl::marray<sycl::half, N> nan(sycl::marray<unsigned short, N> a) {
  return sycl_cts::math::run_func_on_marray<sycl::half, unsigned short, N>(
      [](unsigned short x) { return nan(x); }, a);
}

sycl::half nextafter(sycl::half a, sycl::half b);
sycl::half sinpi(sycl::half a);
sycl::half tanpi(sycl::half a);
#endif  // SYCL_CTS_ENABLE_HALF_TESTS

sycl::half fast_dot(float p0);
sycl::half fast_dot(sycl::float2 p0);
sycl::half fast_dot(sycl::float3 p0);
sycl::half fast_dot(sycl::float4 p0);

sycl::half fast_dot(sycl::mfloat2 p0);
sycl::half fast_dot(sycl::mfloat3 p0);
sycl::half fast_dot(sycl::mfloat4 p0);

#if SYCL_CTS_ENABLE_DOUBLE_TESTS

template <>
struct higher_accuracy<double> {
  using type = long double;
};

double bitselect(double a, double b, double c);
double degrees(double a);
sycl_cts::resultRef<double> mix(const double a, const double b, const double c);
double radians(double a);
double step(double a, double b);
sycl_cts::resultRef<double> smoothstep(double a, double b, double c);
double sign(double a);

double acospi(double a);
double asinpi(double a);
double atanpi(double a);
double atan2pi(double a, double b);
double cospi(double a);
double fma(double a, double b, double c);
double fract(double a, double* b);
double nan(unsigned long a);
double nan(unsigned long long a);

template <typename T, int N>
std::enable_if_t<std::is_same_v<unsigned long, T> ||
                     std::is_same_v<unsigned long long, T>,
                 sycl::vec<double, N>>
nan(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<double, T, N>(
      [](T x) { return nan(x); }, a);
}

template <typename T, size_t N>
std::enable_if_t<std::is_same_v<unsigned long, T> ||
                     std::is_same_v<unsigned long long, T>,
                 sycl::marray<double, N>>
nan(sycl::marray<T, N> a) {
  return sycl_cts::math::run_func_on_marray<double, T, N>(
      [](T x) { return nan(x); }, a);
}

double sinpi(double a);
double tanpi(double a);

sycl::double4 cross(sycl::double4 p0, sycl::double4 p1);
sycl::double3 cross(sycl::double3 p0, sycl::double3 p1);

sycl::mdouble4 cross(sycl::mdouble4 p0, sycl::mdouble4 p1);
sycl::mdouble3 cross(sycl::mdouble3 p0, sycl::mdouble3 p1);

#endif  // SYCL_CTS_ENABLE_DOUBLE_TESTS

// sycl::vec overloads of the above. Some vector functions reference their
// scalar counterparts, so all scalar overloads (float, half, double) must have
// been declared previously - otherwise they will not participate in overload
// resolution, even if the point of template instantiation is outside this file.
#define MAKE_VEC_VERSION(func)                          \
  template <typename T, int N>                          \
  sycl::vec<T, N> func(sycl::vec<T, N> a) {             \
    return sycl_cts::math::run_func_on_vector<T, T, N>( \
        [](T x) { return func(x); }, a);                \
  }

#define MAKE_VEC_VERSION_2ARGS(func)                           \
  template <typename T, int N>                                 \
  sycl::vec<T, N> func(sycl::vec<T, N> a, sycl::vec<T, N> b) { \
    return sycl_cts::math::run_func_on_vector<T, T, N>(        \
        [](T x, T y) { return func(x, y); }, a, b);            \
  }

#define MAKE_VEC_VERSION_3ARGS(func)                           \
  template <typename T, int N>                                 \
  sycl::vec<T, N> func(sycl::vec<T, N> a, sycl::vec<T, N> b,   \
                       sycl::vec<T, N> c) {                    \
    return sycl_cts::math::run_func_on_vector<T, T, N>(        \
        [](T x, T y, T z) { return func(x, y, z); }, a, b, c); \
  }

#define MAKE_VEC_VERSION_WITH_SCALAR(func)              \
  template <typename T, int N>                          \
  sycl::vec<T, N> func(sycl::vec<T, N> a, T b) {        \
    return sycl_cts::math::run_func_on_vector<T, T, N>( \
        [](T x, T y) { return func(x, y); }, a, b);     \
  }

// Common functions

template <typename T, int N>
int any(sycl::vec<T, N> a) {
  for (int i = 0; i < N; i++) {
    if (any(a[i]) == 1) return true;
  }
  return false;
}

template <typename T, int N>
int all(sycl::vec<T, N> a) {
  for (int i = 0; i < N; i++) {
    if (all(a[i]) == 0) return false;
  }
  return true;
}

MAKE_VEC_VERSION_3ARGS(bitselect)

template <typename T, typename K, int N>
sycl::vec<T, N> select(sycl::vec<T, N> a, sycl::vec<T, N> b,
                       sycl::vec<K, N> c) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    if (any(c[i]) == 1)
      res[i] = b[i];
    else
      res[i] = a[i];
  }
  return res;
}

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> abs(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x) { return abs(x); }, a);
}

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> abs_diff(sycl::vec<T, N> a,
                                              sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return abs_diff(x, y); }, a, b);
}

MAKE_VEC_VERSION_2ARGS(add_sat)
MAKE_VEC_VERSION_2ARGS(hadd)
MAKE_VEC_VERSION_2ARGS(rhadd)

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
    sycl_cts::resultRef<T> element = clamp(a[i], b, c);
    if (element.undefined.empty())
      res[i] = element.res;
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::vec<T, N>>(res, undefined);
}

MAKE_VEC_VERSION(clz)
MAKE_VEC_VERSION(ctz)

MAKE_VEC_VERSION_3ARGS(mad_sat)

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

MAKE_VEC_VERSION_2ARGS(mul_hi)
MAKE_VEC_VERSION_3ARGS(mad_hi)
MAKE_VEC_VERSION_2ARGS(rotate)
MAKE_VEC_VERSION_2ARGS(sub_sat)

template <typename T, int N>
sycl::vec<typename upsample_t<T>::type, N> upsample(
    sycl::vec<T, N> a, sycl::vec<typename std::make_unsigned<T>::type, N> b) {
  return sycl_cts::math::run_func_on_vector<typename upsample_t<T>::type, T, N>(
      [](T x, T y) { return upsample(x, y); }, a, b);
}

MAKE_VEC_VERSION(popcount)

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> mad24(sycl::vec<T, N> a, sycl::vec<T, N> b,
                                           sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return mad24(x, y, z); }, a, b, c);
}

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> mul24(sycl::vec<T, N> a,
                                           sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return mul24(x, y); }, a, b);
}

MAKE_VEC_VERSION(degrees)

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
    sycl_cts::resultRef<T> element = mix(a[i], b[i], c);
    if (element.undefined.empty())
      res[i] = element.res;
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::vec<T, N>>(res, undefined);
}

MAKE_VEC_VERSION(radians)

template <typename T, int N>
sycl::vec<T, N> step(T a, sycl::vec<T, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = step(a, b[i]);
  }
  return res;
}

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
    sycl_cts::resultRef<T> element = smoothstep(a, b, c[i]);
    if (element.undefined.empty())
      res[i] = element.res;
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::vec<T, N>>(res, undefined);
}

// Math functions

template <typename T, int N>
struct higher_accuracy<sycl::vec<T, N>> {
  using type = sycl::vec<typename higher_accuracy<T>::type, N>;
};

MAKE_VEC_VERSION(acos)
MAKE_VEC_VERSION(acosh)
MAKE_VEC_VERSION(acospi)
MAKE_VEC_VERSION(asin)
MAKE_VEC_VERSION(asinh)
MAKE_VEC_VERSION(asinpi)
MAKE_VEC_VERSION(atan)
MAKE_VEC_VERSION_2ARGS(atan2)
MAKE_VEC_VERSION(atanh)
MAKE_VEC_VERSION(atanpi)
MAKE_VEC_VERSION_2ARGS(atan2pi)
MAKE_VEC_VERSION(cbrt)
MAKE_VEC_VERSION(ceil)
MAKE_VEC_VERSION_2ARGS(copysign)
MAKE_VEC_VERSION(cos)
MAKE_VEC_VERSION(cosh)
MAKE_VEC_VERSION(cospi)
MAKE_VEC_VERSION(erfc)
MAKE_VEC_VERSION(erf)
MAKE_VEC_VERSION(exp)
MAKE_VEC_VERSION(exp2)
MAKE_VEC_VERSION(exp10)
MAKE_VEC_VERSION(expm1)
MAKE_VEC_VERSION(fabs)
MAKE_VEC_VERSION_2ARGS(fdim)
MAKE_VEC_VERSION(floor)
MAKE_VEC_VERSION_3ARGS(fma)
MAKE_VEC_VERSION_2ARGS(fmax)
MAKE_VEC_VERSION_WITH_SCALAR(fmax)
MAKE_VEC_VERSION_2ARGS(fmin)
MAKE_VEC_VERSION_WITH_SCALAR(fmin)
MAKE_VEC_VERSION_2ARGS(fmod)

template <typename T, int N>
sycl::vec<T, N> fract(sycl::vec<T, N> a, sycl::vec<T, N>* b) {
  sycl::vec<T, N> res;
  sycl::vec<T, N> resPtr;
  for (int i = 0; i < N; i++) {
    T value;
    res[i] = reference::fract(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

template <typename T, int N>
sycl::vec<T, N> frexp(sycl::vec<T, N> a, sycl::vec<int, N>* b) {
  sycl::vec<T, N> res;
  sycl::vec<int, N> resPtr;
  for (int i = 0; i < N; i++) {
    int value;
    res[i] = reference::frexp(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

MAKE_VEC_VERSION_2ARGS(hypot)

template <typename T, int N>
sycl::vec<int, N> ilogb(sycl::vec<T, N> a) {
  sycl::vec<int, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = reference::ilogb(a[i]);
  }
  return res;
}

template <typename T, int N>
sycl::vec<T, N> ldexp(sycl::vec<T, N> a, sycl::vec<int, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = reference::ldexp(a[i], b[i]);
  }
  return res;
}
template <typename T, int N>
sycl::vec<T, N> ldexp(sycl::vec<T, N> a, int b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = reference::ldexp(a[i], b);
  }
  return res;
}

MAKE_VEC_VERSION(lgamma)

template <typename T, int N>
sycl::vec<T, N> lgamma_r(sycl::vec<T, N> a, sycl::vec<int, N>* b) {
  sycl::vec<T, N> res;
  sycl::vec<int, N> resPtr;
  for (int i = 0; i < N; i++) {
    int value;
    res[i] = reference::lgamma_r(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

MAKE_VEC_VERSION(log)
MAKE_VEC_VERSION(log2)
MAKE_VEC_VERSION(log10)
MAKE_VEC_VERSION(log1p)
MAKE_VEC_VERSION(logb)

MAKE_VEC_VERSION_3ARGS(mad)
MAKE_VEC_VERSION_2ARGS(maxmag)
MAKE_VEC_VERSION_2ARGS(minmag)

template <typename T, int N>
sycl::vec<T, N> modf(sycl::vec<T, N> a, sycl::vec<T, N>* b) {
  sycl::vec<T, N> res;
  sycl::vec<T, N> resPtr;
  for (int i = 0; i < N; i++) {
    T value;
    res[i] = reference::modf(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

template <int N>
sycl::vec<float, N> nan(sycl::vec<unsigned int, N> a) {
  return sycl_cts::math::run_func_on_vector<float, unsigned int, N>(
      [](unsigned int x) { return nan(x); }, a);
}

MAKE_VEC_VERSION_2ARGS(nextafter)
MAKE_VEC_VERSION_2ARGS(pow)

template <typename T, int N>
sycl::vec<T, N> pown(sycl::vec<T, N> a, sycl::vec<int, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = reference::pown(a[i], b[i]);
  }
  return res;
}

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> powr(sycl::vec<T, N> a,
                                          sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return reference::powr(x, y); }, a, b);
}

MAKE_VEC_VERSION_2ARGS(remainder)

template <typename T, int N>
sycl::vec<T, N> remquo(sycl::vec<T, N> a, sycl::vec<T, N> b,
                       sycl::vec<int, N>* c) {
  sycl::vec<T, N> res;
  sycl::vec<int, N> resPtr;
  for (int i = 0; i < N; i++) {
    int value;
    res[i] = reference::remquo(a[i], b[i], &value);
    resPtr[i] = value;
  }
  *c = resPtr;
  return res;
}

MAKE_VEC_VERSION(rint)

template <typename T, int N>
sycl::vec<T, N> rootn(sycl::vec<T, N> a, sycl::vec<int, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    res[i] = reference::rootn(a[i], b[i]);
  }
  return res;
}

MAKE_VEC_VERSION(round)
MAKE_VEC_VERSION(rsqrt)
MAKE_VEC_VERSION(sign)

template <typename T, int N>
sycl::vec<T, N> sincos(sycl::vec<T, N> a, sycl::vec<T, N>* b) {
  sycl::vec<T, N> res;
  sycl::vec<T, N> resPtr;
  for (int i = 0; i < N; i++) {
    T value;
    res[i] = reference::sincos(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

MAKE_VEC_VERSION(sin)
MAKE_VEC_VERSION(sinh)
MAKE_VEC_VERSION(sinpi)
MAKE_VEC_VERSION(sqrt)
MAKE_VEC_VERSION_2ARGS(step)
MAKE_VEC_VERSION(tan)
MAKE_VEC_VERSION(tanh)
MAKE_VEC_VERSION(tanpi)
MAKE_VEC_VERSION(tgamma)
MAKE_VEC_VERSION(trunc)
MAKE_VEC_VERSION(recip)
MAKE_VEC_VERSION_2ARGS(divide)

// Geometric functions

template <typename T, int N>
T dot(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  T res = 0;
  for (int i = 0; i < N; i++) res += a[i] * b[i];
  return res;
}

#define MAKE_MARRAY_VERSION(func)                       \
  template <typename T, size_t N>                       \
  sycl::marray<T, N> func(sycl::marray<T, N> a) {       \
    return sycl_cts::math::run_func_on_marray<T, T, N>( \
        [](T x) { return func(x); }, a);                \
  }

#define MAKE_MARRAY_VERSION_2ARGS(func)                                 \
  template <typename T, size_t N>                                       \
  sycl::marray<T, N> func(sycl::marray<T, N> a, sycl::marray<T, N> b) { \
    return sycl_cts::math::run_func_on_marray<T, T, N>(                 \
        [](T x, T y) { return func(x, y); }, a, b);                     \
  }

#define MAKE_MARRAY_VERSION_3ARGS(func)                               \
  template <typename T, size_t N>                                     \
  sycl::marray<T, N> func(sycl::marray<T, N> a, sycl::marray<T, N> b, \
                          sycl::marray<T, N> c) {                     \
    return sycl_cts::math::run_func_on_marray<T, T, N>(               \
        [](T x, T y, T z) { return func(x, y, z); }, a, b, c);        \
  }

#define MAKE_MARRAY_VERSION_WITH_SCALAR(func)           \
  template <typename T, size_t N>                       \
  sycl::marray<T, N> func(sycl::marray<T, N> a, T b) {  \
    return sycl_cts::math::run_func_on_marray<T, T, N>( \
        [](T x, T y) { return func(x, y); }, a, b);     \
  }

// Common functions.

template <typename T, size_t N>
bool any(sycl::marray<T, N> a) {
  for (size_t i = 0; i < N; i++) {
    if (any(a[i]) == 1) return true;
  }
  return false;
}

template <typename T, size_t N>
bool all(sycl::marray<T, N> a) {
  for (size_t i = 0; i < N; i++) {
    if (all(a[i]) == 0) return false;
  }
  return true;
}

MAKE_MARRAY_VERSION_3ARGS(bitselect)

template <typename T, size_t N>
sycl::marray<T, N> select(sycl::marray<T, N> a, sycl::marray<T, N> b,
                          sycl::marray<bool, N> c) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = c[i] ? b[i] : a[i];
  }
  return res;
}

template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> abs(sycl::marray<T, N> a) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x) { return abs(x); }, a);
}

template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> abs_diff(sycl::marray<T, N> a,
                                                 sycl::marray<T, N> b) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y) { return abs_diff(x, y); }, a, b);
}

MAKE_MARRAY_VERSION_2ARGS(add_sat)
MAKE_MARRAY_VERSION_2ARGS(hadd)
MAKE_MARRAY_VERSION_2ARGS(rhadd)

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

MAKE_MARRAY_VERSION(clz)
MAKE_MARRAY_VERSION(ctz)

MAKE_MARRAY_VERSION_3ARGS(mad_sat)

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

MAKE_MARRAY_VERSION_2ARGS(mul_hi)
MAKE_MARRAY_VERSION_3ARGS(mad_hi)
MAKE_MARRAY_VERSION_2ARGS(rotate)
MAKE_MARRAY_VERSION_2ARGS(sub_sat)

template <typename T, size_t N>
sycl::marray<typename upsample_t<T>::type, N> upsample(
    sycl::marray<T, N> a,
    sycl::marray<typename std::make_unsigned<T>::type, N> b) {
  return sycl_cts::math::run_func_on_marray<typename upsample_t<T>::type, T, N>(
      [](T x, T y) { return upsample(x, y); }, a, b);
}

MAKE_MARRAY_VERSION(popcount)

template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> mad24(sycl::marray<T, N> a,
                                              sycl::marray<T, N> b,
                                              sycl::marray<T, N> c) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y, T z) { return mad24(x, y, z); }, a, b, c);
}

template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> mul24(sycl::marray<T, N> a,
                                              sycl::marray<T, N> b) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y) { return mul24(x, y); }, a, b);
}

MAKE_MARRAY_VERSION(degrees)

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

MAKE_MARRAY_VERSION(radians)

template <typename T, size_t N>
sycl::marray<T, N> step(T a, sycl::marray<T, N> b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = step(a, b[i]);
  }
  return res;
}

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

// Math functions

template <typename T, size_t N>
struct higher_accuracy<sycl::marray<T, N>> {
  using type = sycl::marray<typename higher_accuracy<T>::type, N>;
};

MAKE_MARRAY_VERSION(acos)
MAKE_MARRAY_VERSION(acosh)
MAKE_MARRAY_VERSION(acospi)
MAKE_MARRAY_VERSION(asin)
MAKE_MARRAY_VERSION(asinh)
MAKE_MARRAY_VERSION(asinpi)
MAKE_MARRAY_VERSION(atan)
MAKE_MARRAY_VERSION_2ARGS(atan2)
MAKE_MARRAY_VERSION(atanh)
MAKE_MARRAY_VERSION(atanpi)
MAKE_MARRAY_VERSION_2ARGS(atan2pi)
MAKE_MARRAY_VERSION(cbrt)
MAKE_MARRAY_VERSION(ceil)
MAKE_MARRAY_VERSION_2ARGS(copysign)
MAKE_MARRAY_VERSION(cos)
MAKE_MARRAY_VERSION(cosh)
MAKE_MARRAY_VERSION(cospi)
MAKE_MARRAY_VERSION(erfc)
MAKE_MARRAY_VERSION(erf)
MAKE_MARRAY_VERSION(exp)
MAKE_MARRAY_VERSION(exp2)
MAKE_MARRAY_VERSION(exp10)
MAKE_MARRAY_VERSION(expm1)
MAKE_MARRAY_VERSION(fabs)
MAKE_MARRAY_VERSION_2ARGS(fdim)
MAKE_MARRAY_VERSION(floor)
MAKE_MARRAY_VERSION_3ARGS(fma)
MAKE_MARRAY_VERSION_2ARGS(fmax)
MAKE_MARRAY_VERSION_WITH_SCALAR(fmax)
MAKE_MARRAY_VERSION_2ARGS(fmin)
MAKE_MARRAY_VERSION_WITH_SCALAR(fmin)
MAKE_MARRAY_VERSION_2ARGS(fmod)

template <typename T, size_t N>
sycl::marray<T, N> fract(sycl::marray<T, N> a, sycl::marray<T, N>* b) {
  sycl::marray<T, N> res;
  sycl::marray<T, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    T value;
    res[i] = reference::fract(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

template <typename T, size_t N>
sycl::marray<T, N> frexp(sycl::marray<T, N> a, sycl::marray<int, N>* b) {
  sycl::marray<T, N> res;
  sycl::marray<int, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    int value;
    res[i] = reference::frexp(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

MAKE_MARRAY_VERSION_2ARGS(hypot)

template <typename T, size_t N>
sycl::marray<int, N> ilogb(sycl::marray<T, N> a) {
  sycl::marray<int, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = reference::ilogb(a[i]);
  }
  return res;
}

template <typename T, size_t N>
sycl::marray<T, N> ldexp(sycl::marray<T, N> a, sycl::marray<int, N> b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = reference::ldexp(a[i], b[i]);
  }
  return res;
}
template <typename T, size_t N>
sycl::marray<T, N> ldexp(sycl::marray<T, N> a, int b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = reference::ldexp(a[i], b);
  }
  return res;
}

MAKE_MARRAY_VERSION(lgamma)

template <typename T, size_t N>
sycl::marray<T, N> lgamma_r(sycl::marray<T, N> a, sycl::marray<int, N>* b) {
  sycl::marray<T, N> res;
  sycl::marray<int, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    int value;
    res[i] = reference::lgamma_r(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

MAKE_MARRAY_VERSION(log)
MAKE_MARRAY_VERSION(log2)
MAKE_MARRAY_VERSION(log10)
MAKE_MARRAY_VERSION(log1p)
MAKE_MARRAY_VERSION(logb)

MAKE_MARRAY_VERSION_3ARGS(mad)
MAKE_MARRAY_VERSION_2ARGS(maxmag)
MAKE_MARRAY_VERSION_2ARGS(minmag)

template <typename T, size_t N>
sycl::marray<T, N> modf(sycl::marray<T, N> a, sycl::marray<T, N>* b) {
  sycl::marray<T, N> res;
  sycl::marray<T, N> resPtr;
  for (int i = 0; i < N; i++) {
    T value;
    res[i] = reference::modf(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

template <size_t N>
sycl::marray<float, N> nan(sycl::marray<unsigned int, N> a) {
  return sycl_cts::math::run_func_on_marray<float, unsigned int, N>(
      [](unsigned int x) { return nan(x); }, a);
}

MAKE_MARRAY_VERSION_2ARGS(nextafter)
MAKE_MARRAY_VERSION_2ARGS(pow)

template <typename T, size_t N>
sycl::marray<T, N> pown(sycl::marray<T, N> a, sycl::marray<int, N> b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = reference::pown(a[i], b[i]);
  }
  return res;
}

template <typename T, size_t N>
sycl_cts::resultRef<sycl::marray<T, N>> powr(sycl::marray<T, N> a,
                                             sycl::marray<T, N> b) {
  return sycl_cts::math::run_func_on_marray_result_ref<T, N>(
      [](T x, T y) { return reference::powr(x, y); }, a, b);
}

MAKE_MARRAY_VERSION_2ARGS(remainder)

template <typename T, size_t N>
sycl::marray<T, N> remquo(sycl::marray<T, N> a, sycl::marray<T, N> b,
                          sycl::marray<int, N>* c) {
  sycl::marray<T, N> res;
  sycl::marray<int, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    int value;
    res[i] = reference::remquo(a[i], b[i], &value);
    resPtr[i] = value;
  }
  *c = resPtr;
  return res;
}

MAKE_MARRAY_VERSION(rint)

template <typename T, size_t N>
sycl::marray<T, N> rootn(sycl::marray<T, N> a, sycl::marray<int, N> b) {
  sycl::marray<T, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = reference::rootn(a[i], b[i]);
  }
  return res;
}

MAKE_MARRAY_VERSION(round)
MAKE_MARRAY_VERSION(rsqrt)
MAKE_MARRAY_VERSION(sign)

template <typename T, size_t N>
sycl::marray<T, N> sincos(sycl::marray<T, N> a, sycl::marray<T, N>* b) {
  sycl::marray<T, N> res;
  sycl::marray<T, N> resPtr;
  for (size_t i = 0; i < N; i++) {
    T value;
    res[i] = reference::sincos(a[i], &value);
    resPtr[i] = value;
  }
  *b = resPtr;
  return res;
}

MAKE_MARRAY_VERSION(sin)
MAKE_MARRAY_VERSION(sinh)
MAKE_MARRAY_VERSION(sinpi)
MAKE_MARRAY_VERSION(sqrt)
MAKE_MARRAY_VERSION_2ARGS(step)
MAKE_MARRAY_VERSION(tan)
MAKE_MARRAY_VERSION(tanh)
MAKE_MARRAY_VERSION(tanpi)
MAKE_MARRAY_VERSION(tgamma)
MAKE_MARRAY_VERSION(trunc)
MAKE_MARRAY_VERSION(recip)
MAKE_MARRAY_VERSION_2ARGS(divide)

template <typename T, size_t N>
T dot(sycl::marray<T, N> a, sycl::marray<T, N> b) {
  T res = 0;
  for (size_t i = 0; i < N; i++) res += a[i] * b[i];
  return res;
}

// Generic functions over both scalars and vec / marray types.
// These need to be defined last.

template <typename T>
auto length(T p) {
  return reference::sqrt(reference::dot(p, p));
}

template <typename T>
auto distance(T p0, T p1) {
  return reference::length(p0 - p1);
}

template <typename T, int N>
sycl::vec<T, N> normalize(sycl::vec<T, N> a) {
  sycl::vec<T, N> res;
  T len_a = reference::length(a);
  if (len_a == 0) return sycl::vec<T, N>(0);
  for (int i = 0; i < N; i++) res[i] = a[i] / len_a;
  return res;
}

template <typename T, size_t N>
sycl::marray<T, N> normalize(sycl::marray<T, N> a) {
  sycl::marray<T, N> res;
  T len_a = reference::length(a);
  if (len_a == 0) return sycl::marray<T, N>(0);
  for (size_t i = 0; i < N; i++) res[i] = a[i] / len_a;
  return res;
}

template <typename T>
float fast_length(T p0) {
  return reference::sqrt(fast_dot(p0));
}

template <typename T>
float fast_distance(T p0, T p1) {
  return reference::fast_length(p0 - p1);
}

template <typename T>
T fast_normalize(T p0) {
  return p0 * reference::rsqrt(fast_dot(p0));
}

}  // namespace reference

#endif  // __SYCLCTS_UTIL_MATH_REFERENCE_H
