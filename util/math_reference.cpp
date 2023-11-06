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

#include "math_reference.h"
#include "../oclmath/reference_math.h"
#include "stl.h"

#include <cfloat>

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI (3.14159265358979323846264338327950288)
#endif

namespace {

template <typename A, typename B>
void type_punn(const A &from, B &to) {
  static_assert(sizeof(A) == sizeof(B),
                "type punning of incompatible sized types");
  std::memcpy(reinterpret_cast<void *>(&to),
              reinterpret_cast<const void *>(&from), sizeof(A));
}

#define MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))
#define MIN(_a, _b) ((_a) < (_b) ? (_a) : (_b))

}  // namespace

namespace reference {

template <typename T>
T bitselect_t(T x, T y, T z) {
  return (z & y) | (~z & x);
}

template <typename I, typename T>
T bitselect_f_t(T x, T y, T z) {
  I a, b, c;
  type_punn(x, a);
  type_punn(y, b);
  type_punn(z, c);
  I res_t = bitselect_t(a, b, c);
  T res;
  type_punn(res_t, res);
  return res;
}
float bitselect(float a, float b, float c) {
  return bitselect_f_t<int32_t>(a, b, c);
}
double bitselect(double a, double b, double c) {
  return bitselect_f_t<int64_t>(a, b, c);
}
sycl::half bitselect(sycl::half a, sycl::half b, sycl::half c) {
  return bitselect_f_t<int16_t>(a, b, c);
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- DEGREES
 *
 */

template <typename T>
T degrees_t(T a) {
  return a * (180.0 / M_PI);
}

sycl::half degrees(sycl::half a) { return degrees_t(a); }

float degrees(float a) { return degrees_t(a); }

double degrees(double a) { return degrees_t(a); }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- RADIANS
 *
 */

template <typename T>
T radians_t(T a) {
  return a * (M_PI / 180.0);
}

sycl::half radians(sycl::half a) { return radians_t(a); }

float radians(float a) { return radians_t(a); }

double radians(double a) { return radians_t(a); }
/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- STEP
 *
 */

template <typename T>
T step_t(T a, T b) {
  if (b < a) return 0.0;
  return 1.0;
}

sycl::half step(sycl::half a, sycl::half b) { return step_t(a, b); }

float step(float a, float b) { return step_t(a, b); }

double step(double a, double b) { return step_t(a, b); }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- SMOOTHSTEP
 *
 */
template <typename T>
sycl_cts::resultRef<T> smoothstep_t(T a, T b, T c) {
  if (std::isnan(a) || std::isnan(b) || std::isnan(c) || a >= b)
    return sycl_cts::resultRef<T>(T(), true);
  auto t = clamp<T>((c - a) / (b - a), 0, 1).res;
  return t * t * (3 - 2 * t);
}

sycl_cts::resultRef<sycl::half> smoothstep(sycl::half a, sycl::half b,
                                           sycl::half c) {
  return smoothstep_t(a, b, c);
}
sycl_cts::resultRef<float> smoothstep(float a, float b, float c) {
  return smoothstep_t(a, b, c);
}
sycl_cts::resultRef<double> smoothstep(double a, double b, double c) {
  return smoothstep_t(a, b, c);
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- SIGN
 *
 */

template <typename T>
T sign_t(T a) {
  if (std::isnan(a)) return 0.0;
  if (a > T(0)) return 1.0;
  if (a < T(0)) return -1.0;
  if (std::signbit(a)) return -0.0;
  return +0.0;
}

sycl::half sign(sycl::half a) { return sign_t(a); }

float sign(float a) { return sign_t(a); }

double sign(double a) { return sign_t(a); }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAD_SAT
 *
 */
template <typename K, typename T>
T mad_sat_unsigned(T x, T y, T z) {
  K a = static_cast<K>(x) * static_cast<K>(y) + static_cast<K>(z);
  if (a > std::numeric_limits<T>::max()) return std::numeric_limits<T>::max();
  return a;
}

template <typename T>
T mad_sat_unsigned_long(T x, T y, T z) {
  long double a = static_cast<long double>(x) * static_cast<long double>(y) +
                  static_cast<long double>(z);
  if (a > static_cast<double>(std::numeric_limits<T>::max()) * 1.1l)
    return std::numeric_limits<T>::max();
  T mul = x * y;
  if (mul < x || mul < y) return std::numeric_limits<T>::max();
  return add_sat(mul, z);
}

template <typename K, typename T>
T mad_sat_signed(T x, T y, T z) {
  K a = static_cast<K>(x) * static_cast<K>(y) + static_cast<K>(z);
  if (a > std::numeric_limits<T>::max()) return std::numeric_limits<T>::max();
  if (a < std::numeric_limits<T>::min()) return std::numeric_limits<T>::min();
  return a;
}

template <typename T>
T mad_sat_signed_long(T x, T y, T z) {
  long double a = static_cast<long double>(x) * static_cast<long double>(y) +
                  static_cast<long double>(z);
  if (a > static_cast<long double>(std::numeric_limits<T>::max()) * 1.1l)
    return std::numeric_limits<T>::max();
  if (a < static_cast<long double>(std::numeric_limits<T>::min()) * 1.1l)
    return std::numeric_limits<T>::min();

  T mul = x * y;
  if ((x > 0 && y > 0) || (x < 0 && y < 0))
    if (mul > 0 && std::abs(mul) > std::abs(x) && std::abs(mul) > std::abs(y))
      return add_sat(mul, z);
    else if (z < 0 && mul - std::numeric_limits<T>::min() < abs(z))
      return std::numeric_limits<T>::max() + z;
    else
      return std::numeric_limits<T>::max();
  else if (mul < 0 && std::abs(mul) > std::abs(x) &&
           std::abs(mul) > std::abs(y))
    return add_sat(mul, z);
  else if (z > 0 && std::numeric_limits<T>::max() - mul < z)
    return std::numeric_limits<T>::min() + z;
  else
    return std::numeric_limits<T>::min();
}

unsigned char mad_sat(unsigned char a, unsigned char b, unsigned char c) {
  return mad_sat_unsigned<cl_ulong>(a, b, c);
}
unsigned short mad_sat(unsigned short a, unsigned short b, unsigned short c) {
  return mad_sat_unsigned<cl_ulong>(a, b, c);
}
unsigned int mad_sat(unsigned int a, unsigned int b, unsigned int c) {
  return mad_sat_unsigned<cl_ulong>(a, b, c);
}
unsigned long mad_sat(unsigned long a, unsigned long b, unsigned long c) {
  return mad_sat_unsigned_long(a, b, c);
}
unsigned long long mad_sat(unsigned long long a, unsigned long long b,
                           unsigned long long c) {
  return mad_sat_unsigned_long(a, b, c);
}
char mad_sat(char a, char b, char c) {
  return mad_sat_signed<cl_long>(a, b, c);
}
signed char mad_sat(signed char a, signed char b, signed char c) {
  return mad_sat_signed<cl_long>(a, b, c);
}
short mad_sat(short a, short b, short c) {
  return mad_sat_signed<cl_long>(a, b, c);
}
int mad_sat(int a, int b, int c) { return mad_sat_signed<cl_long>(a, b, c); }
long mad_sat(long a, long b, long c) {
  return mad_sat_signed_long<long>(a, b, c);
}
long long mad_sat(long long a, long long b, long long c) {
  return mad_sat_signed_long<long long>(a, b, c);
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MIX
 *
 */
template <typename T>
sycl_cts::resultRef<T> mix_t(T x, T y, T a) {
  if (a >= T(0.0) && a <= T(1.0)) return x + (y - x) * a;
  return sycl_cts::resultRef<T>(T(), true);
}

sycl_cts::resultRef<sycl::half> mix(const sycl::half a, const sycl::half b,
                                    const sycl::half c) {
  return mix_t(a, b, c);
}

sycl_cts::resultRef<float> mix(const float a, const float b, const float c) {
  return mix_t(a, b, c);
}

sycl_cts::resultRef<double> mix(const double a, const double b,
                                const double c) {
  return mix_t(a, b, c);
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MUL_HI
 *
 */
template <typename T>
T mul_hi_unsigned(T x, T y) {
  // All shifts are half the size of T in bits
  size_t shft = sizeof(T) * 4;

  // hi and lo are the upper and lower parts of the result
  // p, q, r and s are the masked and shifted parts of a
  // b, splitting a and b each into two Ts
  // cross 1 and 2 are the crosswise terms of the multiplication
  T hi, lo, p, q, r, s, cross1, cross2;

  // The mask used to get the lower half of a T
  T mask = -1;
  mask >>= shft;

  // Split a and b in two - upper halves in p and q, lower
  // halves in r and s.
  p = x >> shft;
  q = y >> shft;
  r = x & mask;
  s = y & mask;

  lo = r * s;
  hi = p * q;
  cross1 = (p * s);
  cross2 = (q * r);

  lo >>= shft;
  lo += (cross1 & mask) + (cross2 & mask);
  lo >>= shft;
  hi += lo + (cross1 >> shft) + (cross2 >> shft);

  return hi;
}

/**
 * @brief Function to get high sizeof(T)*8 bits of the product of two signed T

   @tparam T signed type of operand
   @param a The first operand of multiply
   @param b The second operand of multiply
   @return T with high sizeof(T)*8 bits of the sizeof(T)*2*8 bits result of the
   multiplication
 */
template <typename T>
T mul_hi_signed(T a, T b) {
  // All shifts are half the size of T in bits
  size_t shft = sizeof(T) * 4;
  using U = std::make_unsigned_t<T>;
  // hi and lo are the upper and lower parts of the result
  // p, q, r and s are the masked and shifted parts of a
  // b, splitting a and b each into two Ts
  // cross 1 and 2 are the crosswise terms of the multiplication
  U hi, lo, p, q, r, s, cross1, cross2;

  // The mask used to get the lower half of a T
  U mask = -1;
  mask >>= shft;

  U a_pos = std::abs(a);
  U b_pos = std::abs(b);

  p = a_pos >> shft;
  q = b_pos >> shft;
  r = a_pos & mask;
  s = b_pos & mask;

  // Compute half products
  lo = r * s;
  hi = p * q;
  cross1 = p * s;
  cross2 = q * r;

  lo >>= shft;
  lo += (cross1 & mask) + (cross2 & mask);
  lo >>= shft;
  hi += lo + (cross1 >> shft) + (cross2 >> shft);

  T result = hi;
  // if result is negative
  if ((a < 0) != (b < 0)) {
    result = ~result;
    // check that the low half is zero to see if we need to carry
    T lo_half = a * b;
    if (0 == lo_half) {
      result += 1;
    }
  }

  return result;
}

unsigned char mul_hi(unsigned char a, unsigned char b) {
  return mul_hi_unsigned(a, b);
}
unsigned short mul_hi(unsigned short a, unsigned short b) {
  return mul_hi_unsigned(a, b);
}
unsigned int mul_hi(unsigned int a, unsigned int b) {
  return mul_hi_unsigned(a, b);
}
unsigned long mul_hi(unsigned long a, unsigned long b) {
  return mul_hi_unsigned(a, b);
}
unsigned long long mul_hi(unsigned long long a, unsigned long long b) {
  return mul_hi_unsigned(a, b);
}
char mul_hi(char a, char b) { return mul_hi_signed(a, b); }
signed char mul_hi(signed char a, signed char b) { return mul_hi_signed(a, b); }
short mul_hi(short a, short b) { return mul_hi_signed(a, b); }
int mul_hi(int a, int b) { return mul_hi_signed(a, b); }
long mul_hi(long a, long b) { return mul_hi_signed(a, b); }
long long mul_hi(long long a, long long b) { return mul_hi_signed(a, b); }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- UPSAMPLE
 *
 */
uint16_t upsample(uint8_t h, uint8_t l) {
  return (uint16_t(h) << 8) | uint16_t(l);
}

uint32_t upsample(uint16_t h, uint16_t l) {
  return (uint32_t(h) << 16) | uint32_t(l);
}

uint64_t upsample(uint32_t h, uint32_t l) {
  return (uint64_t(h) << 32) | uint64_t(l);
}

int16_t upsample(int8_t h, uint8_t l) {
  return (int16_t(h) << 8) | uint16_t(l);
}

int32_t upsample(int16_t h, uint16_t l) {
  return (int32_t(h) << 16) | uint32_t(l);
}

int64_t upsample(int32_t h, uint32_t l) {
  return (int64_t(h) << 32) | uint64_t(l);
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAD24
 * technically undefined in C++17 but defined and valid from C++20
 */
template <typename T>
std::enable_if_t<std::is_signed_v<T>, bool> in_range_24(T v) {
  return v >= -(1 << 23) && v < (1 << 23);
}

template <typename T>
std::enable_if_t<std::is_unsigned_v<T>, bool> in_range_24(T v) {
  return v < (1 << 24);
}

sycl_cts::resultRef<int32_t> mad24(int32_t x, int32_t y, int32_t z) {
  if (!in_range_24(x) || !in_range_24(y))
    return sycl_cts::resultRef<int32_t>(0, true);
  return int32_t(int64_t(x) * int64_t(y) + int64_t(z));
}
sycl_cts::resultRef<uint32_t> mad24(uint32_t x, uint32_t y, uint32_t z) {
  if (!in_range_24(x) || !in_range_24(y))
    return sycl_cts::resultRef<uint32_t>(0, true);
  return uint32_t(uint64_t(x) * uint64_t(y) + uint64_t(z));
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MUL24
 * technically undefined in C++17 but defined and valid from C++20
 */
sycl_cts::resultRef<int32_t> mul24(int32_t x, int32_t y) {
  if (!in_range_24(x) || !in_range_24(y))
    return sycl_cts::resultRef<int32_t>(0, true);
  return int32_t(int64_t(x) * int64_t(y));
}
sycl_cts::resultRef<uint32_t> mul24(uint32_t x, uint32_t y) {
  if (!in_range_24(x) || !in_range_24(y))
    return sycl_cts::resultRef<uint32_t>(0, true);
  return uint32_t(uint64_t(x) * uint64_t(y));
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MATH
 *
 */

sycl::half acospi(sycl::half a) { return reference_acospi(a); }
float acospi(float a) { return reference_acospi(a); }
double acospi(double a) { return reference_acospil(a); }

sycl::half asinpi(sycl::half a) { return reference_asinpi(a); }
float asinpi(float a) { return reference_asinpi(a); }
double asinpi(double a) { return reference_asinpil(a); }

sycl::half atanpi(sycl::half a) { return reference_atanpi(a); }
float atanpi(float a) { return reference_atanpi(a); }
double atanpi(double a) { return reference_atanpil(a); }

sycl::half atan2pi(sycl::half a, sycl::half b) {
  return reference_atan2pi(a, b);
}
float atan2pi(float a, float b) { return reference_atan2pi(a, b); }
double atan2pi(double a, double b) { return reference_atan2pil(a, b); }

sycl::half cospi(sycl::half a) { return reference_cospi(a); }
float cospi(float a) { return reference_cospi(a); }
double cospi(double a) { return reference_cospil(a); }

sycl::half fma(sycl::half a, sycl::half b, sycl::half c) {
  return reference_fma(a, b, c, 0);
}
float fma(float a, float b, float c) { return reference_fma(a, b, c, 0); }
double fma(double a, double b, double c) { return reference_fmal(a, b, c); }

// hipSYCL does not yet support sycl::bit_cast, which is used in `nextafter`.
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
sycl::half fdim(sycl::half a, sycl::half b) {
  if (a > b) {
    // to get rounding to nearest even
    double resd = static_cast<double>(a) - static_cast<double>(b);
    sycl::half res = static_cast<sycl::half>(resd);
    double diff = resd - static_cast<double>(res);
    sycl::half next = nextafter(res, static_cast<sycl::half>(DBL_MAX * diff));
    if (static_cast<double>(next) - resd == diff) {
      int16_t rep;
      type_punn(next, rep);
      if (rep % 2 == 0) return next;
    }
    return res;
  }
  return +0;
}
#endif

sycl::half fract(sycl::half a, sycl::half *b) {
  *b = std::floor(a);
  return std::fmin(a - *b, nextafter(sycl::half(1.0), sycl::half(0.0)));
}
float fract(float a, float *b) {
  *b = std::floor(a);
  return std::fmin(a - *b, nextafter(1.0f, 0.0f));
}
double fract(double a, double *b) {
  *b = std::floor(a);
  return std::fmin(a - *b, nextafter(1.0, 0.0));
}

float nan(unsigned int a) { return std::nanf(std::to_string(a).c_str()); }
double nan(unsigned long a) { return std::nan(std::to_string(a).c_str()); }
double nan(unsigned long long a) { return std::nan(std::to_string(a).c_str()); }

sycl::half modf(sycl::half a, sycl::half *b) {
  float resPtr;
  float res = modf(static_cast<float>(a), &resPtr);
  *b = static_cast<sycl::half>(resPtr);
  return res;
}

// hipSYCL does not yet support sycl::bit_cast
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
sycl::half nextafter(sycl::half x, sycl::half y) {
  if (std::isnan(x)) return x;

  if (std::isnan(y)) return y;

  if (x == y) return y;

  // Transform the signed binary numbers represented as a leading sign bit
  // and 15 bit unsigned value into a 2-complement 16 bit signed integer
  int16_t a = sycl::bit_cast<int16_t>(x);
  int16_t b = sycl::bit_cast<int16_t>(y);

  // 0x8000 for leading 1 check
  if (a & 0x8000) a = 0x8000 - a;
  if (b & 0x8000) b = 0x8000 - b;

  // Increment a towards the direction of b
  a += (a < b) ? 1 : -1;

  // Convert again a from 2-complement signed value
  // into sign bit + unsigned value
  a = (a < 0) ? (int16_t)0x8000 - a : a;

  return sycl::bit_cast<sycl::half>(a);
}
#endif

sycl::half sinpi(sycl::half a) { return reference_sinpi(a); }
float sinpi(float a) { return reference_sinpi(a); }
double sinpi(double a) { return reference_sinpil(a); }

sycl::half tanpi(sycl::half a) { return reference_tanpi(a); }
float tanpi(float a) { return reference_tanpi(a); }
double tanpi(double a) { return reference_tanpil(a); }

// Geometric functions

template <typename T, int N>
sycl::vec<T, N> cross_t(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  sycl::vec<T, N> res;
  std::vector<T> temp_res(4);
  std::vector<T> av({a.x(), a.y(), a.z()});
  std::vector<T> bv({b.x(), b.y(), b.z()});
  temp_res[0] = av[1] * bv[2] - av[2] * bv[1];
  temp_res[1] = av[2] * bv[0] - av[0] * bv[2];
  temp_res[2] = av[0] * bv[1] - av[1] * bv[0];
  temp_res[3] = 0.0;
  for (int i = 0; i < N; i++) res[i] = temp_res[i];

  return res;
}

sycl::float4 cross(sycl::float4 p0, sycl::float4 p1) { return cross_t(p0, p1); }
sycl::float3 cross(sycl::float3 p0, sycl::float3 p1) { return cross_t(p0, p1); }
sycl::double4 cross(sycl::double4 p0, sycl::double4 p1) {
  return cross_t(p0, p1);
}
sycl::double3 cross(sycl::double3 p0, sycl::double3 p1) {
  return cross_t(p0, p1);
}

// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N>
sycl::marray<T, N> cross_t(sycl::marray<T, N> a, sycl::marray<T, N> b) {
  sycl::marray<T, N> res;
  std::vector<T> temp_res(4);
  std::vector<T> av({a[0], a[1], a[2]});
  std::vector<T> bv({b[0], b[1], b[2]});
  temp_res[0] = av[1] * bv[2] - av[2] * bv[1];
  temp_res[1] = av[2] * bv[0] - av[0] * bv[2];
  temp_res[2] = av[0] * bv[1] - av[1] * bv[0];
  temp_res[3] = 0.0;
  for (size_t i = 0; i < N; i++) res[i] = temp_res[i];
  return res;
}

sycl::mfloat4 cross(sycl::mfloat4 p0, sycl::mfloat4 p1) {
  return cross_t(p0, p1);
}
sycl::mfloat3 cross(sycl::mfloat3 p0, sycl::mfloat3 p1) {
  return cross_t(p0, p1);
}
sycl::mdouble4 cross(sycl::mdouble4 p0, sycl::mdouble4 p1) {
  return cross_t(p0, p1);
}
sycl::mdouble3 cross(sycl::mdouble3 p0, sycl::mdouble3 p1) {
  return cross_t(p0, p1);
}
#endif  // SYCL_CTS_COMPILING_WITH_HIPSYCL

sycl::half fast_dot(float p0) { return std::pow(p0, 2); }
sycl::half fast_dot(sycl::float2 p0) {
  return std::pow(p0.x(), 2) + std::pow(p0.y(), 2);
}
sycl::half fast_dot(sycl::float3 p0) {
  return std::pow(p0.x(), 2) + std::pow(p0.y(), 2) + std::pow(p0.z(), 2);
}
sycl::half fast_dot(sycl::float4 p0) {
  return std::pow(p0.x(), 2) + std::pow(p0.y(), 2) + std::pow(p0.z(), 2) +
         std::pow(p0.w(), 2);
}
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
sycl::half fast_dot(sycl::mfloat2 p0) {
  return std::pow(p0[0], 2) + std::pow(p0[1], 2);
}
sycl::half fast_dot(sycl::mfloat3 p0) {
  return std::pow(p0[0], 2) + std::pow(p0[1], 2) + std::pow(p0[2], 2);
}
sycl::half fast_dot(sycl::mfloat4 p0) {
  return std::pow(p0[0], 2) + std::pow(p0[1], 2) + std::pow(p0[2], 2) +
         std::pow(p0[3], 2);
}
#endif

} /* namespace reference */
