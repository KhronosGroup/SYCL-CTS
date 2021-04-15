/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "math_reference.h"
#include "stl.h"
#include "../oclmath/reference_math.h"

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

} /* namespace {} */

namespace reference {

template <typename T> T bitselect_t(T x, T y, T z) {
  return (z & y) | (~z & x);
}

template <typename I, typename T> T bitselect_f_t(T x, T y, T z) {
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
cl::sycl::half bitselect(cl::sycl::half a, cl::sycl::half b, cl::sycl::half c) {
  return bitselect_f_t<int16_t>(a, b, c);
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- DEGREES
 *
 */

template <typename T> T degrees_t(T a) { return a * (180.0L / M_PI); }

float degrees(float a) { return degrees_t(a); }

double degrees(double a) { return degrees_t(a); }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- RADIANS
 *
 */

template <typename T> T radians_t(T a) { return a * (M_PI / 180.0L); }

float radians(float a) { return radians_t(a); }

double radians(double a) { return radians_t(a); }
/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- STEP
 *
 */

template <typename T> T step_t(T a, T b) {
  if (b < a)
    return 0.0;
  return 1.0;
}

float step(float a, float b) { return step_t(a, b); }

double step(double a, double b) { return step_t(a, b); }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- SMOOTHSTEP
 *
 */
template <typename T> sycl_cts::resultRef<T> smoothstep_t(T a, T b, T c) {
  if (std::isnan(a) || std::isnan(b) || std::isnan(c) || a >= b)
    return sycl_cts::resultRef<T>(T(), true);
  if (c <= a)
    return 0.0;
  if (c >= b)
    return 1.0;
  auto t = clamp<T>((c - a) / (b - a), 0, 1).res;
  return t * t * (3 - 2 * t);
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

template <typename T> T sign_t(T a) {
  if (std::isnan(a))
    return 0.0;
  if (a > 0)
    return 1.0;
  if (a < 0)
    return -1.0;
  if (signbit(a))
    return -0.0;
  return +0.0;
}

float sign(float a) { return sign_t(a); }

double sign(double a) { return sign_t(a); }

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MAD_SAT
 *
 */
template <typename K, typename T> T mad_sat_unsigned(T x, T y, T z) {
  K a = static_cast<K>(x) * static_cast<K>(y) + static_cast<K>(z);
  if (a > std::numeric_limits<T>::max())
    return std::numeric_limits<T>::max();
  return a;
}

template <typename T> T mad_sat_unsigned_long(T x, T y, T z) {
  long double a = static_cast<long double>(x) * static_cast<long double>(y) +
                  static_cast<long double>(z);
  if (a > static_cast<double>(std::numeric_limits<T>::max()) * 1.1l)
    return std::numeric_limits<T>::max();
  T mul = x * y;
  if (mul < x || mul < y)
    return std::numeric_limits<T>::max();
  return add_sat(mul, z);
}

template <typename K, typename T> T mad_sat_signed(T x, T y, T z) {
  K a = static_cast<K>(x) * static_cast<K>(y) + static_cast<K>(z);
  if (a > std::numeric_limits<T>::max())
    return std::numeric_limits<T>::max();
  if (a < std::numeric_limits<T>::min())
    return std::numeric_limits<T>::min();
  return a;
}

template <typename T> T mad_sat_signed_long(T x, T y, T z) {
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
  return mad_sat_unsigned<cl::sycl::cl_ulong>(a, b, c);
}
unsigned short mad_sat(unsigned short a, unsigned short b, unsigned short c) {
  return mad_sat_unsigned<cl::sycl::cl_ulong>(a, b, c);
}
unsigned int mad_sat(unsigned int a, unsigned int b, unsigned int c) {
  return mad_sat_unsigned<cl::sycl::cl_ulong>(a, b, c);
}
unsigned long mad_sat(unsigned long a, unsigned long b, unsigned long c) {
  return mad_sat_unsigned_long(a, b, c);
}
unsigned long long mad_sat(unsigned long long a, unsigned long long b,
                           unsigned long long c) {
  return mad_sat_unsigned_long(a, b, c);
}
char mad_sat(char a, char b, char c) {
  return mad_sat_signed<cl::sycl::cl_long>(a, b, c);
}
signed char mad_sat(signed char a, signed char b, signed char c) {
  return mad_sat_signed<cl::sycl::cl_long>(a, b, c);
}
short mad_sat(short a, short b, short c) {
  return mad_sat_signed<cl::sycl::cl_long>(a, b, c);
}
int mad_sat(int a, int b, int c) {
  return mad_sat_signed<cl::sycl::cl_long>(a, b, c);
}
long mad_sat(long a, long b, long c) {
  return mad_sat_signed_long<long>(a, b, c);
}
long long mad_sat(long long a, long long b, long long c) {
  return mad_sat_signed_long<long long>(a, b, c);
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MIX
 *
 */
template <typename T> sycl_cts::resultRef<T> mix_t(T x, T y, T a) {
  if (a >= 0.0 && a <= 1.0)
    return x + (y - x) * a;
  return sycl_cts::resultRef<T>(T(), true);
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
template <typename T> T mul_hi_unsigned(T x, T y) {
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

template <typename T> T mul_hi_signed(T a, T b) {
  // All shifts are half the size of T in bits
  size_t shft = sizeof(T) * 4;
  typedef typename std::make_unsigned<T>::type U;
  // hi and lo are the upper and lower parts of the result
  // p, q, r and s are the masked and shifted parts of a
  // b, splitting a and b each into two Ts
  // cross 1 and 2 are the crosswise terms of the multiplication
  U hi, lo, p, q, r, s, cross1, cross2;

  // The mask used to get the lower half of a T
  U mask = -1;
  mask >>= shft;

  size_t msb = sizeof(T) * 8 - 1;

  // a and b rendered positive
  auto a_pos = (a & (1ull << msb)) ? (~a + 1) : a;
  auto b_pos = (b & (1ull << msb)) ? (~b + 1) : b;

  p = static_cast<U>(a_pos) >> shft;
  q = static_cast<U>(b_pos) >> shft;
  r = static_cast<U>(a_pos) & mask;
  s = static_cast<U>(b_pos) & mask;

  lo = r * s;
  hi = p * q;
  cross1 = (p * s);
  cross2 = (q * r);

  lo >>= shft;
  lo += (cross1 & mask) + (cross2 & mask);
  lo >>= shft;
  hi += lo + (cross1 >> shft) + (cross2 >> shft);

  return (a >> msb) ^ (b >> msb) ? static_cast<T>(~hi) : hi;
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
 *
 */
template <typename T> bool in_signed_range_24(T v) {
  return v >= -223 && v <= 222;
}

sycl_cts::resultRef<int32_t> mad24(int32_t x, int32_t y, int32_t z) {
  if (!in_signed_range_24(x) || !in_signed_range_24(y) ||
      !in_signed_range_24(z))
    return sycl_cts::resultRef<int32_t>(0, true);
  return int32_t(int64_t(x) * int64_t(y) + int64_t(z));
}
sycl_cts::resultRef<uint32_t> mad24(uint32_t x, uint32_t y, uint32_t z) {
  if (x > 223 || y > 223 || z > 223)
    return sycl_cts::resultRef<uint32_t>(0, true);
  return uint32_t(uint64_t(x) * uint64_t(y) + uint64_t(z));
}


/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MUL24
 *
 */
sycl_cts::resultRef<int32_t> mul24(int32_t x, int32_t y) {
  if (!in_signed_range_24(x) || !in_signed_range_24(y))
    return sycl_cts::resultRef<int32_t>(0, true);
  return int32_t(int64_t(x) * int64_t(y));
}
sycl_cts::resultRef<uint32_t> mul24(uint32_t x, uint32_t y) {
  if (x > 223 || y > 223)
    return sycl_cts::resultRef<uint32_t>(0, true);
  return uint32_t(uint64_t(x) * uint64_t(y));
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- MATH
 *
 */
cl::sycl::half acos(cl::sycl::half a) {
  return std::acos(static_cast<float>(a));
}
float acos(float a) { return std::acos(static_cast<double>(a)); }
double acos(double a) { return std::acos(static_cast<long double>(a)); }

cl::sycl::half acosh(cl::sycl::half a) {
  return std::acosh(static_cast<float>(a));
}
float acosh(float a) { return std::acosh(static_cast<double>(a)); }
double acosh(double a) { return std::acosh(static_cast<long double>(a)); }

cl::sycl::half acospi(cl::sycl::half a) { return reference_acospi(a); }
float acospi(float a) { return reference_acospi(a); }
double acospi(double a) { return reference_acospil(a); }

cl::sycl::half asin(cl::sycl::half a) {
  return std::asin(static_cast<float>(a));
}
float asin(float a) { return std::asin(static_cast<double>(a)); }
double asin(double a) { return std::asin(static_cast<long double>(a)); }

cl::sycl::half asinh(cl::sycl::half a) {
  return std::asinh(static_cast<float>(a));
}
float asinh(float a) { return std::asinh(static_cast<double>(a)); }
double asinh(double a) { return std::asinh(static_cast<long double>(a)); }

cl::sycl::half asinpi(cl::sycl::half a) { return reference_asinpi(a); }
float asinpi(float a) { return reference_asinpi(a); }
double asinpi(double a) { return reference_asinpil(a); }

cl::sycl::half atan(cl::sycl::half a) {
  return std::atan(static_cast<float>(a));
}
float atan(float a) { return std::atan(static_cast<double>(a)); }
double atan(double a) { return std::atan(static_cast<long double>(a)); }

cl::sycl::half atan2(cl::sycl::half a, cl::sycl::half b) {
  return std::atan2(static_cast<float>(a), b);
}
float atan2(float a, float b) { return std::atan2(static_cast<double>(a), b); }
double atan2(double a, double b) {
  return std::atan2(static_cast<long double>(a), b);
}

cl::sycl::half atanh(cl::sycl::half a) {
  return std::atanh(static_cast<float>(a));
}
float atanh(float a) { return std::atanh(static_cast<double>(a)); }
double atanh(double a) { return std::atanh(static_cast<long double>(a)); }

cl::sycl::half atanpi(cl::sycl::half a) { return reference_atanpi(a); }
float atanpi(float a) { return reference_atanpi(a); }
double atanpi(double a) { return reference_atanpil(a); }

cl::sycl::half atan2pi(cl::sycl::half a, cl::sycl::half b) {
  return reference_atan2pi(a, b);
}
float atan2pi(float a, float b) { return reference_atan2pi(a, b); }
double atan2pi(double a, double b) { return reference_atan2pil(a, b); }

cl::sycl::half cbrt(cl::sycl::half a) {
  return std::cbrt(static_cast<float>(a));
}
float cbrt(float a) { return std::cbrt(static_cast<double>(a)); }
double cbrt(double a) { return std::cbrt(static_cast<long double>(a)); }

cl::sycl::half cos(cl::sycl::half a) { return std::cos(static_cast<float>(a)); }
float cos(float a) { return std::cos(static_cast<double>(a)); }
double cos(double a) { return std::cos(static_cast<long double>(a)); }

cl::sycl::half cosh(cl::sycl::half a) {
  return std::cosh(static_cast<float>(a));
}
float cosh(float a) { return std::cosh(static_cast<double>(a)); }
double cosh(double a) { return std::cosh(static_cast<long double>(a)); }

cl::sycl::half cospi(cl::sycl::half a) { return reference_cospi(a); }
float cospi(float a) { return reference_cospi(a); }
double cospi(double a) { return reference_cospil(a); }

cl::sycl::half erfc(cl::sycl::half a) {
  return std::erfc(static_cast<float>(a));
}
float erfc(float a) { return std::erfc(static_cast<double>(a)); }
double erfc(double a) { return std::erfc(static_cast<long double>(a)); }

cl::sycl::half erf(cl::sycl::half a) { return std::erf(static_cast<float>(a)); }
float erf(float a) { return std::erf(static_cast<double>(a)); }
double erf(double a) { return std::erf(static_cast<long double>(a)); }

cl::sycl::half exp(cl::sycl::half a) { return std::exp(static_cast<float>(a)); }
float exp(float a) { return std::exp(static_cast<double>(a)); }
double exp(double a) { return std::exp(static_cast<long double>(a)); }

cl::sycl::half exp2(cl::sycl::half a) {
  return std::exp2(static_cast<float>(a));
}
float exp2(float a) { return std::exp2(static_cast<double>(a)); }
double exp2(double a) { return std::exp2(static_cast<long double>(a)); }

template <typename T> T exp10_t(T a) { return std::pow(static_cast<T>(10), a); }
cl::sycl::half exp10(cl::sycl::half a) { return exp10_t<float>(a); }
float exp10(float a) { return exp10_t<double>(a); }
double exp10(double a) { return exp10_t<long double>(a); }

cl::sycl::half expm1(cl::sycl::half a) {
  return std::expm1(static_cast<float>(a));
}
float expm1(float a) { return std::expm1(static_cast<double>(a)); }
double expm1(double a) { return std::expm1(static_cast<long double>(a)); }

cl::sycl::half fma(cl::sycl::half a, cl::sycl::half b, cl::sycl::half c) {
  return reference_fma(a, b, c, 0);
}
float fma(float a, float b, float c) { return reference_fma(a, b, c, 0); }
double fma(double a, double b, double c) { return reference_fmal(a, b, c); }

cl::sycl::half fdim(cl::sycl::half a, cl::sycl::half b) {
  if (a > b) {
    // to get rounding to nearest even
    double resd = static_cast<double>(a) - static_cast<double>(b);
    cl::sycl::half res = static_cast<cl::sycl::half>(resd);
    double diff = resd - static_cast<double>(res);
    cl::sycl::half next =
        nextafter(res, static_cast<cl::sycl::half>(DBL_MAX * diff));
    if (static_cast<double>(next) - resd == diff) {
      int16_t rep;
      type_punn(next, rep);
      if (rep % 2 == 0)
        return next;
    }
    return res;
  }
  return +0;
}

template <typename T> T fract_t(T a, T *b) {
  *b = std::floor(a);
  return std::fmin(a - *b, nextafter(T(1.0), T(0.0)));
}
cl::sycl::half fract(cl::sycl::half a, cl::sycl::half *b) {
  return fract_t(a, b);
}
float fract(float a, float *b) { return fract_t(a, b); }
double fract(double a, double *b) { return fract_t(a, b); }

cl::sycl::half hypot(cl::sycl::half a, cl::sycl::half b) {
  return std::hypot(static_cast<float>(a), b);
}
float hypot(float a, float b) { return std::hypot(static_cast<double>(a), b); }
double hypot(double a, double b) {
  return std::hypot(static_cast<long double>(a), b);
}

template <typename T> T lgamma_r_t(T a, int *b) {
  *b = (std::tgamma(a) > 0) ? 1 : -1;
  return std::lgamma(a);
}
cl::sycl::half lgamma_r(cl::sycl::half a, int *b) { return lgamma_r_t(a, b); }
float lgamma_r(float a, int *b) { return lgamma_r_t(a, b); }
double lgamma_r(double a, int *b) { return lgamma_r_t(a, b); }

cl::sycl::half log(cl::sycl::half a) { return std::log(static_cast<float>(a)); }
float log(float a) { return std::log(static_cast<double>(a)); }
double log(double a) { return std::log(static_cast<long double>(a)); }

cl::sycl::half log2(cl::sycl::half a) {
  return std::log2(static_cast<float>(a));
}
float log2(float a) { return std::log2(static_cast<double>(a)); }
double log2(double a) { return std::log2(static_cast<long double>(a)); }

cl::sycl::half log10(cl::sycl::half a) {
  return std::log10(static_cast<float>(a));
}
float log10(float a) { return std::log10(static_cast<double>(a)); }
double log10(double a) { return std::log10(static_cast<long double>(a)); }

cl::sycl::half log1p(cl::sycl::half a) {
  return std::log1p(static_cast<float>(a));
}
float log1p(float a) { return std::log1p(static_cast<double>(a)); }
double log1p(double a) { return std::log1p(static_cast<long double>(a)); }

template <typename T> T mad_t(T a, T b, T c) { return a * b + c; }
cl::sycl::half mad(cl::sycl::half a, cl::sycl::half b, cl::sycl::half c) {
  return mad_t(a, b, c);
}
float mad(float a, float b, float c) { return mad_t(a, b, c); }
double mad(double a, double b, double c) { return mad_t(a, b, c); }

template <typename T> T maxmag_t(T a, T b) {
  if (fabs(a) > fabs(b))
    return a;
  else if (fabs(b) > fabs(a))
    return b;
  return fmax(a, b);
}
cl::sycl::half maxmag(cl::sycl::half a, cl::sycl::half b) {
  return maxmag_t(a, b);
}
float maxmag(float a, float b) { return maxmag_t(a, b); }
double maxmag(double a, double b) { return maxmag_t(a, b); }

template <typename T> T minmag_t(T a, T b) {
  if (fabs(a) < fabs(b))
    return a;
  else if (fabs(b) < fabs(a))
    return b;
  return fmin(a, b);
}
cl::sycl::half minmag(cl::sycl::half a, cl::sycl::half b) {
  return minmag_t(a, b);
}
float minmag(float a, float b) { return minmag_t(a, b); }
double minmag(double a, double b) { return minmag_t(a, b); }

float nan(unsigned int a) { return std::nanf(std::to_string(a).c_str()); }
double nan(unsigned long a) { return std::nan(std::to_string(a).c_str()); }
double nan(unsigned long long a) { return std::nan(std::to_string(a).c_str()); }

cl::sycl::half modf(cl::sycl::half a, cl::sycl::half *b) {
  float resPtr;
  float res = modf(static_cast<float>(a), &resPtr);
  *b = static_cast<cl::sycl::half>(resPtr);
  return res;
}

cl::sycl::half nextafter(cl::sycl::half x, cl::sycl::half y) {
  if (std::isnan(x))
    return x;

  if (std::isnan(y))
    return y;

  if (x == y)
    return y;

  // Transform the signed binary numbers represented as a leading sign bit
  // and 15 bit unsigned value into a 2-complement 16 bit signed integer
  int16_t a = sycl::bit_cast<int16_t>(x);
  int16_t b = sycl::bit_cast<int16_t>(y);

  // 0x8000 for leading 1 check
  if (a & 0x8000)
    a = 0x8000 - a;
  if (b & 0x8000)
    b = 0x8000 - b;

  // Increment a towards the direction of b
  a += (a < b) ? 1 : -1;

  // Convert again a from 2-complement signed value
  // into sign bit + unsigned value
  a = (a < 0) ? (int16_t)0x8000 - a : a;

  return sycl::bit_cast<cl::sycl::half>(a);
}

cl::sycl::half pow(cl::sycl::half a, cl::sycl::half b) {
  return std::pow(static_cast<float>(a), b);
}
float pow(float a, float b) { return std::pow(static_cast<double>(a), b); }
double pow(double a, double b) {
  return std::pow(static_cast<long double>(a), b);
}

template <typename T> T pown_t(T a, int b) { return std::pow(a, b); }
cl::sycl::half pown(cl::sycl::half a, int b) { return pown_t<float>(a, b); }
float pown(float a, int b) { return pown_t<double>(a, b); }
double pown(double a, int b) { return pown_t<long double>(a, b); }

template <typename T> sycl_cts::resultRef<T> powr_t(T a, T b) {
  if (a < 0)
    return sycl_cts::resultRef<T>(T(), true);
  return std::pow(a, b);
}
sycl_cts::resultRef<cl::sycl::half> powr(cl::sycl::half a, cl::sycl::half b) {
  return powr_t<float>(a, b);
}
sycl_cts::resultRef<float> powr(float a, float b) {
  return powr_t<double>(a, b);
}
sycl_cts::resultRef<double> powr(double a, double b) {
  return powr_t<long double>(a, b);
}

template <typename T> T rootn_t(T a, int b) {
  return std::pow(a, static_cast<T>(1.0 / b));
}
cl::sycl::half rootn(cl::sycl::half a, int b) { return rootn_t<float>(a, b); }
float rootn(float a, int b) { return rootn_t<double>(a, b); }
double rootn(double a, int b) { return rootn_t<long double>(a, b); }

template <typename T> T rsqrt_t(T a) { return 1 / std::sqrt(a); }
cl::sycl::half rsqrt(cl::sycl::half a) { return rsqrt_t<float>(a); }
float rsqrt(float a) { return rsqrt_t<double>(a); }
double rsqrt(double a) { return rsqrt_t<long double>(a); }

template <typename T> T sincos_t(T a, T *b) {
  *b = std::cos(a);
  return std::sin(a);
}
cl::sycl::half sincos(cl::sycl::half a, cl::sycl::half *b) {
  return sincos_t(a, b);
}
float sincos(float a, float *b) { return sincos_t(a, b); }
double sincos(double a, double *b) { return sincos_t(a, b); }

cl::sycl::half sin(cl::sycl::half a) { return std::sin(static_cast<float>(a)); }
float sin(float a) { return std::sin(static_cast<double>(a)); }
double sin(double a) { return std::sin(static_cast<long double>(a)); }

cl::sycl::half sinh(cl::sycl::half a) {
  return std::sinh(static_cast<float>(a));
}
float sinh(float a) { return std::sinh(static_cast<double>(a)); }
double sinh(double a) { return std::sinh(static_cast<long double>(a)); }

cl::sycl::half sinpi(cl::sycl::half a) { return reference_sinpi(a); }
float sinpi(float a) { return reference_sinpi(a); }
double sinpi(double a) { return reference_sinpil(a); }

cl::sycl::half sqrt(cl::sycl::half a) {
  return std::sqrt(static_cast<float>(a));
}
float sqrt(float a) { return std::sqrt(static_cast<double>(a)); }
double sqrt(double a) { return std::sqrt(static_cast<long double>(a)); }

cl::sycl::half tan(cl::sycl::half a) { return std::tan(static_cast<float>(a)); }
float tan(float a) { return std::tan(static_cast<double>(a)); }
double tan(double a) { return std::tan(static_cast<long double>(a)); }

cl::sycl::half tanh(cl::sycl::half a) {
  return std::tanh(static_cast<float>(a));
}
float tanh(float a) { return std::tanh(static_cast<double>(a)); }
double tanh(double a) { return std::tanh(static_cast<long double>(a)); }

cl::sycl::half tanpi(cl::sycl::half a) { return reference_tanpi(a); }
float tanpi(float a) { return reference_tanpi(a); }
double tanpi(double a) { return reference_tanpil(a); }

cl::sycl::half tgamma(cl::sycl::half a) {
  return std::tgamma(static_cast<float>(a));
}
float tgamma(float a) { return std::tgamma(static_cast<double>(a)); }
double tgamma(double a) { return std::tgamma(static_cast<long double>(a)); }

template <typename T> T recip_t(T a) { return 1.0 / a; }
cl::sycl::half recip(cl::sycl::half a) { return recip_t(a); }
float recip(float a) { return recip_t(a); }
double recip(double a) { return recip_t(a); }

template <typename T> T divide_t(T a, T b) { return a / b; }
cl::sycl::half divide(cl::sycl::half a, cl::sycl::half b) {
  return divide_t(a, b);
}
float divide(float a, float b) { return divide_t(a, b); }
double divide(double a, double b) { return divide_t(a, b); }

// Geometric functions

template <typename T, int N>
cl::sycl::vec<T, N> cross_t(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  cl::sycl::vec<T, N> res;
  std::vector<T> temp_res(4);
  std::vector<T> av({a.x(), a.y(), a.z()});
  std::vector<T> bv({b.x(), b.y(), b.z()});
  temp_res[0] = av[1] * bv[2] - av[2] * bv[1];
  temp_res[1] = av[2] * bv[0] - av[0] * bv[2];
  temp_res[2] = av[0] * bv[1] - av[1] * bv[0];
  temp_res[3] = 0.0;
  for (int i = 0; i < N; i++)
    setElement<T, N>(res, i, temp_res[i]);
  return res;
}

cl::sycl::float4 cross(cl::sycl::float4 p0, cl::sycl::float4 p1) {
  return cross_t(p0, p1);
}
cl::sycl::float3 cross(cl::sycl::float3 p0, cl::sycl::float3 p1) {
  return cross_t(p0, p1);
}
cl::sycl::double4 cross(cl::sycl::double4 p0, cl::sycl::double4 p1) {
  return cross_t(p0, p1);
}
cl::sycl::double3 cross(cl::sycl::double3 p0, cl::sycl::double3 p1) {
  return cross_t(p0, p1);
}

template <typename T, int N>
T dot_t(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  T res = 0;
  for (int i = 0; i < N; i++)
    res += getElement<T, N>(a, i) * getElement<T, N>(b, i);
  return res;
}

float dot(float p0, float p1) { return p0 * p1; }
float dot(cl::sycl::float2 p0, cl::sycl::float2 p1) { return dot_t(p0, p1); }
float dot(cl::sycl::float3 p0, cl::sycl::float3 p1) { return dot_t(p0, p1); }
float dot(cl::sycl::float4 p0, cl::sycl::float4 p1) { return dot_t(p0, p1); }
double dot(double p0, double p1) { return p0 * p1; }
double dot(cl::sycl::double2 p0, cl::sycl::double2 p1) { return dot_t(p0, p1); }
double dot(cl::sycl::double3 p0, cl::sycl::double3 p1) { return dot_t(p0, p1); }
double dot(cl::sycl::double4 p0, cl::sycl::double4 p1) { return dot_t(p0, p1); }

float distance(float p0, float p1) { return fabs(p0 - p1); }
float distance(cl::sycl::float2 p0, cl::sycl::float2 p1) {
  return length(p0 - p1);
}
float distance(cl::sycl::float3 p0, cl::sycl::float3 p1) {
  return length(p0 - p1);
}
float distance(cl::sycl::float4 p0, cl::sycl::float4 p1) {
  return length(p0 - p1);
}
double distance(double p0, double p1) { return fabs(p0 - p1); }
double distance(cl::sycl::double2 p0, cl::sycl::double2 p1) {
  return length(p0 - p1);
}
double distance(cl::sycl::double3 p0, cl::sycl::double3 p1) {
  return length(p0 - p1);
}
double distance(cl::sycl::double4 p0, cl::sycl::double4 p1) {
  return length(p0 - p1);
}

float length(float p) { return p; }
float length(cl::sycl::float2 p) { return sqrt(dot(p, p)); }
float length(cl::sycl::float3 p) { return sqrt(dot(p, p)); }
float length(cl::sycl::float4 p) { return sqrt(dot(p, p)); }
double length(double p) { return p; }
double length(cl::sycl::double2 p) { return sqrt(dot(p, p)); }
double length(cl::sycl::double3 p) { return sqrt(dot(p, p)); }
double length(cl::sycl::double4 p) { return sqrt(dot(p, p)); }

template <typename T, int N>
cl::sycl::vec<T, N> normalize_t(cl::sycl::vec<T, N> a) {
  cl::sycl::vec<T, N> res;
  T dot_a = length(a);
  if (dot_a == 0)
    return cl::sycl::vec<T, N>(0);
  for (int i = 0; i < N; i++)
    setElement<T, N>(res, i, getElement<T, N>(a, i) / dot_a);
  return res;
}

float normalize(float p) { return 1; }
cl::sycl::float2 normalize(cl::sycl::float2 p) { return normalize_t(p); }
cl::sycl::float3 normalize(cl::sycl::float3 p) { return normalize_t(p); }
cl::sycl::float4 normalize(cl::sycl::float4 p) { return normalize_t(p); }
double normalize(double p) { return 1; }
cl::sycl::double2 normalize(cl::sycl::double2 p) { return normalize_t(p); }
cl::sycl::double3 normalize(cl::sycl::double3 p) { return normalize_t(p); }
cl::sycl::double4 normalize(cl::sycl::double4 p) { return normalize_t(p); }

float fast_distance(float p0, float p1) { return distance(p0, p1); }
float fast_distance(cl::sycl::float2 p0, cl::sycl::float2 p1) {
  return distance(p0, p1);
}
float fast_distance(cl::sycl::float3 p0, cl::sycl::float3 p1) {
  return distance(p0, p1);
}
float fast_distance(cl::sycl::float4 p0, cl::sycl::float4 p1) {
  return distance(p0, p1);
}

float fast_length(float p) { return length(p); }
float fast_length(cl::sycl::float2 p) { return length(p); }
float fast_length(cl::sycl::float3 p) { return length(p); }
float fast_length(cl::sycl::float4 p) { return length(p); }

float fast_normalize(float p) { return normalize(p); }
cl::sycl::float2 fast_normalize(cl::sycl::float2 p) { return normalize(p); }
cl::sycl::float3 fast_normalize(cl::sycl::float3 p) { return normalize(p); }
cl::sycl::float4 fast_normalize(cl::sycl::float4 p) { return normalize(p); }

} /* namespace reference */
