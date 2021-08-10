/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_MATH_REFERENCE_H
#define __SYCLCTS_UTIL_MATH_REFERENCE_H

#include "../tests/common/sycl.h"
#include "./math_helper.h"

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
int any(T x) {
   return sycl_cts::math::if_msb_set(x);
}
template <typename T, int N> int any(sycl::vec<T, N> a) {
  for (int i = 0; i < N; i++) {
    if (any(getElement(a, i)) == 1)
      return 1;
  }
  return 0;
}

template <typename T>
int all(T x) {
   return sycl_cts::math::if_msb_set(x);
}
template <typename T, int N> int all(sycl::vec<T, N> a) {
  for (int i = 0; i < N; i++) {
    if (all(getElement(a, i)) == 0)
      return 0;
  }
  return 1;
}

template <typename T>
T bitselect(T a, T b, T c) {
  return (c & b) | (~c & a);
}
float bitselect(float a, float b, float c);
double bitselect(double a, double b, double c);
sycl::half bitselect(sycl::half a, sycl::half b, sycl::half c);
template <typename T, int N>
sycl::vec<T, N> bitselect(sycl::vec<T, N> a, sycl::vec<T, N> b,
                              sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y, T z) { return bitselect(x, y, z); }, a, b, c);
}

template <typename T, typename U>
T select(T a, T b, U c) { return c ? b : a; }

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

/* absolute value */
template <typename T> auto abs(T x) { return x < 0 ? -x : x; }

template <typename T, int N, typename R = typename std::make_unsigned<T>::type>
sycl::vec<R, N> abs(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<R, T, N>([](T x) { return abs(x); },
                                                     a);
}

/* absolute difference */
template <typename T> T abs_diff(T a, T b) {
  T h = (a > b) ? a : b;
  T l = (a <= b) ? a : b;
  return h - l;
}
template <typename T, int N, typename R = typename std::make_unsigned<T>::type>
sycl::vec<typename std::make_unsigned<T>::type, N>
abs_diff(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<R, T, N>(
      [](T x, T y) { return abs_diff(x, y); }, a, b);
}

/* add with saturation */
template <typename T> T add_sat(T a, T b) {
  if (std::is_unsigned<T>::value) {
    T res = a + b;
    if (res < a)
      res = -1;
    return res;
  } else {
    typedef typename std::make_unsigned<T>::type U;
    T r = T(U(a) + U(b));
    if (b > 0) {
      if (r < a)
        return std::numeric_limits<T>::max();
    } else {
      if (r > a)
        return std::numeric_limits<T>::min();
    }
    return r;
  }
}

template <typename T, int N>
sycl::vec<T, N> add_sat(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return add_sat(x, y); }, a, b);
}

/* half add */
template <typename T> T hadd(T a, T b) {
  if (std::is_unsigned<T>::value)
    return (a >> 1) + (b >> 1) + ((a & b) & 0x1);
  return (a >> 1) + (b >> 1) + (a & b & 1);
}

template <typename T, int N>
sycl::vec<T, N> hadd(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return hadd(x, y); }, a, b);
}

/* round up half add */
template <typename T> T rhadd(T a, T b) {
  return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

template <typename T, int N>
sycl::vec<T, N> rhadd(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return rhadd(x, y); }, a, b);
}

/* clamp */
template <typename T> sycl_cts::resultRef<T> clamp(T v, T minv, T maxv) {
  if (minv > maxv)
    return sycl_cts::resultRef<T>(T(), true);
  return (v < minv) ? minv : ((v > maxv) ? maxv : v);
}

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>>
clamp(sycl::vec<T, N> a, sycl::vec<T, N> b, sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return clamp(x, y, z); }, a, b, c);
}

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> clamp(sycl::vec<T, N> a, T b,
                                               T c) {
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

/* degrees */

float degrees(float a);
double degrees(double a);

template <typename T, int N>
sycl::vec<T, N> degrees(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return degrees(x); }, a);
}

/* radians */
float radians(float a);
double radians(double a);

template <typename T, int N>
sycl::vec<T, N> radians(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return radians(x); }, a);
}

/* step */
float step(float a, float b);
double step(double a, double b);

template <typename T, int N>
sycl::vec<T, N> step(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, step(getElement(a, i), getElement(b, i)));
  }
  return res;
}

template <typename T, int N>
sycl::vec<T, N> step(T a, sycl::vec<T, N> b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, step(a, getElement(b, i)));
  }
  return res;
}

/* smoothstep */
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
sycl_cts::resultRef<sycl::vec<T, N>> smoothstep(T a, T b,
                                                    sycl::vec<T, N> c) {
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

/* sign */
float sign(float a);
double sign(double a);

template <typename T, int N> sycl::vec<T, N> sign(sycl::vec<T, N> a) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, sign(getElement(a, i)));
  }
  return res;
}

/* count leading zeros */
template <typename T> T clz(T x) {
  int lz = 0;
  for (int i = 0; i < sycl_cts::math::num_bits(x); i++)
    if (x & (1ull << i))
      lz = 0;
    else
      lz++;
  return static_cast<T>(lz);
}

template <typename T, int N> sycl::vec<T, N> clz(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return clz(x); },
                                                     a);
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

template <typename T, int N>
sycl::vec<T, N> mul_hi(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return mul_hi(x, y); }, a, b);
}

/* multiply add, get high part */
template <typename T> T mad_hi(T x, T y, T z) { return mul_hi(x, y) + z; }

template <typename T, int N>
sycl::vec<T, N> mad_hi(sycl::vec<T, N> a, sycl::vec<T, N> b,
                           sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y, T z) { return mad_hi(x, y, z); }, a, b, c);
}

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

template <typename T, int N>
sycl::vec<T, N> mad_sat(sycl::vec<T, N> a, sycl::vec<T, N> b,
                            sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y, T z) { return mad_sat(x, y, z); }, a, b, c);
}

/* maximum value */
template <typename T> T max(T a, T b) { return (a > b) ? a : b; }

template <typename T, int N>
sycl::vec<T, N> max(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return max(x, y); }, a, b);
}

template <typename T, int N>
sycl::vec<T, N> max(sycl::vec<T, N> a, T b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, max(getElement(a, i), b));
  }
  return res;
}

/* minimum value */
template <typename T> T min(T a, T b) { return (a < b) ? a : b; }

template <typename T, int N>
sycl::vec<T, N> min(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return min(x, y); }, a, b);
}

template <typename T, int N>
sycl::vec<T, N> min(sycl::vec<T, N> a, T b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, min(getElement(a, i), b));
  }
  return res;
}

/* mix */
sycl_cts::resultRef<float> mix(const float a, const float b, const float c);
sycl_cts::resultRef<double> mix(const double a, const double b, const double c);

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>>
mix(sycl::vec<T, N> a, sycl::vec<T, N> b, sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return mix(x, y, z); }, a, b, c);
}

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> mix(sycl::vec<T, N> a,
                                             sycl::vec<T, N> b, T c) {
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

/* bitwise rotate */
template <typename T> T rotate(T v, T i) {
  if (std::is_unsigned<T>::value) {
    i = i % sycl_cts::math::num_bits(v);
    if (i == 0)
      return v;
    size_t nBits = sycl_cts::math::num_bits(v) - size_t(i);
    return T((v << i) | ((v >> nBits)));
  }
  typedef typename std::make_unsigned<T>::type R;
  R i_mod = R(i) % sycl_cts::math::num_bits(v);
  if (i_mod == 0)
    return v;
  T mask = T((T(1) << i_mod) - T(1));
  size_t nBits = sycl_cts::math::num_bits(v) - size_t(i_mod);
  return T((v << i_mod) | ((v >> nBits) & mask));
}

template <typename T, int N>
sycl::vec<T, N> rotate(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return rotate(x, y); }, a, b);
}

/* return number of non zero bits in x */
template <typename T> T popcount(T x) {
  int lz = 0;
  for (int i = 0; i < sycl_cts::math::num_bits(x); i++)
    if (x & (1ull << i))
      lz++;
  return lz;
}

template <typename T, int N>
sycl::vec<T, N> popcount(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return popcount(x); }, a);
}

/* substract with saturation */
template <typename T> T sub_sat(T x, T y) {
  if (std::is_unsigned<T>::value)
    return x <= y ? 0 : x - y;

  const T max_val = std::numeric_limits<T>::max();
  const T min_val = std::numeric_limits<T>::min();
  if (x > 0) {
    if (y > 0) {
      return x - y;
    } else // x > 0, y <= 0
    {
      return (x - max_val) > y ? max_val : x - y;
    }
  } else // x <= 0
  {
    if (y > 0) {
      return (x - min_val) < y ? min_val : x - y;
    } else // x <= 0, y <= 0
    {
      return x - y;
    }
  }
}

template <typename T, int N>
sycl::vec<T, N> sub_sat(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return sub_sat(x, y); }, a, b);
}

/* upsample */
uint16_t upsample(uint8_t h, uint8_t l);
uint32_t upsample(uint16_t h, uint16_t l);
uint64_t upsample(uint32_t h, uint32_t l);
int16_t upsample(int8_t h, uint8_t l);
int32_t upsample(int16_t h, uint16_t l);
int64_t upsample(int32_t h, uint32_t l);

template <typename T> struct upsample_t;

template <> struct upsample_t<uint8_t> { using type = uint16_t; };

template <> struct upsample_t<uint16_t> { using type = uint32_t; };

template <> struct upsample_t<uint32_t> { using type = uint64_t; };

template <> struct upsample_t<int8_t> { using type = int16_t; };

template <> struct upsample_t<int16_t> { using type = int32_t; };

template <> struct upsample_t<int32_t> { using type = int64_t; };

template <typename T, int N>
sycl::vec<typename upsample_t<T>::type, N>
upsample(sycl::vec<T, N> a,
         sycl::vec<typename std::make_unsigned<T>::type, N> b) {
  return sycl_cts::math::run_func_on_vector<typename upsample_t<T>::type, T, N>(
      [](T x, T y) { return upsample(x, y); }, a, b);
}

/* fast multiply add 24bits */
sycl_cts::resultRef<int32_t> mad24(int32_t x, int32_t y, int32_t z);
sycl_cts::resultRef<uint32_t> mad24(uint32_t x, uint32_t y, uint32_t z);

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>>
mad24(sycl::vec<T, N> a, sycl::vec<T, N> b, sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return mad24(x, y, z); }, a, b, c);
}

/* fast multiply 24bits */
sycl_cts::resultRef<int32_t> mul24(int32_t x, int32_t y);
sycl_cts::resultRef<uint32_t> mul24(uint32_t x, uint32_t y);

template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> mul24(sycl::vec<T, N> a,
                                               sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return mul24(x, y); }, a, b);
}
// Math Functions

template <typename T> struct higher_accuracy;

template <> struct higher_accuracy<sycl::half> { using type = float; };

template <> struct higher_accuracy<float> { using type = double; };

template <> struct higher_accuracy<double> { using type = long double; };

template <typename T>
T acos(T a) {
  return std::acos(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> acos(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return acos(x); }, a);
}

template <typename T>
T acosh(T a) {
  return std::acosh(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> acosh(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return acosh(x); }, a);
}

sycl::half acospi(sycl::half a);
float acospi(float a);
double acospi(double a);
template <typename T, int N> sycl::vec<T, N> acospi(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return acospi(x); }, a);
}

template <typename T>
T asin(T a) {
  return std::asin(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> asin(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return asin(x); }, a);
}

template <typename T>
T asinh(T a) {
  return std::asinh(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> asinh(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return asinh(x); }, a);
}

sycl::half asinpi(sycl::half a);
float asinpi(float a);
double asinpi(double a);
template <typename T, int N> sycl::vec<T, N> asinpi(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return asinpi(x); }, a);
}

template <typename T>
T atan(T a) {
  return std::atan(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> atan(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return atan(x); }, a);
}

template <typename T>
T atan2(T a, T b) {
  return std::atan2(static_cast<typename higher_accuracy<T>::type>(a), b);
}
template <typename T, int N>
sycl::vec<T, N> atan2(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return atan2(x, y); }, a, b);
}

template <typename T>
T atanh(T a) {
  return std::atanh(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> atanh(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return atanh(x); }, a);
}

sycl::half atanpi(sycl::half a);
float atanpi(float a);
double atanpi(double a);
template <typename T, int N> sycl::vec<T, N> atanpi(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return atanpi(x); }, a);
}

sycl::half atan2pi(sycl::half a, sycl::half b);
float atan2pi(float a, float b);
double atan2pi(double a, double b);
template <typename T, int N>
sycl::vec<T, N> atan2pi(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return atan2pi(x, y); }, a, b);
}

template <typename T>
T cbrt(T a) {
  return std::cbrt(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> cbrt(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return cbrt(x); }, a);
}

using std::ceil;
template <typename T, int N> sycl::vec<T, N> ceil(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return ceil(x); }, a);
}

using std::copysign;
template <typename T, int N>
sycl::vec<T, N> copysign(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return copysign(x, y); }, a, b);
}

template <typename T>
T cos(T a) {
  return std::cos(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> cos(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return cos(x); },
                                                     a);
}

template <typename T>
T cosh(T a) {
  return std::cosh(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> cosh(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return cosh(x); }, a);
}

sycl::half cospi(sycl::half a);
float cospi(float a);
double cospi(double a);
template <typename T, int N> sycl::vec<T, N> cospi(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return cospi(x); }, a);
}

template <typename T>
T erfc(T a) {
  return std::erfc(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> erfc(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return erfc(x); }, a);
}

template <typename T>
T erf(T a) {
  return std::erf(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> erf(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return erf(x); },
                                                     a);
}

template <typename T>
T exp(T a) {
  return std::exp(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> exp(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return exp(x); },
                                                     a);
}

template <typename T>
T exp2(T a) {
  return std::exp2(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> exp2(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return exp2(x); }, a);
}

template <typename T>
T exp10(T a) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(10),
      static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> exp10(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return exp10(x); }, a);
}

template <typename T>
T expm1(T a) {
  return std::expm1(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> expm1(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return expm1(x); }, a);
}

using std::fabs;
template <typename T, int N> sycl::vec<T, N> fabs(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return fabs(x); }, a);
}

using std::fdim;
sycl::half fdim(sycl::half a, sycl::half b);
template <typename T, int N>
sycl::vec<T, N> fdim(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return fdim(x, y); }, a, b);
}

using std::floor;
template <typename T, int N> sycl::vec<T, N> floor(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return floor(x); }, a);
}

sycl::half fma(sycl::half a, sycl::half b, sycl::half c);
float fma(float a, float b, float c);
double fma(double a, double b, double c);
template <typename T, int N>
sycl::vec<T, N> fma(sycl::vec<T, N> a, sycl::vec<T, N> b,
                        sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y, T z) { return fma(x, y, z); }, a, b, c);
}

using std::fmax;
template <typename T, int N>
sycl::vec<T, N> fmax(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return fmax(x, y); }, a, b);
}

template <typename T, int N>
sycl::vec<T, N> fmax(sycl::vec<T, N> a, T b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, fmax(getElement<T, N>(a, i), b));
  }
  return res;
}

using std::fmin;
template <typename T, int N>
sycl::vec<T, N> fmin(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return fmin(x, y); }, a, b);
}
template <typename T, int N>
sycl::vec<T, N> fmin(sycl::vec<T, N> a, T b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, fmin(getElement<T, N>(a, i), b));
  }
  return res;
}

using std::fmod;
template <typename T, int N>
sycl::vec<T, N> fmod(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return fmod(x, y); }, a, b);
}

template <typename T>
T fract(T a, T *b) {
  *b = std::floor(a);
  return std::fmin(a - *b, nextafter(T(1.0), T(0.0)));
}
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

template <typename T>
T hypot(T a, T b) {
  return std::hypot(static_cast<typename higher_accuracy<T>::type>(a), b);
}
template <typename T, int N>
sycl::vec<T, N> hypot(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return hypot(x, y); }, a, b);
}

using std::ilogb;
template <typename T, int N>
sycl::vec<int, N> ilogb(sycl::vec<T, N> a) {
  sycl::vec<int, N> res;
  for (int i = 0; i < N; i++) {
    setElement<int, N>(res, i, ilogb(getElement<T, N>(a, i)));
  }
  return res;
}

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

template <typename T, int N>
sycl::vec<T, N> ldexp(sycl::vec<T, N> a, int b) {
  sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, ldexp(getElement<T, N>(a, i), b));
  }
  return res;
}

using std::lgamma;
template <typename T, int N> sycl::vec<T, N> lgamma(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return lgamma(x); }, a);
}

template <typename T> T lgamma_r(T a, int *b) {
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

template <typename T>
T log(T a) {
  return std::log(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> log(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return log(x); },
                                                     a);
}

template <typename T>
T log2(T a) {
  return std::log2(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> log2(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return log2(x); }, a);
}

template <typename T>
T log10(T a) {
  return std::log10(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> log10(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return log10(x); }, a);
}

template <typename T>
T log1p(T a) {
  return std::log1p(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> log1p(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return log1p(x); }, a);
}

using std::logb;
template <typename T, int N> sycl::vec<T, N> logb(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return logb(x); }, a);
}

template <typename T> T mad(T a, T b, T c) {
  return a * b + c;
  }
template <typename T, int N>
sycl::vec<T, N> mad(sycl::vec<T, N> a, sycl::vec<T, N> b,
                        sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y, T z) { return fma(x, y, z); }, a, b, c);
}

template <typename T> T maxmag(T a, T b) {
  if (fabs(a) > fabs(b))
    return a;
  else if (fabs(b) > fabs(a))
    return b;
  return fmax(a, b);
}
template <typename T, int N>
sycl::vec<T, N> maxmag(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return maxmag(x, y); }, a, b);
}

template <typename T> T minmag(T a, T b) {
  if (fabs(a) < fabs(b))
    return a;
  else if (fabs(b) < fabs(a))
    return b;
  return fmin(a, b);
}
template <typename T, int N>
sycl::vec<T, N> minmag(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return minmag(x, y); }, a, b);
}

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

float nan(unsigned int a);
double nan(unsigned long a);
double nan(unsigned long long a);
template <int N> sycl::vec<float, N> nan(sycl::vec<unsigned int, N> a) {
  return sycl_cts::math::run_func_on_vector<float, unsigned int, N>(
      [](unsigned int x) { return nan(x); }, a);
}

template <typename T, int N>
typename std::enable_if<std::is_same<unsigned long, T>::value ||
                            std::is_same<unsigned long long, T>::value,
                        sycl::vec<double, N>>::type
nan(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<double, T, N>(
      [](T x) { return nan(x); }, a);
}

using std::nextafter;
sycl::half nextafter(sycl::half a, sycl::half b);
template <typename T, int N>
sycl::vec<T, N> nextafter(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return nextafter(x, y); }, a, b);
}

template <typename T>
T pow(T a, T b) {
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
      static_cast<typename higher_accuracy<T>::type>(b));
}
template <typename T, int N>
sycl::vec<T, N> pow(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return pow(x, y); }, a, b);
}

template <typename T> T pown(T a, int b) {
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

template <typename T>
sycl_cts::resultRef<T> powr(T a, T b) {
  if (a < 0)
    return sycl_cts::resultRef<T>(T(), true);
  return std::pow(static_cast<typename higher_accuracy<T>::type>(a),
      static_cast<typename higher_accuracy<T>::type>(b));
}
template <typename T, int N>
sycl_cts::resultRef<sycl::vec<T, N>> powr(sycl::vec<T, N> a,
                                              sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return powr(x, y); }, a, b);
}

using std::remainder;
template <typename T, int N>
sycl::vec<T, N> remainder(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return remainder(x, y); }, a, b);
}

using std::remquo;
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

using std::rint;
template <typename T, int N> sycl::vec<T, N> rint(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return rint(x); }, a);
}

template <typename T> T rootn(T a, int b) {
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

using std::round;
template <typename T, int N> sycl::vec<T, N> round(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return round(x); }, a);
}

template <typename T> T rsqrt(T a) {
  return 1 / std::sqrt(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> rsqrt(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return rsqrt(x); }, a);
}

template <typename T> T sincos(T a, T *b) {
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

template <typename T>
T sin(T a) {
  return std::sin(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> sin(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return sin(x); },
                                                     a);
}

template <typename T>
T sinh(T a) {
  return std::sinh(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> sinh(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return sinh(x); }, a);
}

sycl::half sinpi(sycl::half a);
float sinpi(float a);
double sinpi(double a);
template <typename T, int N> sycl::vec<T, N> sinpi(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return sinpi(x); }, a);
}

template <typename T>
T sqrt(T a) {
  return std::sqrt(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> sqrt(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return sqrt(x); }, a);
}

template <typename T>
T tan(T a) {
  return std::tan(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> tan(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return tan(x); },
                                                     a);
}

template <typename T>
T tanh(T a) {
  return std::tanh(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> tanh(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return tanh(x); }, a);
}

sycl::half tanpi(sycl::half a);
float tanpi(float a);
double tanpi(double a);
template <typename T, int N> sycl::vec<T, N> tanpi(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return tanpi(x); }, a);
}

template <typename T>
T tgamma(T a) {
  return std::tgamma(static_cast<typename higher_accuracy<T>::type>(a));
}
template <typename T, int N> sycl::vec<T, N> tgamma(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return tgamma(x); }, a);
}

using std::trunc;
template <typename T, int N> sycl::vec<T, N> trunc(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return trunc(x); }, a);
}

template <typename T> T recip(T a) { return 1.0 / a; }
template <typename T, int N> sycl::vec<T, N> recip(sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return recip(x); }, a);
}

template <typename T>
T divide(T a, T b) {
  return a / b;
}
template <typename T, int N>
sycl::vec<T, N> divide(sycl::vec<T, N> a, sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return divide(x, y); }, a, b);
}

// Geometric funcs

sycl::float4 cross(sycl::float4 p0, sycl::float4 p1);
sycl::float3 cross(sycl::float3 p0, sycl::float3 p1);
sycl::double4 cross(sycl::double4 p0, sycl::double4 p1);
sycl::double3 cross(sycl::double3 p0, sycl::double3 p1);

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

template <typename T>
auto length(T p) {
  return sqrt(reference::dot(p, p));
}

template <typename T>
auto distance(T p0, T p1) {
  return reference::length(p0 - p1);
}

template <typename T>
T normalize(T p) { return 1; }

template <typename T, int N>
sycl::vec<T, N> normalize(sycl::vec<T, N> a) {
  sycl::vec<T, N> res;
  T dot_a = reference::length(a);
  if (dot_a == 0)
    return sycl::vec<T, N>(0);
  for (int i = 0; i < N; i++)
    setElement<T, N>(res, i, getElement<T, N>(a, i) / dot_a);
  return res;
}

template <typename T>
auto fast_distance(T p0, T p1) {
  return reference::distance<T>(p0, p1);
}

template <typename T>
auto fast_length(T p0) {
  return reference::length(p0);
}

template <typename T>
auto fast_normalize(T p0) {
  return reference::normalize(p0);
}

} // reference

#endif // __SYCLCTS_UTIL_MATH_REFERENCE_H
