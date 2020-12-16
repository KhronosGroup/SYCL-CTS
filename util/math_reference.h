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
template <typename T, int N> int any(cl::sycl::vec<T, N> a) {
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
template <typename T, int N> int all(cl::sycl::vec<T, N> a) {
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
cl::sycl::half bitselect(cl::sycl::half a, cl::sycl::half b, cl::sycl::half c);
template <typename T, int N>
cl::sycl::vec<T, N> bitselect(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                              cl::sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y, T z) { return bitselect(x, y, z); }, a, b, c);
}

template <typename T, typename U>
T select(T a, T b, U c) { return c ? b : a; }

template <typename T, typename K, int N>
cl::sycl::vec<T, N> select(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                           cl::sycl::vec<K, N> c) {
  cl::sycl::vec<T, N> res;
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
cl::sycl::vec<R, N> abs(cl::sycl::vec<T, N> a) {
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
cl::sycl::vec<typename std::make_unsigned<T>::type, N>
abs_diff(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
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
cl::sycl::vec<T, N> add_sat(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
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
cl::sycl::vec<T, N> hadd(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return hadd(x, y); }, a, b);
}

/* round up half add */
template <typename T> T rhadd(T a, T b) {
  return (a >> 1) + (b >> 1) + ((a & 1) | (b & 1));
}

template <typename T, int N>
cl::sycl::vec<T, N> rhadd(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
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
sycl_cts::resultRef<cl::sycl::vec<T, N>>
clamp(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b, cl::sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return clamp(x, y, z); }, a, b, c);
}

template <typename T, int N>
sycl_cts::resultRef<cl::sycl::vec<T, N>> clamp(cl::sycl::vec<T, N> a, T b,
                                               T c) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    sycl_cts::resultRef<T> element = clamp(getElement(a, i), b, c);
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<cl::sycl::vec<T, N>>(res, undefined);
}

/* degrees */

float degrees(float a);
double degrees(double a);

template <typename T, int N>
cl::sycl::vec<T, N> degrees(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return degrees(x); }, a);
}

/* radians */
float radians(float a);
double radians(double a);

template <typename T, int N>
cl::sycl::vec<T, N> radians(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return radians(x); }, a);
}

/* step */
float step(float a, float b);
double step(double a, double b);

template <typename T, int N>
cl::sycl::vec<T, N> step(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, step(getElement(a, i), getElement(b, i)));
  }
  return res;
}

template <typename T, int N>
cl::sycl::vec<T, N> step(T a, cl::sycl::vec<T, N> b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, step(a, getElement(b, i)));
  }
  return res;
}

/* smoothstep */
sycl_cts::resultRef<float> smoothstep(float a, float b, float c);
sycl_cts::resultRef<double> smoothstep(double a, double b, double c);

template <typename T, int N>
sycl_cts::resultRef<cl::sycl::vec<T, N>> smoothstep(cl::sycl::vec<T, N> a,
                                                    cl::sycl::vec<T, N> b,
                                                    cl::sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return smoothstep(x, y, z); }, a, b, c);
}

template <typename T, int N>
sycl_cts::resultRef<cl::sycl::vec<T, N>> smoothstep(T a, T b,
                                                    cl::sycl::vec<T, N> c) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    sycl_cts::resultRef<T> element = smoothstep(a, b, getElement(c, i));
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<cl::sycl::vec<T, N>>(res, undefined);
}

/* sign */
float sign(float a);
double sign(double a);

template <typename T, int N> cl::sycl::vec<T, N> sign(cl::sycl::vec<T, N> a) {
  cl::sycl::vec<T, N> res;
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

template <typename T, int N> cl::sycl::vec<T, N> clz(cl::sycl::vec<T, N> a) {
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
cl::sycl::vec<T, N> mul_hi(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return mul_hi(x, y); }, a, b);
}

/* multiply add, get high part */
template <typename T> T mad_hi(T x, T y, T z) { return mul_hi(x, y) + z; }

template <typename T, int N>
cl::sycl::vec<T, N> mad_hi(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                           cl::sycl::vec<T, N> c) {
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
cl::sycl::vec<T, N> mad_sat(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                            cl::sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y, T z) { return mad_sat(x, y, z); }, a, b, c);
}

/* maximum value */
template <typename T> T max(T a, T b) { return (a > b) ? a : b; }

template <typename T, int N>
cl::sycl::vec<T, N> max(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return max(x, y); }, a, b);
}

template <typename T, int N>
cl::sycl::vec<T, N> max(cl::sycl::vec<T, N> a, T b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, max(getElement(a, i), b));
  }
  return res;
}

/* minimum value */
template <typename T> T min(T a, T b) { return (a < b) ? a : b; }

template <typename T, int N>
cl::sycl::vec<T, N> min(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return min(x, y); }, a, b);
}

template <typename T, int N>
cl::sycl::vec<T, N> min(cl::sycl::vec<T, N> a, T b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, min(getElement(a, i), b));
  }
  return res;
}

/* mix */
sycl_cts::resultRef<float> mix(const float a, const float b, const float c);
sycl_cts::resultRef<double> mix(const double a, const double b, const double c);

template <typename T, int N>
sycl_cts::resultRef<cl::sycl::vec<T, N>>
mix(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b, cl::sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return mix(x, y, z); }, a, b, c);
}

template <typename T, int N>
sycl_cts::resultRef<cl::sycl::vec<T, N>> mix(cl::sycl::vec<T, N> a,
                                             cl::sycl::vec<T, N> b, T c) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    sycl_cts::resultRef<T> element = mix(getElement(a, i), getElement(b, i), c);
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<cl::sycl::vec<T, N>>(res, undefined);
}

/* bitwise rotate */
template <typename T> T rotate(T v, T i) {
  if (std::is_unsigned<T>::value) {
    i = i % sycl_cts::math::num_bits(v);
    size_t nBits = sycl_cts::math::num_bits(v) - size_t(i);
    return T((v << i) | ((v >> nBits)));
  }
  typedef typename std::make_unsigned<T>::type R;
  R i_mod = R(i) % sycl_cts::math::num_bits(v);
  T mask = T((T(1) << i_mod) - T(1));
  size_t nBits = sycl_cts::math::num_bits(v) - size_t(i_mod);
  return T((v << i_mod) | ((v >> nBits) & mask));
}

template <typename T, int N>
cl::sycl::vec<T, N> rotate(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
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
cl::sycl::vec<T, N> popcount(cl::sycl::vec<T, N> a) {
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
cl::sycl::vec<T, N> sub_sat(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
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
cl::sycl::vec<typename upsample_t<T>::type, N>
upsample(cl::sycl::vec<T, N> a,
         cl::sycl::vec<typename std::make_unsigned<T>::type, N> b) {
  return sycl_cts::math::run_func_on_vector<typename upsample_t<T>::type, T, N>(
      [](T x, T y) { return upsample(x, y); }, a, b);
}

/* fast multiply add 24bits */
sycl_cts::resultRef<int32_t> mad24(int32_t x, int32_t y, int32_t z);
sycl_cts::resultRef<uint32_t> mad24(uint32_t x, uint32_t y, uint32_t z);

template <typename T, int N>
sycl_cts::resultRef<cl::sycl::vec<T, N>>
mad24(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b, cl::sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y, T z) { return mad24(x, y, z); }, a, b, c);
}

/* fast multiply 24bits */
sycl_cts::resultRef<int32_t> mul24(int32_t x, int32_t y);
sycl_cts::resultRef<uint32_t> mul24(uint32_t x, uint32_t y);

template <typename T, int N>
sycl_cts::resultRef<cl::sycl::vec<T, N>> mul24(cl::sycl::vec<T, N> a,
                                               cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return mul24(x, y); }, a, b);
}
// Math Functions

cl::sycl::half acos(cl::sycl::half a);
float acos(float a);
double acos(double a);
template <typename T, int N> cl::sycl::vec<T, N> acos(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return acos(x); }, a);
}

cl::sycl::half acosh(cl::sycl::half a);
float acosh(float a);
double acosh(double a);
template <typename T, int N> cl::sycl::vec<T, N> acosh(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return acosh(x); }, a);
}

cl::sycl::half acospi(cl::sycl::half a);
float acospi(float a);
double acospi(double a);
template <typename T, int N> cl::sycl::vec<T, N> acospi(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return acospi(x); }, a);
}

cl::sycl::half asin(cl::sycl::half a);
float asin(float a);
double asin(double a);
template <typename T, int N> cl::sycl::vec<T, N> asin(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return asin(x); }, a);
}

cl::sycl::half asinh(cl::sycl::half a);
float asinh(float a);
double asinh(double a);
template <typename T, int N> cl::sycl::vec<T, N> asinh(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return asinh(x); }, a);
}

cl::sycl::half asinpi(cl::sycl::half a);
float asinpi(float a);
double asinpi(double a);
template <typename T, int N> cl::sycl::vec<T, N> asinpi(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return asinpi(x); }, a);
}

cl::sycl::half atan(cl::sycl::half a);
float atan(float a);
double atan(double a);
template <typename T, int N> cl::sycl::vec<T, N> atan(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return atan(x); }, a);
}

cl::sycl::half atan2(cl::sycl::half a, cl::sycl::half b);
float atan2(float a, float b);
double atan2(double a, double b);
template <typename T, int N>
cl::sycl::vec<T, N> atan2(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return atan2(x, y); }, a, b);
}

cl::sycl::half atanh(cl::sycl::half a);
float atanh(float a);
double atanh(double a);
template <typename T, int N> cl::sycl::vec<T, N> atanh(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return atanh(x); }, a);
}

cl::sycl::half atanpi(cl::sycl::half a);
float atanpi(float a);
double atanpi(double a);
template <typename T, int N> cl::sycl::vec<T, N> atanpi(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return atanpi(x); }, a);
}

cl::sycl::half atan2pi(cl::sycl::half a, cl::sycl::half b);
float atan2pi(float a, float b);
double atan2pi(double a, double b);
template <typename T, int N>
cl::sycl::vec<T, N> atan2pi(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return atan2pi(x, y); }, a, b);
}

cl::sycl::half cbrt(cl::sycl::half a);
float cbrt(float a);
double cbrt(double a);
template <typename T, int N> cl::sycl::vec<T, N> cbrt(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return cbrt(x); }, a);
}

using std::ceil;
template <typename T, int N> cl::sycl::vec<T, N> ceil(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return ceil(x); }, a);
}

using std::copysign;
template <typename T, int N>
cl::sycl::vec<T, N> copysign(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return copysign(x, y); }, a, b);
}

cl::sycl::half cos(cl::sycl::half a);
float cos(float a);
double cos(double a);
template <typename T, int N> cl::sycl::vec<T, N> cos(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return cos(x); },
                                                     a);
}

cl::sycl::half cosh(cl::sycl::half a);
float cosh(float a);
double cosh(double a);
template <typename T, int N> cl::sycl::vec<T, N> cosh(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return cosh(x); }, a);
}

cl::sycl::half cospi(cl::sycl::half a);
float cospi(float a);
double cospi(double a);
template <typename T, int N> cl::sycl::vec<T, N> cospi(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return cospi(x); }, a);
}

cl::sycl::half erfc(cl::sycl::half a);
float erfc(float a);
double erfc(double a);
template <typename T, int N> cl::sycl::vec<T, N> erfc(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return erfc(x); }, a);
}

cl::sycl::half erf(cl::sycl::half a);
float erf(float a);
double erf(double a);
template <typename T, int N> cl::sycl::vec<T, N> erf(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return erf(x); },
                                                     a);
}

cl::sycl::half exp(cl::sycl::half a);
float exp(float a);
double exp(double a);
template <typename T, int N> cl::sycl::vec<T, N> exp(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return exp(x); },
                                                     a);
}

cl::sycl::half exp2(cl::sycl::half a);
float exp2(float a);
double exp2(double a);
template <typename T, int N> cl::sycl::vec<T, N> exp2(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return exp2(x); }, a);
}

float exp10(float a);
double exp10(double a);
template <typename T, int N> cl::sycl::vec<T, N> exp10(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return exp10(x); }, a);
}

cl::sycl::half expm1(cl::sycl::half a);
float expm1(float a);
double expm1(double a);
template <typename T, int N> cl::sycl::vec<T, N> expm1(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return expm1(x); }, a);
}

using std::fabs;
template <typename T, int N> cl::sycl::vec<T, N> fabs(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return fabs(x); }, a);
}

using std::fdim;
cl::sycl::half fdim(cl::sycl::half a, cl::sycl::half b);
template <typename T, int N>
cl::sycl::vec<T, N> fdim(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return fdim(x, y); }, a, b);
}

using std::floor;
template <typename T, int N> cl::sycl::vec<T, N> floor(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return floor(x); }, a);
}

cl::sycl::half fma(cl::sycl::half a, cl::sycl::half b, cl::sycl::half c);
float fma(float a, float b, float c);
double fma(double a, double b, double c);
template <typename T, int N>
cl::sycl::vec<T, N> fma(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                        cl::sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y, T z) { return fma(x, y, z); }, a, b, c);
}

using std::fmax;
template <typename T, int N>
cl::sycl::vec<T, N> fmax(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return fmax(x, y); }, a, b);
}

template <typename T, int N>
cl::sycl::vec<T, N> fmax(cl::sycl::vec<T, N> a, T b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, fmax(getElement<T, N>(a, i), b));
  }
  return res;
}

using std::fmin;
template <typename T, int N>
cl::sycl::vec<T, N> fmin(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return fmin(x, y); }, a, b);
}
template <typename T, int N>
cl::sycl::vec<T, N> fmin(cl::sycl::vec<T, N> a, T b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, fmin(getElement<T, N>(a, i), b));
  }
  return res;
}

using std::fmod;
template <typename T, int N>
cl::sycl::vec<T, N> fmod(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return fmod(x, y); }, a, b);
}

cl::sycl::half fract(cl::sycl::half a, cl::sycl::half *b);
float fract(float a, float *b);
double fract(double a, double *b);
template <typename T, int N>
cl::sycl::vec<T, N> fract(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> *b) {
  cl::sycl::vec<T, N> res;
  cl::sycl::vec<T, N> resPtr;
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
cl::sycl::vec<T, N> frexp(cl::sycl::vec<T, N> a, cl::sycl::vec<int, N> *b) {
  cl::sycl::vec<T, N> res;
  cl::sycl::vec<int, N> resPtr;
  for (int i = 0; i < N; i++) {
    int value;
    setElement<T, N>(res, i, frexp(getElement(a, i), &value));
    setElement<int, N>(resPtr, i, value);
  }
  *b = resPtr;
  return res;
}

cl::sycl::half hypot(cl::sycl::half a, cl::sycl::half b);
float hypot(float a, float b);
double hypot(double a, double b);
template <typename T, int N>
cl::sycl::vec<T, N> hypot(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return hypot(x, y); }, a, b);
}

using std::ilogb;
template <typename T, int N>
cl::sycl::vec<int, N> ilogb(cl::sycl::vec<T, N> a) {
  cl::sycl::vec<int, N> res;
  for (int i = 0; i < N; i++) {
    setElement<int, N>(res, i, ilogb(getElement<T, N>(a, i)));
  }
  return res;
}

using std::ldexp;
template <typename T, int N>
cl::sycl::vec<T, N> ldexp(cl::sycl::vec<T, N> a, cl::sycl::vec<int, N> b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i,
                     ldexp(getElement<T, N>(a, i), getElement<int, N>(b, i)));
  }
  return res;
}

template <typename T, int N>
cl::sycl::vec<T, N> ldexp(cl::sycl::vec<T, N> a, int b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, ldexp(getElement<T, N>(a, i), b));
  }
  return res;
}

using std::lgamma;
template <typename T, int N> cl::sycl::vec<T, N> lgamma(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return lgamma(x); }, a);
}

cl::sycl::half lgamma_r(cl::sycl::half a, int *b);
float lgamma_r(float a, int *b);
double lgamma_r(double a, int *b);
template <typename T, int N>
cl::sycl::vec<T, N> lgamma_r(cl::sycl::vec<T, N> a, cl::sycl::vec<int, N> *b) {
  cl::sycl::vec<T, N> res;
  cl::sycl::vec<int, N> resPtr;
  for (int i = 0; i < N; i++) {
    int value;
    setElement<T, N>(res, i, lgamma_r(getElement(a, i), &value));
    setElement<int, N>(resPtr, i, value);
  }
  *b = resPtr;
  return res;
}

cl::sycl::half log(cl::sycl::half a);
float log(float a);
double log(double a);
template <typename T, int N> cl::sycl::vec<T, N> log(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return log(x); },
                                                     a);
}

cl::sycl::half log2(cl::sycl::half a);
float log2(float a);
double log2(double a);
template <typename T, int N> cl::sycl::vec<T, N> log2(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return log2(x); }, a);
}

cl::sycl::half log10(cl::sycl::half a);
float log10(float a);
double log10(double a);
template <typename T, int N> cl::sycl::vec<T, N> log10(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return log10(x); }, a);
}

cl::sycl::half log1p(cl::sycl::half a);
float log1p(float a);
double log1p(double a);
template <typename T, int N> cl::sycl::vec<T, N> log1p(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return log1p(x); }, a);
}

using std::logb;
template <typename T, int N> cl::sycl::vec<T, N> logb(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return logb(x); }, a);
}

cl::sycl::half mad(cl::sycl::half a, cl::sycl::half b, cl::sycl::half c);
float mad(float a, float b, float c);
double mad(double a, double b, double c);
template <typename T, int N>
cl::sycl::vec<T, N> mad(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                        cl::sycl::vec<T, N> c) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y, T z) { return fma(x, y, z); }, a, b, c);
}

cl::sycl::half maxmag(cl::sycl::half a, cl::sycl::half b);
float maxmag(float a, float b);
double maxmag(double a, double b);
template <typename T, int N>
cl::sycl::vec<T, N> maxmag(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return maxmag(x, y); }, a, b);
}

cl::sycl::half minmag(cl::sycl::half a, cl::sycl::half b);
float minmag(float a, float b);
double minmag(double a, double b);
template <typename T, int N>
cl::sycl::vec<T, N> minmag(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return minmag(x, y); }, a, b);
}

using std::modf;
cl::sycl::half modf(cl::sycl::half a, cl::sycl::half *b);
template <typename T, int N>
cl::sycl::vec<T, N> modf(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> *b) {
  cl::sycl::vec<T, N> res;
  cl::sycl::vec<T, N> resPtr;
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
template <int N> cl::sycl::vec<float, N> nan(cl::sycl::vec<unsigned int, N> a) {
  return sycl_cts::math::run_func_on_vector<float, unsigned int, N>(
      [](unsigned int x) { return nan(x); }, a);
}

template <typename T, int N>
typename std::enable_if<std::is_same<unsigned long, T>::value ||
                            std::is_same<unsigned long long, T>::value,
                        cl::sycl::vec<double, N>>::type
nan(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<double, T, N>(
      [](T x) { return nan(x); }, a);
}

using std::nextafter;
cl::sycl::half nextafter(cl::sycl::half a, cl::sycl::half b);
template <typename T, int N>
cl::sycl::vec<T, N> nextafter(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return nextafter(x, y); }, a, b);
}

cl::sycl::half pow(cl::sycl::half a, cl::sycl::half b);
float pow(float a, float b);
double pow(double a, double b);
template <typename T, int N>
cl::sycl::vec<T, N> pow(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return pow(x, y); }, a, b);
}

cl::sycl::half pown(cl::sycl::half a, int b);
float pown(float a, int b);
double pown(double a, int b);
template <typename T, int N>
cl::sycl::vec<T, N> pown(cl::sycl::vec<T, N> a, cl::sycl::vec<int, N> b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i,
                     pown(getElement<T, N>(a, i), getElement<int, N>(b, i)));
  }
  return res;
}

sycl_cts::resultRef<cl::sycl::half> powr(cl::sycl::half a, cl::sycl::half b);
sycl_cts::resultRef<float> powr(float a, float b);
sycl_cts::resultRef<double> powr(double a, double b);
template <typename T, int N>
sycl_cts::resultRef<cl::sycl::vec<T, N>> powr(cl::sycl::vec<T, N> a,
                                              cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector_result_ref<T, N>(
      [](T x, T y) { return powr(x, y); }, a, b);
}

using std::remainder;
template <typename T, int N>
cl::sycl::vec<T, N> remainder(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return remainder(x, y); }, a, b);
}

using std::remquo;
template <typename T, int N>
cl::sycl::vec<T, N> remquo(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                           cl::sycl::vec<int, N> *c) {
  cl::sycl::vec<T, N> res;
  cl::sycl::vec<int, N> resPtr;
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
template <typename T, int N> cl::sycl::vec<T, N> rint(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return rint(x); }, a);
}

cl::sycl::half rootn(cl::sycl::half a, int b);
float rootn(float a, int b);
double rootn(double a, int b);
template <typename T, int N>
cl::sycl::vec<T, N> rootn(cl::sycl::vec<T, N> a, cl::sycl::vec<int, N> b) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i,
                     rootn(getElement<T, N>(a, i), getElement<int, N>(b, i)));
  }
  return res;
}

using std::round;
template <typename T, int N> cl::sycl::vec<T, N> round(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return round(x); }, a);
}

cl::sycl::half rsqrt(cl::sycl::half a);
float rsqrt(float a);
double rsqrt(double a);
template <typename T, int N> cl::sycl::vec<T, N> rsqrt(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return rsqrt(x); }, a);
}

cl::sycl::half sincos(cl::sycl::half a, cl::sycl::half *b);
float sincos(float a, float *b);
double sincos(double a, double *b);
template <typename T, int N>
cl::sycl::vec<T, N> sincos(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> *b) {
  cl::sycl::vec<T, N> res;
  cl::sycl::vec<T, N> resPtr;
  for (int i = 0; i < N; i++) {
    T value;
    setElement<T, N>(res, i, sincos(getElement(a, i), &value));
    setElement<T, N>(resPtr, i, value);
  }
  *b = resPtr;
  return res;
}

cl::sycl::half sin(cl::sycl::half a);
float sin(float a);
double sin(double a);
template <typename T, int N> cl::sycl::vec<T, N> sin(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return sin(x); },
                                                     a);
}

cl::sycl::half sinh(cl::sycl::half a);
float sinh(float a);
double sinh(double a);
template <typename T, int N> cl::sycl::vec<T, N> sinh(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return sinh(x); }, a);
}

cl::sycl::half sinpi(cl::sycl::half a);
float sinpi(float a);
double sinpi(double a);
template <typename T, int N> cl::sycl::vec<T, N> sinpi(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return sinpi(x); }, a);
}

cl::sycl::half sqrt(cl::sycl::half a);
float sqrt(float a);
double sqrt(double a);
template <typename T, int N> cl::sycl::vec<T, N> sqrt(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return sqrt(x); }, a);
}

cl::sycl::half tan(cl::sycl::half a);
float tan(float a);
double tan(double a);
template <typename T, int N> cl::sycl::vec<T, N> tan(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>([](T x) { return tan(x); },
                                                     a);
}

cl::sycl::half tanh(cl::sycl::half a);
float tanh(float a);
double tanh(double a);
template <typename T, int N> cl::sycl::vec<T, N> tanh(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return tanh(x); }, a);
}

cl::sycl::half tanpi(cl::sycl::half a);
float tanpi(float a);
double tanpi(double a);
template <typename T, int N> cl::sycl::vec<T, N> tanpi(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return tanpi(x); }, a);
}

cl::sycl::half tgamma(cl::sycl::half a);
float tgamma(float a);
double tgamma(double a);
template <typename T, int N> cl::sycl::vec<T, N> tgamma(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return tgamma(x); }, a);
}

using std::trunc;
template <typename T, int N> cl::sycl::vec<T, N> trunc(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return trunc(x); }, a);
}

cl::sycl::half recip(cl::sycl::half a);
float recip(float a);
double recip(double a);
template <typename T, int N> cl::sycl::vec<T, N> recip(cl::sycl::vec<T, N> a) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x) { return recip(x); }, a);
}

cl::sycl::half divide(cl::sycl::half a, cl::sycl::half b);
float divide(float a, float b);
double divide(double a, double b);
template <typename T, int N>
cl::sycl::vec<T, N> divide(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b) {
  return sycl_cts::math::run_func_on_vector<T, T, N>(
      [](T x, T y) { return divide(x, y); }, a, b);
}

} // reference

#endif // __SYCLCTS_UTIL_MATH_REFERENCE_H
