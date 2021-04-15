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
#include "./stl.h"
#include <map>

/* for functions that can have undefined results */
template <typename returnT> struct resultRef {
  returnT res;
  std::map<int, bool> undefined;

  template <typename U>
  resultRef(U res_t, std::map<int, bool> und_t)
      : res(res_t), undefined(und_t) {}

  template <typename U>
  resultRef(U res_t, bool und_t)
      : res(res_t), undefined({{0, und_t}}) {}

  template <typename U> resultRef(U res_t) : res(res_t) {}

  template <class U>
  resultRef(const resultRef<U> &other)
      : res(other.res), undefined(other.undefined) {}
};

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
uint8_t abs(const uint8_t);
uint16_t abs(const uint16_t);
uint32_t abs(const uint32_t);
uint64_t abs(const uint64_t);
int8_t abs(const int8_t);
int16_t abs(const int16_t);
int32_t abs(const int32_t);
int64_t abs(const int64_t);

/* absolute difference */
uint8_t abs_diff(const uint8_t a, const uint8_t b);
uint16_t abs_diff(const uint16_t a, const uint16_t b);
uint32_t abs_diff(const uint32_t a, const uint32_t b);
uint64_t abs_diff(const uint64_t a, const uint64_t b);
int8_t abs_diff(const int8_t a, const int8_t b);
int16_t abs_diff(const int16_t a, const int16_t b);
int32_t abs_diff(const int32_t a, const int32_t b);
int64_t abs_diff(const int64_t a, const int64_t b);

/* add with saturation */
uint8_t add_sat(const uint8_t a, const uint8_t b);
uint16_t add_sat(const uint16_t a, const uint16_t b);
uint32_t add_sat(const uint32_t a, const uint32_t b);
uint64_t add_sat(const uint64_t a, const uint64_t b);
int8_t add_sat(const int8_t a, const int8_t b);
int16_t add_sat(const int16_t a, const int16_t b);
int32_t add_sat(const int32_t a, const int32_t b);
int64_t add_sat(const int64_t a, const int64_t b);

/* half add */
uint8_t hadd(const uint8_t a, const uint8_t b);
uint16_t hadd(const uint16_t a, const uint16_t b);
uint32_t hadd(const uint32_t a, const uint32_t b);
uint64_t hadd(const uint64_t a, const uint64_t b);
int8_t hadd(const int8_t a, const int8_t b);
int16_t hadd(const int16_t a, const int16_t b);
int32_t hadd(const int32_t a, const int32_t b);
int64_t hadd(const int64_t a, const int64_t b);

/* round up half add */
uint8_t rhadd(const uint8_t a, const uint8_t b);
uint16_t rhadd(const uint16_t a, const uint16_t b);
uint32_t rhadd(const uint32_t a, const uint32_t b);
uint64_t rhadd(const uint64_t a, const uint64_t b);
int8_t rhadd(const int8_t a, const int8_t b);
int16_t rhadd(const int16_t a, const int16_t b);
int32_t rhadd(const int32_t a, const int32_t b);
int64_t rhadd(const int64_t a, const int64_t b);

/* clamp */
resultRef<uint8_t> clamp(const uint8_t a, const uint8_t b, const uint8_t c);
resultRef<uint16_t> clamp(const uint16_t a, const uint16_t b, const uint8_t c);
resultRef<uint32_t> clamp(const uint32_t a, const uint32_t b, const uint8_t c);
resultRef<uint64_t> clamp(const uint64_t a, const uint64_t b, const uint8_t c);
resultRef<int8_t> clamp(const int8_t a, const int8_t b, const uint8_t c);
resultRef<int16_t> clamp(const int16_t a, const int16_t b, const uint8_t c);
resultRef<int32_t> clamp(const int32_t a, const int32_t b, const uint8_t c);
resultRef<int64_t> clamp(const int64_t a, const int64_t b, const uint8_t c);
resultRef<double> clamp(const double a, const double b, const double c);
resultRef<float> clamp(const float a, const float b, const float c);

template <typename T, int N>
resultRef<cl::sycl::vec<T, N>>
clamp(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b, cl::sycl::vec<T, N> c) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    resultRef<T> element =
        clamp(getElement(a, i), getElement(b, i), getElement(c, i));
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return resultRef<cl::sycl::vec<T, N>>(res, undefined);
}

template <typename T, int N>
resultRef<cl::sycl::vec<T, N>> clamp(cl::sycl::vec<T, N> a, T b, T c) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    resultRef<T> element = clamp(getElement(a, i), b, c);
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return resultRef<cl::sycl::vec<T, N>>(res, undefined);
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
resultRef<float> smoothstep(float a, float b, float c);
resultRef<double> smoothstep(double a, double b, double c);

template <typename T, int N>
resultRef<cl::sycl::vec<T, N>> smoothstep(cl::sycl::vec<T, N> a,
                                          cl::sycl::vec<T, N> b,
                                          cl::sycl::vec<T, N> c) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    resultRef<T> element =
        smoothstep(getElement(a, i), getElement(b, i), getElement(c, i));
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return resultRef<cl::sycl::vec<T, N>>(res, undefined);
}

template <typename T, int N>
resultRef<cl::sycl::vec<T, N>> smoothstep(T a, T b, cl::sycl::vec<T, N> c) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    resultRef<T> element = smoothstep(a, b, getElement(c, i));
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return resultRef<cl::sycl::vec<T, N>>(res, undefined);
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
uint8_t clz(const uint8_t);
uint16_t clz(const uint16_t);
uint32_t clz(const uint32_t);
uint64_t clz(const uint64_t);
int8_t clz(const int8_t);
int16_t clz(const int16_t);
int32_t clz(const int32_t);
int64_t clz(const int64_t);

/* multiply add, get high part */
uint8_t mad_hi(const uint8_t a, const uint8_t b, const uint8_t c);
uint16_t mad_hi(const uint16_t a, const uint16_t b, const uint16_t c);
uint32_t mad_hi(const uint32_t a, const uint32_t b, const uint32_t c);
uint64_t mad_hi(const uint64_t a, const uint64_t b, const uint64_t c);
int8_t mad_hi(const int8_t a, const int8_t b, const int8_t c);
int16_t mad_hi(const int16_t a, const int16_t b, const int16_t c);
int32_t mad_hi(const int32_t a, const int32_t b, const int32_t c);
int64_t mad_hi(const int64_t a, const int64_t b, const int64_t c);

/* multiply add saturate */
uint8_t mad_sat(const uint8_t a, const uint8_t b, const uint8_t c);
uint16_t mad_sat(const uint16_t a, const uint16_t b, const uint8_t c);
uint32_t mad_sat(const uint32_t a, const uint32_t b, const uint8_t c);
uint64_t mad_sat(const uint64_t a, const uint64_t b, const uint8_t c);
int8_t mad_sat(const int8_t a, const int8_t b, const uint8_t c);
int16_t mad_sat(const int16_t a, const int16_t b, const uint8_t c);
int32_t mad_sat(const int32_t a, const int32_t b, const uint8_t c);
int64_t mad_sat(const int64_t a, const int64_t b, const uint8_t c);

/* maximum value */
uint8_t max(const uint8_t a, const uint8_t b);
uint16_t max(const uint16_t a, const uint16_t b);
uint32_t max(const uint32_t a, const uint32_t b);
uint64_t max(const uint64_t a, const uint64_t b);
int8_t max(const int8_t a, const int8_t b);
int16_t max(const int16_t a, const int16_t b);
int32_t max(const int32_t a, const int32_t b);
int64_t max(const int64_t a, const int64_t b);
float max(const float a, const float b);
double max(const double a, const double b);

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
uint8_t min(const uint8_t a, const uint8_t b);
uint16_t min(const uint16_t a, const uint16_t b);
uint32_t min(const uint32_t a, const uint32_t b);
uint64_t min(const uint64_t a, const uint64_t b);
int8_t min(const int8_t a, const int8_t b);
int16_t min(const int16_t a, const int16_t b);
int32_t min(const int32_t a, const int32_t b);
int64_t min(const int64_t a, const int64_t b);
float min(const float a, const float b);
double min(const double a, const double b);

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
resultRef<float> mix(const float a, const float b, const float c);
resultRef<double> mix(const double a, const double b, const double c);

template <typename T, int N>
resultRef<cl::sycl::vec<T, N>> mix(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                                   cl::sycl::vec<T, N> c) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    resultRef<T> element =
        mix(getElement(a, i), getElement(b, i), getElement(c, i));
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return resultRef<cl::sycl::vec<T, N>>(res, undefined);
}

template <typename T, int N>
resultRef<cl::sycl::vec<T, N>> mix(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                                   T c) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    resultRef<T> element = mix(getElement(a, i), getElement(b, i), c);
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return resultRef<cl::sycl::vec<T, N>>(res, undefined);
}

/* multiply and return high part */
uint8_t mul_hi(const uint8_t a, const uint8_t b);
uint16_t mul_hi(const uint16_t a, const uint16_t b);
uint32_t mul_hi(const uint32_t a, const uint32_t b);
uint64_t mul_hi(const uint64_t a, const uint64_t b);
int8_t mul_hi(const int8_t a, const int8_t b);
int16_t mul_hi(const int16_t a, const int16_t b);
int32_t mul_hi(const int32_t a, const int32_t b);
int64_t mul_hi(const int64_t a, const int64_t b);

/* bitwise rotate */
uint8_t rotate(const uint8_t a, const uint8_t b);
uint16_t rotate(const uint16_t a, const uint16_t b);
uint32_t rotate(const uint32_t a, const uint32_t b);
uint64_t rotate(const uint64_t a, const uint64_t b);
int8_t rotate(const int8_t a, const int8_t b);
int16_t rotate(const int16_t a, const int16_t b);
int32_t rotate(const int32_t a, const int32_t b);
int64_t rotate(const int64_t a, const int64_t b);

/* multiply and return high part */
uint8_t rotate(const uint8_t a, const uint8_t b);
uint16_t rotate(const uint16_t a, const uint16_t b);
uint32_t rotate(const uint32_t a, const uint32_t b);
uint64_t rotate(const uint64_t a, const uint64_t b);
int8_t rotate(const int8_t a, const int8_t b);
int16_t rotate(const int16_t a, const int16_t b);
int32_t rotate(const int32_t a, const int32_t b);
int64_t rotate(const int64_t a, const int64_t b);

/* return number of non zero bits in x */
uint8_t popcount(const uint8_t);
uint16_t popcount(const uint16_t);
uint32_t popcount(const uint32_t);
uint64_t popcount(const uint64_t);
int8_t popcount(const int8_t);
int16_t popcount(const int16_t);
int32_t popcount(const int32_t);
int64_t popcount(const int64_t);

/* fast multiply add 24bits */
int32_t mad24(int32_t x, int32_t y, int32_t z);
uint32_t mad24(uint32_t x, uint32_t y, uint32_t z);
cl::sycl::int2 mad24(cl::sycl::int2 x, cl::sycl::int2 y, cl::sycl::int2 z);
cl::sycl::int3 mad24(cl::sycl::int3 x, cl::sycl::int3 y, cl::sycl::int3 z);
cl::sycl::int4 mad24(cl::sycl::int4 x, cl::sycl::int4 y, cl::sycl::int4 z);
cl::sycl::int8 mad24(cl::sycl::int8 x, cl::sycl::int8 y, cl::sycl::int8 z);
cl::sycl::int16 mad24(cl::sycl::int16 x, cl::sycl::int16 y, cl::sycl::int16 z);
cl::sycl::uint2 mad24(cl::sycl::uint2 x, cl::sycl::uint2 y, cl::sycl::uint2 z);
cl::sycl::uint3 mad24(cl::sycl::uint3 x, cl::sycl::uint3 y, cl::sycl::uint3 z);
cl::sycl::uint4 mad24(cl::sycl::uint4 x, cl::sycl::uint4 y, cl::sycl::uint4 z);
cl::sycl::uint8 mad24(cl::sycl::uint8 x, cl::sycl::uint8 y, cl::sycl::uint8 z);
cl::sycl::uint16 mad24(cl::sycl::uint16 x, cl::sycl::uint16 y,
                       cl::sycl::uint16 z);

/* fast multiply 24bits */
int32_t mul24(int32_t x, int32_t y);
uint32_t mul24(uint32_t x, uint32_t y);
cl::sycl::int2 mul24(cl::sycl::int2 x, cl::sycl::int2 y);
cl::sycl::int3 mul24(cl::sycl::int3 x, cl::sycl::int3 y);
cl::sycl::int4 mul24(cl::sycl::int4 x, cl::sycl::int4 y);
cl::sycl::int8 mul24(cl::sycl::int8 x, cl::sycl::int8 y);
cl::sycl::int16 mul24(cl::sycl::int16 x, cl::sycl::int16 y);
cl::sycl::uint2 mul24(cl::sycl::uint2 x, cl::sycl::uint2 y);
cl::sycl::uint3 mul24(cl::sycl::uint3 x, cl::sycl::uint3 y);
cl::sycl::uint4 mul24(cl::sycl::uint4 x, cl::sycl::uint4 y);
cl::sycl::uint8 mul24(cl::sycl::uint8 x, cl::sycl::uint8 y);
cl::sycl::uint16 mul24(cl::sycl::uint16 x, cl::sycl::uint16 y);

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

resultRef<cl::sycl::half> powr(cl::sycl::half a, cl::sycl::half b);
resultRef<float> powr(float a, float b);
resultRef<double> powr(double a, double b);
template <typename T, int N>
resultRef<cl::sycl::vec<T, N>> powr(cl::sycl::vec<T, N> a,
                                    cl::sycl::vec<T, N> b) {
  cl::sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    resultRef<T> element = powr(getElement(a, i), getElement(b, i));
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return resultRef<cl::sycl::vec<T, N>>(res, undefined);
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

// Geometric funcs

cl::sycl::float4 cross(cl::sycl::float4 p0, cl::sycl::float4 p1);
cl::sycl::float3 cross(cl::sycl::float3 p0, cl::sycl::float3 p1);
cl::sycl::double4 cross(cl::sycl::double4 p0, cl::sycl::double4 p1);
cl::sycl::double3 cross(cl::sycl::double3 p0, cl::sycl::double3 p1);

float dot(float p0, float p1);
float dot(cl::sycl::float2 p0, cl::sycl::float2 p1);
float dot(cl::sycl::float3 p0, cl::sycl::float3 p1);
float dot(cl::sycl::float4 p0, cl::sycl::float4 p1);
double dot(double p0, double p1);
double dot(cl::sycl::double2 p0, cl::sycl::double2 p1);
double dot(cl::sycl::double3 p0, cl::sycl::double3 p1);
double dot(cl::sycl::double4 p0, cl::sycl::double4 p1);

float distance(float p0, float p1);
float distance(cl::sycl::float2 p0, cl::sycl::float2 p1);
float distance(cl::sycl::float3 p0, cl::sycl::float3 p1);
float distance(cl::sycl::float4 p0, cl::sycl::float4 p1);
double distance(double p0, double p1);
double distance(cl::sycl::double2 p0, cl::sycl::double2 p1);
double distance(cl::sycl::double3 p0, cl::sycl::double3 p1);
double distance(cl::sycl::double4 p0, cl::sycl::double4 p1);

float length(float p);
float length(cl::sycl::float2 p);
float length(cl::sycl::float3 p);
float length(cl::sycl::float4 p);
double length(double p);
double length(cl::sycl::double2 p);
double length(cl::sycl::double3 p);
double length(cl::sycl::double4 p);

float normalize(float p);
cl::sycl::float2 normalize(cl::sycl::float2 p);
cl::sycl::float3 normalize(cl::sycl::float3 p);
cl::sycl::float4 normalize(cl::sycl::float4 p);
double normalize(double p);
cl::sycl::double2 normalize(cl::sycl::double2 p);
cl::sycl::double3 normalize(cl::sycl::double3 p);
cl::sycl::double4 normalize(cl::sycl::double4 p);

float fast_distance(float p0, float p1);
float fast_distance(cl::sycl::float2 p0, cl::sycl::float2 p1);
float fast_distance(cl::sycl::float3 p0, cl::sycl::float3 p1);
float fast_distance(cl::sycl::float4 p0, cl::sycl::float4 p1);

float fast_length(float p);
float fast_length(cl::sycl::float2 p);
float fast_length(cl::sycl::float3 p);
float fast_length(cl::sycl::float4 p);

float fast_normalize(float p);
cl::sycl::float2 fast_normalize(cl::sycl::float2 p);
cl::sycl::float3 fast_normalize(cl::sycl::float3 p);
cl::sycl::float4 fast_normalize(cl::sycl::float4 p);

} // reference

#endif // __SYCLCTS_UTIL_MATH_REFERENCE_H
