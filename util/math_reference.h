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
uint8_t clamp(const uint8_t a, const uint8_t b, const uint8_t c);
uint16_t clamp(const uint16_t a, const uint16_t b, const uint8_t c);
uint32_t clamp(const uint32_t a, const uint32_t b, const uint8_t c);
uint64_t clamp(const uint64_t a, const uint64_t b, const uint8_t c);
int8_t clamp(const int8_t a, const int8_t b, const uint8_t c);
int16_t clamp(const int16_t a, const int16_t b, const uint8_t c);
int32_t clamp(const int32_t a, const int32_t b, const uint8_t c);
int64_t clamp(const int64_t a, const int64_t b, const uint8_t c);
double clamp(const double a, const double b, const double c);
float clamp(const float a, const float b, const float c);

template <typename T, int N>
cl::sycl::vec<T, N> clamp(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                          cl::sycl::vec<T, N> c) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(
        res, i, clamp(getElement(a, i), getElement(b, i), getElement(c, i)));
  }
  return res;
}

template <typename T, int N>
cl::sycl::vec<T, N> clamp(cl::sycl::vec<T, N> a, T b, T c) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, clamp(getElement(a, i), b, c));
  }
  return res;
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
float smoothstep(float a, float b, float c);
double smoothstep(double a, double b, double c);

template <typename T, int N>
cl::sycl::vec<T, N> smoothstep(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                               cl::sycl::vec<T, N> c) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, smoothstep(getElement(a, i), getElement(b, i),
                                        getElement(c, i)));
  }
  return res;
}

template <typename T, int N>
cl::sycl::vec<T, N> smoothstep(T a, T b, cl::sycl::vec<T, N> c) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, smoothstep(a, b, getElement(c, i)));
  }
  return res;
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
float mix(const float a, const float b, const float c);
double mix(const double a, const double b, const double c);

template <typename T, int N>
cl::sycl::vec<T, N> mix(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b,
                        cl::sycl::vec<T, N> c) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i,
                     mix(getElement(a, i), getElement(b, i), getElement(c, i)));
  }
  return res;
}

template <typename T, int N>
cl::sycl::vec<T, N> mix(cl::sycl::vec<T, N> a, cl::sycl::vec<T, N> b, T c) {
  cl::sycl::vec<T, N> res;
  for (int i = 0; i < N; i++) {
    setElement<T, N>(res, i, mix(getElement(a, i), getElement(b, i), c));
  }
  return res;
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
}

#endif  // __SYCLCTS_UTIL_MATH_REFERENCE_H