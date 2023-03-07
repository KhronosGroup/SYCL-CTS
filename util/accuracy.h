/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Support for accuracy check
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_ACCURACY_H
#define __SYCLCTS_UTIL_ACCURACY_H

#include <cmath>
#include <limits>
#include <sycl/sycl.hpp>
/**
 * @brief Provides ulp(x) by definition given in OpenCL 1.2 rev. 19, 7.4
 *        See Jean-Michel Muller "On the definition of ulp (x)", definition 7
 *        Using std functions.
 */
template <typename T> T get_ulp_std(T x) {
  const T inf = std::numeric_limits<T>::infinity();
  const T negative = std::fabs(std::nextafter(x, -inf) - x);
  const T positive = std::fabs(std::nextafter(x, inf) - x);
  return std::fmin(negative, positive);
}
template <>
inline sycl::half get_ulp_std<sycl::half>(sycl::half x) {
  const auto ulp = get_ulp_std<float>(x);
  const float multiplier = 8192.0f;
  // Multiplier is set according to the difference in precision
  return static_cast<sycl::half>(ulp * multiplier);
}
/**
 * @brief Provides ulp(x) by definition given in OpenCL 1.2 rev. 19, 7.4
 *        See Jean-Michel Muller "On the definition of ulp (x)", definition 7
 *        Using sycl functions.
 */
template <typename T> T get_ulp_sycl(T x) {
  const T inf = std::numeric_limits<T>::infinity();
  const T negative = sycl::fabs(sycl::nextafter(x, -inf) - x);
  const T positive = sycl::fabs(sycl::nextafter(x, inf) - x);
  return sycl::fmin(negative, positive);
}
template <>
inline sycl::half get_ulp_sycl<sycl::half>(sycl::half x) {
  const auto ulp = get_ulp_sycl<float>(x);
  const float multiplier = 8192.0f;
  // Multiplier is set according to the difference in precision
  return static_cast<sycl::half>(ulp * multiplier);
}

template <typename T>
bool compare_with_ulp(T actual, T expected, unsigned int ulpsExpected) {
  if (actual == expected)
    return true;
  else if constexpr (!std::is_integral_v<T>) {
    const T difference = static_cast<T>(std::fabs(actual - expected));
    const T differenceExpected = ulpsExpected * get_ulp_sycl(expected);

    return (difference <= differenceExpected)
           // for close to inf cases
           || (actual + differenceExpected == expected) ||
           (expected + differenceExpected == actual) ||
           (actual - differenceExpected == expected) ||
           (expected - differenceExpected == actual);
  }
  return false;
}

#endif // __SYCLCTS_UTIL_ACCURACY_H
