/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Support for accuracy check
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_ACCURACY_H
#define __SYCLCTS_UTIL_ACCURACY_H

#include <cmath>
#include <limits>

/**
 * @brief Provides ulp(x) by definition given in OpenCL 1.2 rev. 19, 7.4
 *        See Jean-Michel Muller "On the definition of ulp (x)", definition 7
 *        Using sycl functions.
 */
template <typename T> T get_ulp_sycl(T x) {
  const T inf = std::numeric_limits<T>::infinity();
  const T negative = cl::sycl::fabs(cl::sycl::nextafter(x, -1.0f * inf) - x);
  const T positive = cl::sycl::fabs(cl::sycl::nextafter(x, inf) - x);
  return cl::sycl::fmin(negative, positive);
}
template <>
inline cl::sycl::half get_ulp_sycl<cl::sycl::half>(cl::sycl::half x) {
  const auto ulp = get_ulp_sycl<float>(x);
  const float multiplier = 8192.0f;
  // Multiplier is set according to the difference in precision
  return static_cast<cl::sycl::half>(ulp * multiplier);
}

#endif // __SYCLCTS_UTIL_ACCURACY_H
