/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Common type traits support
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_TYPE_TRAITS_H
#define __SYCLCTS_UTIL_TYPE_TRAITS_H

#include <type_traits>

namespace {
/**
 * @brief Checks whether T is a floating-point sycl type
 */
template <typename T> struct is_cl_float_type {
  static constexpr bool value = std::is_floating_point<T>::value ||
                                std::is_same<cl::sycl::half, T>::value ||
                                std::is_same<cl::sycl::cl_float, T>::value ||
                                std::is_same<cl::sycl::cl_double, T>::value ||
                                std::is_same<cl::sycl::cl_half, T>::value;
};

} // namespace

#endif // __SYCLCTS_UTIL_TYPE_TRAITS_H
