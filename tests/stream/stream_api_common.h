/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common methods for stream tests
//
*******************************************************************************/

#ifndef SYCL_1_2_1_TESTS_STREAM_API_COMMON_H
#define SYCL_1_2_1_TESTS_STREAM_API_COMMON_H

#include "../common/common.h"
#include <type_traits>

/**
 * Function to force const-correctness check for cl::sycl::stream usage with scalar types
 */
template <typename T>
void check_type(const cl::sycl::stream &os, T&& var ) {
  typename std::decay<T>::type const scalar{var};
  os << scalar;
}

/**
 * Function that streams a scalar and vec for each type using the cl::sycl::stream object
 */
template <typename T>
void check_all_vec_dims(const cl::sycl::stream &os, T&& var){
  check_type(os, var);

  const cl::sycl::vec<T, 1> vec1(var);
  os << vec1;
  const cl::sycl::vec<T, 2> vec2(var);
  os << vec2;
  const cl::sycl::vec<T, 3> vec3(var);
  os << vec3;
  const cl::sycl::vec<T, 4> vec4(var);
  os << vec4;
  const cl::sycl::vec<T, 8> vec8(var);
  os << vec8;
  const cl::sycl::vec<T, 16> vec16(var);
  os << vec16;
}

#endif // SYCL_1_2_1_TESTS_STREAM_API_COMMON_H
