/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2025 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SCALARS_HELPERS_H
#define __SYCLCTS_TESTS_SCALARS_HELPERS_H

#include <sycl/sycl.hpp>

#include "../common/macros.h"

#include <string>
#include <type_traits>

/**
 * @brief Helper function to see if a type is of the wrong sign
 */
template <typename T>
bool check_type_sign(bool expected_sign) {
  return (std::is_signed<T>::value == expected_sign);
}

#if SYCL_CTS_ENABLE_HALF_TESTS
/**
 * @brief Helper function to see if sycl::half is of the wrong sign
 */
template <>
inline bool check_type_sign<sycl::half>(bool expected_sign) {
  bool is_signed = sycl::half(1) > sycl::half(-1);
  return is_signed == expected_sign;
}
#endif

/**
 * @brief Helper function to see if a type is of the wrong size
 */
template <typename T>
bool check_type_min_size(size_t minSize) {
  return !(sizeof(T) < minSize);
}

/**
 * @brief Helper function to log a failure if a type is of the wrong size or
 * sign
 */
template <typename T>
void check_type_min_size_sign_log(size_t minSize, bool expected_sign,
                                  std::string typeName) {
  if (!check_type_min_size<T>(minSize)) {
    FAIL(std::string(
             "The following host type does not have the correct size: ") +
         typeName);
  }
  if (!check_type_sign<T>(expected_sign)) {
    FAIL(std::string(
             "The following host type does not have the correct sign: ") +
         typeName);
  }
}

/**
 * @deprecated Use overload without logger.
 */
template <typename T>
void check_type_min_size_sign_log(sycl_cts::util::logger& log, size_t minSize,
                                  bool expected_sign, std::string typeName) {
  check_type_min_size_sign_log<T>(minSize, expected_sign, typeName);
}

#endif  // __SYCLCTS_TESTS_SCALARS_HELPERS_H
