/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_EXTENSION_KHR_SPLIT_HEADERS_UTIL_H
#define __SYCLCTS_TESTS_EXTENSION_KHR_SPLIT_HEADERS_UTIL_H

#include <type_traits>

namespace sycl_cts::util {

template <typename T, typename = void>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

template <typename T>
inline constexpr bool is_complete_v = is_complete<T>::value;

}  // namespace sycl_cts::util

#endif  // __SYCLCTS_TESTS_EXTENSION_KHR_SPLIT_HEADERS_UTIL_H
