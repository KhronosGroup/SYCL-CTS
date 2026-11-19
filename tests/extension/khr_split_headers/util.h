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

// Unlike the other KHR extensions, sycl_khr_split_headers cannot be detected by
// checking its feature-test macro directly: the macro is defined by the very
// headers the extension introduces, so an implementation that does not provide
// them cannot be queried at all. Detect it the way the extension specification
// prescribes instead -- probe for the header, include it if it exists, and then
// test SYCL_KHR_SPLIT_HEADERS, which is only guaranteed to be defined once the
// header has been included. The test cases of this category are guarded on that
// macro so that they are compiled out when the extension is not implemented;
// version.cpp reports the skip for the whole category.
#if __has_include(<sycl/khr/split_headers/version.hpp>)
#include <sycl/khr/split_headers/version.hpp>
#endif

namespace sycl_cts::util {

template <typename T, typename = void>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

template <typename T>
inline constexpr bool is_complete_v = is_complete<T>::value;

}  // namespace sycl_cts::util

#endif  // __SYCLCTS_TESTS_EXTENSION_KHR_SPLIT_HEADERS_UTIL_H
