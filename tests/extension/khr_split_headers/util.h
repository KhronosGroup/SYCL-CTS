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
// them cannot be queried at all. Probe for the header that provides the macro
// instead, and expose the result as SYCL_CTS_KHR_SPLIT_HEADERS so that the test
// cases of this category can be compiled out (and reported as skipped) when the
// extension is not implemented.
#if __has_include(<sycl/khr/split_headers/version.hpp>)
#include <sycl/khr/split_headers/version.hpp>
#endif

#ifdef SYCL_KHR_SPLIT_HEADERS
#define SYCL_CTS_KHR_SPLIT_HEADERS 1
#else
#define SYCL_CTS_KHR_SPLIT_HEADERS 0
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
