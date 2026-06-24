/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2026 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_EXTENSION_KHR_INCLUDES_UTIL_H
#define __SYCLCTS_TESTS_EXTENSION_KHR_INCLUDES_UTIL_H

#include <type_traits>

namespace sycl_cts::util {

template <typename T, typename = void>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, std::void_t<decltype(sizeof(T))>> : std::true_type {};

template <typename T>
inline constexpr bool is_complete_v = is_complete<T>::value;

}  // namespace sycl_cts::util

#endif  // __SYCLCTS_TESTS_EXTENSION_KHR_INCLUDES_UTIL_H
