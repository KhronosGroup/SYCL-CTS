/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#ifndef __SYCLCTS_TESTS_COMMON_DEVICE_EVAL_H
#define __SYCLCTS_TESTS_COMMON_DEVICE_EVAL_H

#include <sycl/sycl.hpp>

/** Variadic parameter is the kernel name. */
#define DEVICE_EVAL_T(T, expr, ...)                                 \
  ([=] {                                                            \
    sycl::buffer<std::decay_t<T>, 1> result_buf{1};                 \
    sycl_cts::util::get_cts_object::queue()                         \
        .submit([=, &result_buf](sycl::handler& cgh) {              \
          sycl::accessor result{result_buf, cgh, sycl::write_only}; \
          cgh.single_task<__VA_ARGS__>([=] { result[0] = expr; });  \
        })                                                          \
        .wait_and_throw();                                          \
    sycl::host_accessor acc{result_buf, sycl::read_only};           \
    return acc[0];                                                  \
  })()

/**
 * Evaluates a given expression on the SYCL device and returns the result.
 * A unique kernel name must be passed in via variadic arguments.
 *
 * Limitations:
 * - Operands must exist in surrounding scope ([=] capture).
 * - No lambda expressions (requires C++20). Use DEVICE_EVAL_T instead.
 */
#define DEVICE_EVAL(expr, ...) DEVICE_EVAL_T(decltype(expr), expr, __VA_ARGS__)

#endif  // __SYCLCTS_TESTS_COMMON_DEVICE_EVAL_H
