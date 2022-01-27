/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_DEVICE_EVAL_H
#define __SYCLCTS_TESTS_COMMON_DEVICE_EVAL_H

#include <sycl/sycl.hpp>

#define DEVICE_EVAL_T(T, expr)                                      \
  ([=] {                                                            \
    sycl::buffer<std::decay_t<T>, 1> result_buf{1};                 \
    sycl_cts::util::get_cts_object::queue()                         \
        .submit([=, &result_buf](sycl::handler& cgh) {              \
          sycl::accessor result{result_buf, cgh, sycl::write_only}; \
          cgh.single_task([=] { result[0] = expr; });               \
        })                                                          \
        .wait_and_throw();                                          \
    sycl::host_accessor acc{result_buf, sycl::read_only};           \
    return acc[0];                                                  \
  })()

/**
 * Evaluates a given expression on the SYCL device and returns the result.
 *
 * Limitations:
 * - Operands must exist in surrounding scope ([=] capture).
 * - No lambda expressions (requires C++20). Use DEVICE_EVAL_T instead.
 */
#define DEVICE_EVAL(expr) DEVICE_EVAL_T(decltype(expr), expr)

#endif  // __SYCLCTS_TESTS_COMMON_DEVICE_EVAL_H
