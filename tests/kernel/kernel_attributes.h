/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"

namespace kernel_attributes {

template <typename F>
struct kernel_functor_st;
template <typename F>
struct kernel_functor;
template <typename F>
struct kernel_functor_wg;

template <class vec_t>
struct kernel_separate_lambda_st;
template <int dims, int size, typename T = int>
struct kernel_separate_lambda;
template <int dims, int size, typename T = int>
struct kernel_separate_lambda_wg;

template <class vec_t>
struct kernel_lambda_st;
template <int dims, int size, typename T = int>
struct kernel_lambda;
template <int dims, int size, typename T = int>
struct kernel_lambda_wg;

template <int dims>
inline constexpr int expected_val() {
  return 40 + dims;
}

const sycl::range<1> range(1);

}  // namespace kernel_attributes
