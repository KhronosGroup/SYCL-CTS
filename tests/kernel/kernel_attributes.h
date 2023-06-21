/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
