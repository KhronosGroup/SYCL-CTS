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

#if SYCL_CTS_ENABLE_FEATURE_SET_FULL

#include "../common/common.h"

TEST_CASE("static const variable use in kernel", "[full_feature_set]") {
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();

  // constant-initialized, so allowed by full feature set to use in device func
  static const int value = 42;
  int result{};
  {
    sycl::buffer<int, 1> buffer_result{&result, sycl::range<1>{1}};

    // specify runtime index so compiler does not optimize out the address op
    size_t index = 0;
    sycl::buffer<size_t, 1> buffer_index{&index, sycl::range<1>{1}};

    queue
        .submit([&](sycl::handler& cgh) {
          auto acc_result =
              buffer_result.template get_access<sycl::access_mode::write>(cgh);
          auto acc_index =
              buffer_index.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task<class kernel_static_const>([=]() {
            // an object is odr-used when its address is taken
            // full feature set allows this, but not reduced feature set
            const int* ptr = &value;

            // take the value of the pointer so it can be checked
            acc_result[result] = ptr[acc_index[0]];
          });
        })
        .wait_and_throw();
  }

  CHECK(value == result);
}

// a copy of the above test, with a static constexpr value instead
TEST_CASE("static constexpr variable use in kernel", "[full_feature_set]") {
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();

  // constant-initialized, so allowed by full feature set to use in device func
  static constexpr int value = 41;
  int result{};
  {
    sycl::buffer<int, 1> buffer_result{&result, sycl::range<1>{1}};

    // specify runtime index so compiler does not optimize out the address op
    size_t index = 0;
    sycl::buffer<size_t, 1> buffer_index{&index, sycl::range<1>{1}};

    queue
        .submit([&](sycl::handler& cgh) {
          auto acc_result =
              buffer_result.template get_access<sycl::access_mode::write>(cgh);
          auto acc_index =
              buffer_index.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task<class kernel_static_constexpr>([=]() {
            // an object is odr-used when its address is taken
            // full feature set allows this, but not reduced feature set
            const int* ptr = &value;

            // take the value of the pointer so it can be checked
            acc_result[result] = ptr[acc_index[0]];
          });
        })
        .wait_and_throw();
  }

  CHECK(value == result);
}

#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
