/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
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
#include "handler_invoke_api.h"

TEST_CASE("handler.parallel_for(nd_range) with nd_item and kernel_handler",
          "[handler]") {
  using handler = sycl::handler;

  TestConstants constants;

  auto queue = sycl_cts::util::get_cts_object::queue();
  auto deviceList = queue.get_context().get_devices();

  /* parallel_for over nd_range with nd_item */
  check_api_call(
      "parallel_for(nd_range, lambda) with nd_item and kernel_handler", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<class parallel_for_nd_range_nd_item_kern_hand_kernel>(
            constants.ndRange,
            [=](sycl::nd_item<1> ndItem, sycl::kernel_handler kh) {
              parallel_for_nd_range_nd_item_functor_with_kernel_handler f(acc);
              f(ndItem, kh);
            });
      });
  check_api_call(
      "parallel_for(nd_range, functor) with nd_item and kernel_handler", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<
            class parallel_for_nd_range_nd_item_kern_hand_functor_kernel>(
            constants.ndRange,
            parallel_for_nd_range_nd_item_functor_with_kernel_handler(acc));
      });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(nd_range, lambda) with nd_item and kernel_handler, no "
      "kernel name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(constants.ndRange, [=](sycl::nd_item<1> ndItem,
                                                sycl::kernel_handler kh) {
          parallel_for_nd_range_nd_item_functor_with_kernel_handler f(acc);
          f(ndItem, kh);
        });
      });
  check_api_call(
      "parallel_for(nd_range, functor) with nd_item and kernel_handler, no "
      "kernel name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(
            constants.ndRange,
            parallel_for_nd_range_nd_item_functor_with_kernel_handler(acc));
      });
#endif

  /* parallel_for over nd_range with nd_item and offset */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  check_api_call(
      "parallel_for(nd_range, lambda) with nd_item, kernel_handler and offset",
      queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<
            class parallel_for_nd_range_offset_nd_item_kern_hand_kernel>(
            constants.offsetNdRange,
            [=](sycl::nd_item<1> ndItem, sycl::kernel_handler kh) {
              parallel_for_nd_range_nd_item_functor_with_kernel_handler f(acc);
              f(ndItem, kh);
            });
      },
      constants.offset[0], constants.offsetRange[0]);
  check_api_call(
      "parallel_for(nd_range, functor) with nd_item, kernel_handler and offset",
      queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<
            class
            parallel_for_nd_range_offset_nd_item_kern_hand_functor_kernel>(
            constants.offsetNdRange,
            parallel_for_nd_range_nd_item_functor_with_kernel_handler(acc));
      },
      constants.offset[0], constants.offsetRange[0]);
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(nd_range, lambda) with nd_item, kernel_handler and offset, "
      "no kernel name",
      queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(constants.offsetNdRange, [=](sycl::nd_item<1> ndItem,
                                                      sycl::kernel_handler kh) {
          parallel_for_nd_range_nd_item_functor_with_kernel_handler f(acc);
          f(ndItem, kh);
        });
      },
      constants.offset[0], constants.offsetRange[0]);
  check_api_call(
      "parallel_for(nd_range, functor) with nd_item, kernel_handler and "
      "offset, no kernel name",
      queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(
            constants.offsetNdRange,
            parallel_for_nd_range_nd_item_functor_with_kernel_handler(acc));
      },
      constants.offset[0], constants.offsetRange[0]);
#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
}
