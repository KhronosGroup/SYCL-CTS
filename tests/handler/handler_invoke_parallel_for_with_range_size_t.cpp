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
#include "handler_invoke_api.hpp"
#include "../common/disabled_for_test_case.h"

DISABLED_FOR_TEST_CASE(DPCPP)
("handler.parallel_for(range) with size_t", "[handler]")({
    using handler = sycl::handler;

TestConstants constants;

    auto queue = sycl_cts::util::get_cts_object::queue();
    auto deviceList = queue.get_context().get_devices();

    /* parallel_for with size_t */
  check_api_call("parallel_for(range, lambda) with size_t", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for<class parallel_for_range_size_t_kernel>(
                       constants.defaultRange, [=](size_t ind) {
                         parallel_for_range_size_t_functor<use_offset::no> f(acc);
                         f(ind);
                       });
                 });
  check_api_call("parallel_for(range, functor) with size_t", queue,
                 [&](handler &cgh, accessor_t acc) {
                   using functor =
                       parallel_for_range_size_t_functor<use_offset::no>;
                   cgh.parallel_for<functor>(constants.defaultRange, functor(acc));
                 });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call("parallel_for(range, lambda) with size_t, no kernel name", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for(constants.defaultRange, [=](size_t ind) {
                     parallel_for_range_size_t_functor<use_offset::no> f(acc);
                     f(ind);
                   });
                 });
  check_api_call("parallel_for(range, functor) with size_t, no kernel name", queue,
                 [&](handler &cgh, accessor_t acc) {
                   cgh.parallel_for(
                       constants.defaultRange,
                       parallel_for_range_size_t_functor<use_offset::no>(acc));
                 });
#endif

  /* parallel_for with size_t and offset */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  check_api_call(
      "parallel_for(range, id, lambda) with size_t", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for<class parallel_for_range_offset_size_t_kernel>(
            constants.offsetRange, constants.offset, [=](size_t ind) {
              parallel_for_range_size_t_functor<use_offset::yes> f(acc);
              f(ind);
            });
      },
      constants.offset[0], constants.offsetRange[0]);
  check_api_call(
      "parallel_for(range, id, functor) with size_t", queue,
      [&](handler &cgh, accessor_t acc) {
        using functor = parallel_for_range_size_t_functor<use_offset::yes>;
        cgh.parallel_for<functor>(constants.offsetRange, constants.offset, functor(acc));
      },
      constants.offset[0], constants.offsetRange[0]);
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(range, id, lambda) with size_t, no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for(constants.offsetRange, constants.offset, [=](size_t ind) {
          parallel_for_range_size_t_functor<use_offset::yes> f(acc);
          f(ind);
        });
      },
      constants.offset[0], constants.offsetRange[0]);
  check_api_call(
      "parallel_for(range, id, functor) with size_t, no kernel name", queue,
      [&](handler &cgh, accessor_t acc) {
        cgh.parallel_for(constants.offsetRange, constants.offset,
                         parallel_for_range_size_t_functor<use_offset::yes>(acc));
      },
      constants.offset[0], constants.offsetRange[0]);
#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
});
