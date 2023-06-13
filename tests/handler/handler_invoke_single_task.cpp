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

TEST_CASE("handler.", "[handler]") {
  using handler = sycl::handler;

  TestConstants constants;

  auto queue = sycl_cts::util::get_cts_object::queue();
  auto deviceList = queue.get_context().get_devices();

  check_api_call("single_task(lambda)", queue,
                 [&](handler& cgh, accessor_t acc) {
                   cgh.single_task<class single_task_kernel>([=]() {
                     single_task_functor f(acc, constants.defaultRange[0]);
                     f();
                   });
                 });
  check_api_call("single_task(functor)", queue,
                 [&](handler& cgh, accessor_t acc) {
                   cgh.single_task<single_task_functor>(
                       single_task_functor(acc, constants.defaultRange[0]));
                 });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call("single_task(lambda), no kernel name", queue,
                 [&](handler& cgh, accessor_t acc) {
                   cgh.single_task([=]() {
                     single_task_functor f(acc, constants.defaultRange[0]);
                     f();
                   });
                 });
  check_api_call(
      "single_task(functor), no kernel name", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.single_task(single_task_functor(acc, constants.defaultRange[0]));
      });
#endif
}
