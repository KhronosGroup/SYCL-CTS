/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022-2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/
#include "handler_invoke_api.h"

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
