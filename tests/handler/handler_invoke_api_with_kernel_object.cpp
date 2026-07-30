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

TEST_CASE("handler_invoke_api with kernel object", "[handler]") {
  using handler = sycl::handler;

  TestConstants constants;

  auto queue = sycl_cts::util::get_cts_object::queue();
  auto deviceList = queue.get_context().get_devices();

  /* single_task with kernel object */
  if (!is_compiler_available(deviceList)) {
    WARN(
        "online compiler is not available -- skipping test of "
        "single_task with kernel object");
  } else {
    {
      using k_name = kernel_test_class0;
      check_api_call("single_task<kernel_test_class>()", queue,
                     [&](handler& cgh, accessor_t acc) {
                       cgh.single_task<k_name>([=]() {
                         for (size_t i = 0; i < constants.defaultRange[0];
                              ++i) {
                           acc[i] = i;
                         }
                       });
                     });
    }
  }

  /* parallel_for with kernel object */
  if (!is_compiler_available(deviceList)) {
    WARN(
        "online compiler is not available -- skipping test of "
        "parallel_for with kernel object");
  } else {
    {
      using k_name = kernel_test_class1;
      check_api_call("parallel_for(range, kernel) with id", queue,
                     [&](handler& cgh, accessor_t acc) {
                       cgh.parallel_for<k_name>(
                           constants.defaultRange,
                           [=](sycl::id<1> id) { acc[id] = id[0]; });
                     });
    }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    {
      using k_name = kernel_test_class2;
      check_api_call(
          "parallel_for(range, offset, kernel) with id", queue,
          [&](handler& cgh, accessor_t acc) {
            cgh.parallel_for<k_name>(constants.offsetRange, constants.offset,
                                     [=](sycl::id<1> id) { acc[id] = id[0]; });
          },
          constants.offset[0], constants.offsetRange[0]);
    }
#endif

    {
      using k_name = kernel_test_class3;
      check_api_call("parallel_for(range, kernel) with item", queue,
                     [&](handler& cgh, accessor_t acc) {
                       cgh.parallel_for<k_name>(
                           constants.defaultRange,
                           [=](sycl::item<1> item) { acc[item] = item[0]; });
                     });
    }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    {
      using k_name = kernel_test_class4;
      check_api_call(
          "parallel_for(range, offset, kernel) with item", queue,
          [&](handler& cgh, accessor_t acc) {
            cgh.parallel_for<k_name>(
                constants.offsetRange, constants.offset,
                [=](sycl::item<1> item) { acc[item] = item[0]; });
          },
          constants.offset[0], constants.offsetRange[0]);
    }
#endif

    {
      using k_name = kernel_test_class5;
      check_api_call("parallel_for(nd_range, kernel);", queue,
                     [&](handler& cgh, accessor_t acc) {
                       cgh.parallel_for<k_name>(constants.ndRange,
                                                [=](sycl::nd_item<1> ndItem) {
                                                  acc[ndItem.get_global_id()] =
                                                      ndItem.get_global_id(0);
                                                });
                     });
    }
  }
}
