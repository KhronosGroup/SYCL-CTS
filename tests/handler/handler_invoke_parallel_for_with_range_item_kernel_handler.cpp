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

TEST_CASE("handler.parallel_for(range) with item and kernel_handler",
          "[handler]") {
  using handler = sycl::handler;

  TestConstants constants;

  auto queue = sycl_cts::util::get_cts_object::queue();
  auto deviceList = queue.get_context().get_devices();

  /* parallel_for with item and kernel_handler */
  check_api_call(
      "parallel_for(range, lambda) with item and kernel_handler", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<class parallel_for_range_item_kernel_handler_kernel>(
            constants.defaultRange,
            [=](sycl::item<1> item, sycl::kernel_handler kh) {
              parallel_for_range_item_functor_with_kernel_handler<
                  use_offset::no>
                  f(acc);
              f(item, kh);
            });
      });
  check_api_call(
      "parallel_for(range, functor) with item and kernel_handler", queue,
      [&](handler& cgh, accessor_t acc) {
        using functor =
            parallel_for_range_item_functor_with_kernel_handler<use_offset::no>;
        cgh.parallel_for<functor>(constants.defaultRange, functor(acc));
      });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(range, lambda) with item and kernel_handler, no kernel "
      "name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(constants.defaultRange, [=](sycl::item<1> item,
                                                     sycl::kernel_handler kh) {
          parallel_for_range_item_functor_with_kernel_handler<use_offset::no> f(
              acc);
          f(item, kh);
        });
      });
  check_api_call(
      "parallel_for(range, functor) with item and kernel_handler, no kernel "
      "name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(
            constants.defaultRange,
            parallel_for_range_item_functor_with_kernel_handler<use_offset::no>(
                acc));
      });
#endif

  /* parallel_for with item, kernel_handler and offset */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  check_api_call(
      "parallel_for(range, id, lambda) with item and kernel_handler", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<
            class parallel_for_range_offset_item_kernel_handler_kernel>(
            constants.offsetRange, constants.offset,
            [=](sycl::item<1> item, sycl::kernel_handler kh) {
              parallel_for_range_item_functor_with_kernel_handler<
                  use_offset::yes>
                  f(acc);
              f(item, kh);
            });
      },
      constants.offset[0], constants.offsetRange[0]);
  check_api_call(
      "parallel_for(range, id, functor) with item and kernel_handler", queue,
      [&](handler& cgh, accessor_t acc) {
        using functor = parallel_for_range_item_functor_with_kernel_handler<
            use_offset::yes>;
        cgh.parallel_for<functor>(constants.offsetRange, constants.offset,
                                  functor(acc));
      },
      constants.offset[0], constants.offsetRange[0]);
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(range, id, lambda) with item and kernel_handler, no kernel "
      "name",
      queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(constants.offsetRange, constants.offset,
                         [=](sycl::item<1> item, sycl::kernel_handler kh) {
                           parallel_for_range_item_functor_with_kernel_handler<
                               use_offset::yes>
                               f(acc);
                           f(item, kh);
                         });
      },
      constants.offset[0], constants.offsetRange[0]);
  check_api_call(
      "parallel_for(range, id, functor) with item and kernel_handler, no "
      "kernel name",
      queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(constants.offsetRange, constants.offset,
                         parallel_for_range_item_functor_with_kernel_handler<
                             use_offset::yes>(acc));
      },
      constants.offset[0], constants.offsetRange[0]);
#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
}
