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

TEST_CASE("handler.parallel_for(nd_range) with custom nd_item", "[handler]") {
  using handler = sycl::handler;

  TestConstants constants;

  auto queue = sycl_cts::util::get_cts_object::queue();
  auto deviceList = queue.get_context().get_devices();

  /* parallel_for over nd_range without with custom nd_item */
  check_api_call(
      "parallel_for(nd_range, lambda) with custom nd_item", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<class parallel_for_nd_range_custom_nd_item_kernel>(
            constants.ndRange, [=](CustomNdItem ndItem) {
              parallel_for_nd_range_custom_nd_item_functor f(acc);
              f(ndItem);
            });
      });
  check_api_call(
      "parallel_for(nd_range, functor) with custom nd_item", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<
            class parallel_for_nd_range_custom_nd_item_functor_kernel>(
            constants.ndRange,
            parallel_for_nd_range_custom_nd_item_functor(acc));
      });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(nd_range, lambda) with custom nd_item, no kernel name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(constants.ndRange, [=](CustomNdItem ndItem) {
          parallel_for_nd_range_custom_nd_item_functor f(acc);
          f(ndItem);
        });
      });
  check_api_call(
      "parallel_for(nd_range, functor) with custom nd_item, no kernel name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(constants.ndRange,
                         parallel_for_nd_range_custom_nd_item_functor(acc));
      });
#endif

  /* parallel_for over nd_range with custom nd_item and offset */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  check_api_call(
      "parallel_for(nd_range, lambda) with custom nd_item and offset", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<
            class parallel_for_nd_range_offset_custom_nd_item_kernel>(
            constants.offsetNdRange, [=](CustomNdItem ndItem) {
              parallel_for_nd_range_custom_nd_item_functor f(acc);
              f(ndItem);
            });
      },
      constants.offset[0], constants.offsetRange[0]);
  check_api_call(
      "parallel_for(nd_range, functor) with custom nd_item and offset", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for<
            class parallel_for_nd_range_offset_custom_nd_item_functor_kernel>(
            constants.offsetNdRange,
            parallel_for_nd_range_custom_nd_item_functor(acc));
      },
      constants.offset[0], constants.offsetRange[0]);
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for(nd_range, lambda) with custom nd_item and offset, no "
      "kernel name",
      queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(constants.offsetNdRange, [=](CustomNdItem ndItem) {
          parallel_for_nd_range_custom_nd_item_functor f(acc);
          f(ndItem);
        });
      },
      constants.offset[0], constants.offsetRange[0]);
  check_api_call(
      "parallel_for(nd_range, functor) with custom nd_item and offset, no "
      "kernel name",
      queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for(constants.offsetNdRange,
                         parallel_for_nd_range_custom_nd_item_functor(acc));
      },
      constants.offset[0], constants.offsetRange[0]);
#endif  // SYCL_CTS_ENABLE_FEATURE_SET_FULL
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
}
