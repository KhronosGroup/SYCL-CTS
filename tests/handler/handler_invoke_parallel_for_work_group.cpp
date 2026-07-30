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

TEST_CASE("handler.parallel_for_work_group() test", "[handler]") {
  using handler = sycl::handler;

  TestConstants constants;

  auto queue = sycl_cts::util::get_cts_object::queue();
  auto deviceList = queue.get_context().get_devices();

  /* parallel_for_work_group (range) */
  check_api_call("parallel_for_work_group(range, lambda)", queue,
                 [&](handler& cgh, accessor_t acc) {
                   cgh.parallel_for_work_group<
                       class parallel_for_work_group_1range_kernel>(
                       constants.defaultRange, [=](sycl::group<1> group) {
                         // cannot instantiate functor as parallel_for_work_item
                         // must be invoked from parallel_for_work_group context
                         PARALLEL_FOR_WORK_GROUP_DYNAMIC_FUNCTOR_BODY;
                       });
                 });
  check_api_call(
      "parallel_for_work_group(range, functor)", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group<parallel_for_work_group_dynamic_functor>(
            constants.defaultRange,
            parallel_for_work_group_dynamic_functor(acc));
      });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call("parallel_for_work_group(range, lambda), no kernel name",
                 queue, [&](handler& cgh, accessor_t acc) {
                   cgh.parallel_for_work_group(
                       constants.defaultRange, [=](sycl::group<1> group) {
                         // cannot instantiate functor as parallel_for_work_item
                         // must be invoked from parallel_for_work_group context
                         PARALLEL_FOR_WORK_GROUP_DYNAMIC_FUNCTOR_BODY;
                       });
                 });
  check_api_call("parallel_for_work_group(range, functor), no kernel name",
                 queue, [&](handler& cgh, accessor_t acc) {
                   cgh.parallel_for_work_group(
                       constants.defaultRange,
                       parallel_for_work_group_dynamic_functor(acc));
                 });
#endif

  /* parallel_for_work_group (range) with kernel_handler*/
  check_api_call(
      "parallel_for_work_group(range, lambda) with kernel handler", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group<
            class parallel_for_work_group_1range_kernel_with_kern_handler>(
            constants.defaultRange,
            [=](sycl::group<1> group, sycl::kernel_handler kh) {
              kh.get_specialization_constant<SpecName>();
              // cannot instantiate functor as parallel_for_work_item
              // must be invoked from parallel_for_work_group context
              PARALLEL_FOR_WORK_GROUP_DYNAMIC_FUNCTOR_BODY;
            });
      });
  check_api_call(
      "parallel_for_work_group(range, functor) with kernel handler", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group<
            parallel_for_work_group_dynamic_with_kern_handler_functor>(
            constants.defaultRange,
            parallel_for_work_group_dynamic_with_kern_handler_functor(acc));
      });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for_work_group(range, lambda) with kernel handler, no kernel "
      "name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group(
            constants.defaultRange,
            [=](sycl::group<1> group, sycl::kernel_handler kh) {
              kh.get_specialization_constant<SpecName>();
              // cannot instantiate functor as parallel_for_work_item
              // must be invoked from parallel_for_work_group context
              PARALLEL_FOR_WORK_GROUP_DYNAMIC_FUNCTOR_BODY;
            });
      });
  check_api_call(
      "parallel_for_work_group(range, functor) with kernel functor, no kernel "
      "name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group(
            constants.defaultRange,
            parallel_for_work_group_dynamic_with_kern_handler_functor(acc));
      });
#endif
  /* parallel_for_work_group (range, range) */
  check_api_call("parallel_for_work_group(range, range, lambda)", queue,
                 [&](handler& cgh, accessor_t acc) {
                   cgh.parallel_for_work_group<
                       class parallel_for_work_group_2range_kernel>(
                       constants.numWorkGroups, constants.workGroupSize,
                       [=](sycl::group<1> group) {
                         // cannot instantiate functor as parallel_for_work_item
                         // must be invoked from parallel_for_work_group context
                         PARALLEL_FOR_WORK_GROUP_FIXED_FUNCTOR_BODY;
                       });
                 });
  check_api_call(
      "parallel_for_work_group(range, range, functor)", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group<parallel_for_work_group_fixed_functor>(
            constants.numWorkGroups, constants.workGroupSize,
            parallel_for_work_group_fixed_functor(acc));
      });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for_work_group(range, range, lambda), no kernel name", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group(
            constants.numWorkGroups, constants.workGroupSize,
            [=](sycl::group<1> group) {
              // cannot instantiate functor as parallel_for_work_item
              // must be invoked from parallel_for_work_group context
              PARALLEL_FOR_WORK_GROUP_FIXED_FUNCTOR_BODY;
            });
      });
  check_api_call(
      "parallel_for_work_group(range, range, functor), no kernel name", queue,
      [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group(constants.numWorkGroups,
                                    constants.workGroupSize,
                                    parallel_for_work_group_fixed_functor(acc));
      });
#endif

  /* parallel_for_work_group (range, range) with kernel handler */
  check_api_call(
      "parallel_for_work_group(range, range, lambda) with kernel handler",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group<
            class parallel_for_work_group_2range_kernel_with_kern_handler>(
            constants.numWorkGroups, constants.workGroupSize,
            [=](sycl::group<1> group, sycl::kernel_handler kh) {
              kh.get_specialization_constant<SpecName>();
              // cannot instantiate functor as parallel_for_work_item
              // must be invoked from parallel_for_work_group context
              PARALLEL_FOR_WORK_GROUP_FIXED_FUNCTOR_BODY;
            });
      });
  check_api_call(
      "parallel_for_work_group(range, range, functor) with kernel handler",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group<
            parallel_for_work_group_fixed_with_kern_handler_functor>(
            constants.numWorkGroups, constants.workGroupSize,
            parallel_for_work_group_fixed_with_kern_handler_functor(acc));
      });
#if SYCL_CTS_ENABLE_FEATURE_SET_FULL
  check_api_call(
      "parallel_for_work_group(range, range, lambda) with kernel handler, no "
      "kernel name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group(
            constants.numWorkGroups, constants.workGroupSize,
            [=](sycl::group<1> group, sycl::kernel_handler kh) {
              kh.get_specialization_constant<SpecName>();
              // cannot instantiate functor as parallel_for_work_item
              // must be invoked from parallel_for_work_group context
              PARALLEL_FOR_WORK_GROUP_FIXED_FUNCTOR_BODY;
            });
      });
  check_api_call(
      "parallel_for_work_group(range, range, functor) with kernel handler, no "
      "kernel name",
      queue, [&](handler& cgh, accessor_t acc) {
        cgh.parallel_for_work_group(
            constants.numWorkGroups, constants.workGroupSize,
            parallel_for_work_group_fixed_with_kern_handler_functor(acc));
      });
#endif
}
