/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for several reductions in one kernel.
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_REDUCTION_WITH_ALL_FUNCTORS_H
#define __SYCL_CTS_TEST_REDUCTION_WITH_ALL_FUNCTORS_H

#include "../../util/usm_helper.h"
#include "../common/common.h"
#include "reduction_common.h"
#include "reduction_get_lambda.h"
// to use size_t
#include <cstddef>

namespace reduction_with_several_reductions_in_kernel_h {

static constexpr size_t number_elements{2};
template <int TestCase>
class kernel;

/** @brief Provide verification test output value and logging message if result
 *         value does not equal to expected value
 *  @param got Value after test running
 *  @param expected Expected value after test running
 *  @param line Line from which checking was called
 *  @param number_test_case Test case numbed e.g.: 1, 2, 3 e.t.c.
 *  @param log sycl_cts::util::logger class object
 */
void check_output_value(int got, int expected, int line, int number_test_case,
                        sycl_cts::util::logger& log) {
  if (got != expected) {
    std::string fail_message{
        "In test for reducion with different functors in test case number: " +
        std::to_string(number_test_case) + ". Got: " + std::to_string(got) +
        " but expected " + std::to_string(expected)};
    log.fail(fail_message, line);
  }
}

/** @brief Filling provided memory with value by default
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and not using
 *          sycl::property::reduction::initialize_to_identity
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam PtrForMemT The type of the smart poiner to USM allocation
 *  @param functor The functor (plus, multiplies e.t.c) with which the test runs
 *  @param ptr_for_mem Link on the allocated USM
 */
template <typename UsePropertyFlagT, typename FunctorT, typename PtrForMemT>
void fill_mem_for_span(FunctorT functor, PtrForMemT& ptr_for_mem) {
  auto value_for_filling{
      reduction_common::get_init_value_for_reduction<int, FunctorT,
                                                     UsePropertyFlagT>()};
  for (size_t i = 0; i < number_elements; i++) {
    ptr_for_mem.get()[i] = value_for_filling;
  }
}

/** @brief Init USM allocation and fill it with value by default
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UsePropertyFlagT UseCombineFlagT std::integral_constant type that
 *          let switch between using and not using
 *          sycl::property::reduction::initialize_to_identity
 *  @param queue sycl::queue class object
 *  @retval Allocated and filled USM
 */
template <typename FunctorT, typename UsePropertyFlagT>
auto get_ptr_to_variable(sycl::queue& queue) {
  auto variable_for_first_reduction{
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, int>(queue)};
  *variable_for_first_reduction.get() =
      reduction_common::get_init_value_for_reduction<int, FunctorT,
                                                     UsePropertyFlagT>();
  return variable_for_first_reduction;
}

/** @brief Construct lambda for interacting with two reducers using sycl::range
 *         or sycl::nd_range
 *  @tparam RangeT Type range
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @retval Lambda that can interact with all reducers
 */
template <typename RangeT, typename AccessorT>
auto get_lambda_for_2_reductions(AccessorT accessor) {
  if constexpr (std::is_same_v<RangeT, decltype(reduction_common::range)>) {
    return [=](sycl::id<1> idx, auto& reducer_1, auto& reducer_2) {
      reducer_1.combine(accessor[idx]);
      reducer_2.combine(accessor[idx]);
    };
  } else {
    return [=](sycl::nd_item<1> nd_item, auto& reducer_1, auto& reducer_2) {
      reducer_1.combine(accessor[nd_item.get_global_id()]);
      reducer_2.combine(accessor[nd_item.get_global_id()]);
    };
  }
}

/** @brief Construct lambda for interacting with three reducers using
 *         sycl::range or sycl::nd_range
 *  @tparam RangeT Type range
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @retval Lambda that can interact with all reducers
 */
template <typename RangeT, typename AccessorT>
auto get_lambda_for_3_reductions(AccessorT accessor) {
  if constexpr (std::is_same_v<RangeT, decltype(reduction_common::range)>) {
    return [=](sycl::id<1> idx, auto& reducer_1, auto& reducer_2,
               auto& reducer_3) {
      reducer_1.combine(accessor[idx]);
      reducer_2.combine(accessor[idx]);
      reducer_3.combine(accessor[idx]);
    };
  } else {
    return [=](sycl::nd_item<1> nd_item, auto& reducer_1, auto& reducer_2,
               auto& reducer_3) {
      reducer_1.combine(accessor[nd_item.get_global_id()]);
      reducer_2.combine(accessor[nd_item.get_global_id()]);
      reducer_3.combine(accessor[nd_item.get_global_id()]);
    };
  }
}

/** @brief Construct lambda for interacting with four reducers using sycl::range
 *         or sycl::nd_range
 *  @tparam RangeT Type range
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @retval Lambda that can interact with all reducers
 */
template <typename RangeT, typename AccessorT>
auto get_lambda_for_4_reductions(AccessorT accessor) {
  if constexpr (std::is_same_v<RangeT, decltype(reduction_common::range)>) {
    return [=](sycl::id<1> idx, auto& reducer_1, auto& reducer_2,
               auto& reducer_3, auto& reducer_4) {
      reducer_1.combine(accessor[idx]);
      reducer_2.combine(accessor[idx]);
      reducer_3.combine(accessor[idx]);
      reducer_4.combine(accessor[idx]);
    };
  } else {
    return [=](sycl::nd_item<1> nd_item, auto& reducer_1, auto& reducer_2,
               auto& reducer_3, auto& reducer_4) {
      reducer_1.combine(accessor[nd_item.get_global_id()]);
      reducer_2.combine(accessor[nd_item.get_global_id()]);
      reducer_3.combine(accessor[nd_item.get_global_id()]);
      reducer_4.combine(accessor[nd_item.get_global_id()]);
    };
  }
}

/** @brief Construct lambda for interacting with five reducers using sycl::range
 *         or sycl::nd_range
 *  @tparam RangeT Type range
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @retval Lambda that can interact with all reducers
 */
template <typename RangeT, typename AccessorT>
auto get_lambda_for_5_reductions(AccessorT accessor) {
  if constexpr (std::is_same_v<RangeT, decltype(reduction_common::range)>) {
    return [=](sycl::id<1> idx, auto& reducer_1, auto& reducer_2,
               auto& reducer_3, auto& reducer_4, auto& reducer_5) {
      reducer_1.combine(accessor[idx]);
      reducer_2.combine(accessor[idx]);
      reducer_3.combine(accessor[idx]);
      reducer_4.combine(accessor[idx]);
      reducer_5.combine(accessor[idx]);
    };
  } else {
    return [=](sycl::nd_item<1> nd_item, auto& reducer_1, auto& reducer_2,
               auto& reducer_3, auto& reducer_4, auto& reducer_5) {
      reducer_1.combine(accessor[nd_item.get_global_id()]);
      reducer_2.combine(accessor[nd_item.get_global_id()]);
      reducer_3.combine(accessor[nd_item.get_global_id()]);
      reducer_4.combine(accessor[nd_item.get_global_id()]);
      reducer_5.combine(accessor[nd_item.get_global_id()]);
    };
  }
}

/** @brief Construct lambda for interacting with six reducers using sycl::range
 *         or sycl::nd_range
 *  @tparam RangeT Type range
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @retval Lambda that can interact with all reducers
 */
template <typename RangeT, typename AccessorT>
auto get_lambda_for_6_reductions(AccessorT accessor) {
  if constexpr (std::is_same_v<RangeT, decltype(reduction_common::range)>) {
    return
        [=](sycl::id<1> idx, auto& reducer_1, auto& reducer_2, auto& reducer_3,
            auto& reducer_4, auto& reducer_5, auto& reducer_6) {
          reducer_1.combine(accessor[idx]);
          reducer_2.combine(accessor[idx]);
          reducer_3.combine(accessor[idx]);
          reducer_4.combine(accessor[idx]);
          reducer_5.combine(accessor[idx]);
          reducer_6.combine(accessor[idx]);
        };
  } else {
    return [=](sycl::nd_item<1> nd_item, auto& reducer_1, auto& reducer_2,
               auto& reducer_3, auto& reducer_4, auto& reducer_5,
               auto& reducer_6) {
      reducer_1.combine(accessor[nd_item.get_global_id()]);
      reducer_2.combine(accessor[nd_item.get_global_id()]);
      reducer_3.combine(accessor[nd_item.get_global_id()]);
      reducer_4.combine(accessor[nd_item.get_global_id()]);
      reducer_5.combine(accessor[nd_item.get_global_id()]);
      reducer_6.combine(accessor[nd_item.get_global_id()]);
    };
  }
}

/** @brief Run test for two reducers in one sycl::handler.parallel_for
 *  @tparam RangeT Type range
 *  @param range sycl::range or sycl::nd_range class object
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
template <typename RangeT>
void run_test_for_two_reductions(RangeT range, sycl::queue& queue,
                                 sycl_cts::util::logger& log) {
  if (!reduction_common::check_usm_shared_aspect(queue, log)) return;
  constexpr int test_case = 1;
  using functor_1 = sycl::plus<int>;
  using functor_2 = sycl::maximum<int>;

  sycl::buffer<int> initial_buf{reduction_common::get_buffer<int>()};
  int expected_value_for_ptr_to_variable{reduction_common::get_expected_value(
      functor_1(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_1>())};
  int expected_value_for_buffer{reduction_common::get_expected_value(
      functor_2(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_2>())};

  int output_result{reduction_common::get_init_value_for_reduction<
      int, functor_1, reduction_common::without_property>()};
  sycl::buffer<int> output_buffer{&output_result, 1};
  auto ptr_for_variable{
      get_ptr_to_variable<functor_2, reduction_common::without_property>(
          queue)};
  queue.submit([&](sycl::handler& cgh) {
    auto reduction_with_ptr_to_variable{
        sycl::reduction(ptr_for_variable.get(), functor_1())};
    auto reduction_with_buffer{
        sycl::reduction(output_buffer, cgh, functor_2())};
    auto buf_accessor =
        initial_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda{get_lambda_for_2_reductions<RangeT>(buf_accessor)};
    cgh.parallel_for<kernel<test_case>>(range, reduction_with_buffer,
                                        reduction_with_ptr_to_variable, lambda);
  });
  check_output_value(*ptr_for_variable.get(),
                     expected_value_for_ptr_to_variable, __LINE__, test_case,
                     log);
  check_output_value(output_buffer.get_host_access()[0],
                     expected_value_for_buffer, __LINE__, test_case, log);
}

/** @brief Run test for three reducers in one sycl::handler.parallel_for
 *  @tparam RangeT Type range
 *  @param range sycl::range or sycl::nd_range class object
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
template <typename RangeT>
void run_test_for_three_reductions(RangeT range, sycl::queue& queue,
                                   sycl_cts::util::logger& log) {
  if (!reduction_common::check_usm_shared_aspect(queue, log)) return;
  constexpr int test_case = 2;
  using functor_1 = sycl::bit_and<int>;
  using functor_2 = reduction_common::op_without_identity<int>;
  using functor_3 = sycl::maximum<int>;

  sycl::buffer<int> initial_buf{reduction_common::get_buffer<int>()};
  int expected_value_for_ptr_to_variable{reduction_common::get_expected_value(
      functor_1(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_1>())};
  int expected_value_for_buffer{reduction_common::get_expected_value(
      functor_2(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_2>())};
  int expected_value_for_span{reduction_common::get_expected_value(
      functor_3(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_3>())};

  int output_result{reduction_common::get_init_value_for_reduction<
      int, functor_1, reduction_common::with_property>()};
  sycl::buffer<int> output_buffer{&output_result, 1};
  auto ptr_for_variable{
      get_ptr_to_variable<functor_2, reduction_common::with_property>(queue)};
  auto mem_for_span{
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, int>(
          queue, number_elements)};
  fill_mem_for_span<reduction_common::with_property>(functor_3(), mem_for_span);
  queue.submit([&](sycl::handler& cgh) {
    auto reduction_with_ptr_to_variable{
        sycl::reduction(ptr_for_variable.get(), functor_1(),
                        {sycl::property::reduction::initialize_to_identity()})};
    auto reduction_with_buffer{sycl::reduction(
        output_buffer, cgh, reduction_common::identity_value, functor_2(),
        {sycl::property::reduction::initialize_to_identity()})};
    sycl::span<int, number_elements> span(mem_for_span.get(), number_elements);
    auto reduction_with_span{
        sycl::reduction(span, reduction_common::identity_value, functor_3())};
    auto buf_accessor =
        initial_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda{get_lambda_for_3_reductions<RangeT>(buf_accessor)};
    cgh.parallel_for<kernel<test_case>>(range, reduction_with_buffer,
                                        reduction_with_span,
                                        reduction_with_ptr_to_variable, lambda);
  });
  check_output_value(output_buffer.get_host_access()[0],
                     expected_value_for_buffer, __LINE__, test_case, log);
  check_output_value(*ptr_for_variable.get(),
                     expected_value_for_ptr_to_variable, __LINE__, test_case,
                     log);
  for (size_t i = 0; i < number_elements; i++) {
    check_output_value(mem_for_span.get()[i], expected_value_for_span, __LINE__,
                       test_case, log);
  }
}

/** @brief Run test for four reducers in one sycl::handler.parallel_for
 *  @tparam RangeT Type range
 *  @param range sycl::range or sycl::nd_range class object
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
template <typename RangeT>
void run_test_for_four_reductions(RangeT range, sycl::queue& queue,
                                  sycl_cts::util::logger& log) {
  if (!reduction_common::check_usm_shared_aspect(queue, log)) return;
  constexpr int test_case = 3;
  using functor_1 = reduction_common::op_without_identity<int>;
  using functor_2 = sycl::multiplies<int>;
  using functor_3 = reduction_common::op_without_identity<int>;
  using functor_4 = sycl::bit_or<int>;

  sycl::buffer<int> initial_buf{reduction_common::get_buffer<int>()};
  int expected_value_for_buffer{reduction_common::get_expected_value(
      functor_1(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_1>())};
  int expected_value_for_span{reduction_common::get_expected_value(
      functor_2(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_2>())};
  int expected_value_for_ptr_to_variable_1{reduction_common::get_expected_value(
      functor_3(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_3>())};
  int expected_value_for_ptr_to_variable_2{reduction_common::get_expected_value(
      functor_4(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_4>())};

  int output_result{reduction_common::get_init_value_for_reduction<
      int, functor_1, reduction_common::without_property>()};
  sycl::buffer<int> output_buffer{&output_result, 1};
  auto ptr_for_variable_1{
      get_ptr_to_variable<functor_3, reduction_common::with_property>(queue)};
  auto ptr_for_variable_2{
      get_ptr_to_variable<functor_4, reduction_common::with_property>(queue)};
  auto mem_for_span{
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, int>(
          queue, number_elements)};
  fill_mem_for_span<reduction_common::with_property>(functor_2(), mem_for_span);
  queue.submit([&](sycl::handler& cgh) {
    auto reduction_with_buffer{sycl::reduction(
        output_buffer, cgh, reduction_common::identity_value, functor_1())};
    sycl::span<int, number_elements> span(mem_for_span.get(), number_elements);
    auto reduction_with_span{
        sycl::reduction(span, functor_2(),
                        {sycl::property::reduction::initialize_to_identity()})};
    auto reduction_with_ptr_to_variable_1{sycl::reduction(
        ptr_for_variable_1.get(), reduction_common::identity_value, functor_3(),
        {sycl::property::reduction::initialize_to_identity()})};
    auto reduction_with_ptr_to_variable_2{
        sycl::reduction(ptr_for_variable_2.get(), functor_4(),
                        {sycl::property::reduction::initialize_to_identity()})};
    auto buf_accessor =
        initial_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda{get_lambda_for_4_reductions<RangeT>(buf_accessor)};
    cgh.parallel_for<kernel<test_case>>(
        range, reduction_with_buffer, reduction_with_span,
        reduction_with_ptr_to_variable_1, reduction_with_ptr_to_variable_2,
        lambda);
  });
  check_output_value(output_buffer.get_host_access()[0],
                     expected_value_for_buffer, __LINE__, test_case, log);
  check_output_value(*ptr_for_variable_1.get(),
                     expected_value_for_ptr_to_variable_1, __LINE__, test_case,
                     log);
  check_output_value(*ptr_for_variable_2.get(),
                     expected_value_for_ptr_to_variable_2, __LINE__, test_case,
                     log);
  for (size_t i = 0; i < number_elements; i++) {
    check_output_value(mem_for_span.get()[i], expected_value_for_span, __LINE__,
                       test_case, log);
  }
}

/** @brief Run test for five reducers in one sycl::handler.parallel_for
 *  @tparam RangeT Type range
 *  @param range sycl::range or sycl::nd_range class object
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
template <typename RangeT>
void run_test_for_five_reductions(RangeT range, sycl::queue& queue,
                                  sycl_cts::util::logger& log) {
  if (!reduction_common::check_usm_shared_aspect(queue, log)) return;
  constexpr int test_case = 4;
  using functor_1 = sycl::multiplies<int>;
  using functor_2 = sycl::plus<int>;
  using functor_3 = sycl::multiplies<int>;
  using functor_4 = sycl::bit_or<int>;
  using functor_5 = reduction_common::op_without_identity<int>;

  sycl::buffer<int> initial_buf{reduction_common::get_buffer<int>()};
  int expected_value_for_buffer{reduction_common::get_expected_value(
      functor_1(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_1>())};
  int expected_value_for_span_1{reduction_common::get_expected_value(
      functor_2(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_2>())};
  int expected_value_for_span_2{reduction_common::get_expected_value(
      functor_3(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_3>())};
  int expected_value_for_ptr_to_variable_1{reduction_common::get_expected_value(
      functor_4(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_4>())};
  int expected_value_for_ptr_to_variable_2{reduction_common::get_expected_value(
      functor_5(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_5>())};

  int output_result{reduction_common::get_init_value_for_reduction<
      int, functor_1, reduction_common::without_property>()};
  sycl::buffer<int> output_buffer{&output_result, 1};
  auto ptr_for_variable_1{
      get_ptr_to_variable<functor_4, reduction_common::with_property>(queue)};
  auto ptr_for_variable_2{
      get_ptr_to_variable<functor_5, reduction_common::without_property>(
          queue)};
  auto mem_for_span_1{
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, int>(
          queue, number_elements)};
  fill_mem_for_span<reduction_common::without_property>(functor_2(),
                                                        mem_for_span_1);
  auto mem_for_span_2{
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, int>(
          queue, number_elements)};
  fill_mem_for_span<reduction_common::with_property>(functor_3(),
                                                     mem_for_span_2);
  queue.submit([&](sycl::handler& cgh) {
    auto reduction_with_buffer{
        sycl::reduction(output_buffer, cgh, functor_1())};
    sycl::span<int, number_elements> span_1(mem_for_span_1.get(),
                                            number_elements);
    auto reduction_with_span_1{
        sycl::reduction(span_1, reduction_common::identity_value, functor_2())};
    sycl::span<int, number_elements> span_2(mem_for_span_2.get(),
                                            number_elements);
    auto reduction_with_span_2{
        sycl::reduction(span_2, reduction_common::identity_value, functor_3(),
                        {sycl::property::reduction::initialize_to_identity()})};
    auto reduction_with_ptr_to_variable_1{
        sycl::reduction(ptr_for_variable_1.get(), functor_4(),
                        {sycl::property::reduction::initialize_to_identity()})};
    auto reduction_with_ptr_to_variable_2{
        sycl::reduction(ptr_for_variable_2.get(), functor_5())};
    auto buf_accessor =
        initial_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda{get_lambda_for_5_reductions<RangeT>(buf_accessor)};
    cgh.parallel_for<kernel<test_case>>(
        range, reduction_with_buffer, reduction_with_span_1,
        reduction_with_span_2, reduction_with_ptr_to_variable_1,
        reduction_with_ptr_to_variable_2, lambda);
  });
  check_output_value(output_buffer.get_host_access()[0],
                     expected_value_for_buffer, __LINE__, test_case, log);
  check_output_value(*ptr_for_variable_1.get(),
                     expected_value_for_ptr_to_variable_1, __LINE__, test_case,
                     log);
  check_output_value(*ptr_for_variable_2.get(),
                     expected_value_for_ptr_to_variable_2, __LINE__, test_case,
                     log);
  for (size_t i = 0; i < number_elements; i++) {
    check_output_value(mem_for_span_1.get()[i], expected_value_for_span_1,
                       __LINE__, test_case, log);
    check_output_value(mem_for_span_2.get()[i], expected_value_for_span_2,
                       __LINE__, test_case, log);
  }
}

/** @brief Run test for six reducers in one sycl::handler.parallel_for
 *  @tparam RangeT Type range
 *  @param range sycl::range or sycl::nd_range class object
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
template <typename RangeT>
void run_test_for_six_reductions(RangeT range, sycl::queue& queue,
                                 sycl_cts::util::logger& log) {
  if (!reduction_common::check_usm_shared_aspect(queue, log)) return;
  constexpr int test_case = 5;
  using functor_1 = reduction_common::op_without_identity<int>;
  using functor_2 = sycl::plus<int>;
  using functor_3 = sycl::multiplies<int>;
  using functor_4 = sycl::bit_and<int>;
  using functor_5 = sycl::minimum<int>;
  using functor_6 = sycl::maximum<int>;
  constexpr int num_lambdas = 2;

  sycl::buffer<int> initial_buf{reduction_common::get_buffer<int>()};
  int expected_value_for_buffer_1{reduction_common::get_expected_value(
      functor_1(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_1>())};
  int expected_value_for_buffer_2{reduction_common::get_expected_value(
      functor_2(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_2>())};
  int expected_value_for_ptr_to_variable_1{reduction_common::get_expected_value(
      functor_3(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_3>())};
  int expected_value_for_ptr_to_variable_2{reduction_common::get_expected_value(
      functor_4(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_4>())};
  int expected_value_for_span_1{reduction_common::get_expected_value(
      functor_5(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_5>())};
  int expected_value_for_span_2{reduction_common::get_expected_value(
      functor_6(), initial_buf,
      reduction_common::get_init_value_for_expected_value<int, functor_6>())};

  int output_result_1{reduction_common::get_init_value_for_reduction<
      int, functor_1, reduction_common::with_property>()};
  sycl::buffer<int> output_buffer_1{&output_result_1, 1};
  int output_result_2{reduction_common::get_init_value_for_reduction<
      int, functor_2, reduction_common::without_property>()};
  sycl::buffer<int> output_buffer_2{&output_result_2, 1};
  auto ptr_for_variable_1{
      get_ptr_to_variable<functor_3, reduction_common::without_property>(
          queue)};
  auto ptr_for_variable_2{
      get_ptr_to_variable<functor_4, reduction_common::with_property>(queue)};
  auto mem_for_span_1{
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, int>(
          queue, number_elements)};
  fill_mem_for_span<reduction_common::without_property>(functor_5(),
                                                        mem_for_span_1);
  auto mem_for_span_2{
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, int>(
          queue, number_elements)};
  fill_mem_for_span<reduction_common::with_property>(functor_6(),
                                                     mem_for_span_2);
  queue.submit([&](sycl::handler& cgh) {
    auto reduction_with_buffer_1{
        sycl::reduction(output_buffer_1, cgh, functor_1(),
                        {sycl::property::reduction::initialize_to_identity()})};
    auto reduction_with_buffer_2{sycl::reduction(
        output_buffer_2, cgh, reduction_common::identity_value, functor_2())};
    sycl::span<int, number_elements> span_1(mem_for_span_1.get(),
                                            number_elements);
    auto reduction_with_span_1{sycl::reduction(span_1, functor_5())};
    sycl::span<int, number_elements> span_2(mem_for_span_2.get(),
                                            number_elements);
    auto reduction_with_span_2{
        sycl::reduction(span_2, reduction_common::identity_value, functor_6(),
                        {sycl::property::reduction::initialize_to_identity()})};
    auto reduction_with_ptr_to_variable_1{
        sycl::reduction(ptr_for_variable_1.get(), functor_3())};
    auto reduction_with_ptr_to_variable_2{sycl::reduction(
        ptr_for_variable_2.get(), reduction_common::identity_value, functor_4(),
        {sycl::property::reduction::initialize_to_identity()})};
    auto buf_accessor =
        initial_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda{get_lambda_for_6_reductions<RangeT>(buf_accessor)};
    cgh.parallel_for<kernel<test_case>>(
        range, reduction_with_buffer_1, reduction_with_buffer_2,
        reduction_with_span_1, reduction_with_span_2,
        reduction_with_ptr_to_variable_1, reduction_with_ptr_to_variable_2,
        lambda);
  });
  check_output_value(output_buffer_1.get_host_access()[0],
                     expected_value_for_buffer_1, __LINE__, test_case, log);
  check_output_value(output_buffer_2.get_host_access()[0],
                     expected_value_for_buffer_2, __LINE__, test_case, log);
  check_output_value(*ptr_for_variable_1.get(),
                     expected_value_for_ptr_to_variable_1, __LINE__, test_case,
                     log);
  check_output_value(*ptr_for_variable_2.get(),
                     expected_value_for_ptr_to_variable_2, __LINE__, test_case,
                     log);
  for (size_t i = 0; i < number_elements; i++) {
    check_output_value(mem_for_span_1.get()[i], expected_value_for_span_1,
                       __LINE__, test_case, log);
    check_output_value(mem_for_span_2.get()[i], expected_value_for_span_2,
                       __LINE__, test_case, log);
  }
}

/** @brief Run tests from two to six reducers with chosed range type
 *  @tparam RangeT Type range
 *  @param range sycl::range or sycl::nd_range class object
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
template <typename RangeT>
void run_all_tests_for_chosen_range(RangeT range, sycl::queue& queue,
                                    sycl_cts::util::logger& log) {
  run_test_for_two_reductions(range, queue, log);
  run_test_for_three_reductions(range, queue, log);
  run_test_for_four_reductions(range, queue, log);
  run_test_for_five_reductions(range, queue, log);
  run_test_for_six_reductions(range, queue, log);
}

/** @brief Run tests run all test with sycl::range and sycl::nd_range types
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
void run_all_tests(sycl::queue& queue, sycl_cts::util::logger& log) {
  run_all_tests_for_chosen_range(reduction_common::range, queue, log);
  run_all_tests_for_chosen_range(reduction_common::nd_range, queue, log);
}

}  // namespace reduction_with_several_reductions_in_kernel_h

#endif  // __SYCL_CTS_TEST_REDUCTION_WITH_ALL_FUNCTORS_H
