/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for tests for reduction with identity parameter
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_REDUCTION_WITH_IDENTITY_PARAM_H
#define __SYCL_CTS_TEST_REDUCTION_WITH_IDENTITY_PARAM_H

#include "../../util/usm_helper.h"
#include "../common/common.h"
#include "reduction_common.h"
#include "reduction_get_lambda.h"

namespace reduction_with_identity_param {
using namespace reduction_common;

static constexpr size_t number_elements = 5;
static constexpr int identity = 0;
static constexpr int initial = 6;
template <typename VariableT, typename RangeT, int TestCase>
class kernel;

template <typename VariableT, typename RangeT>
void run_test_for_value_ptr(RangeT &range_param, sycl::queue &queue) {
  check_usm_shared_aspect(queue);

  sycl::buffer<VariableT> input_buf{range};
  fill_buffer<VariableT>(input_buf);
  VariableT identity_value(identity);
  VariableT initial_value(initial);
  VariableT expected_value = get_expected_value(
      op_without_identity<VariableT>(), input_buf, initial_value);

  auto variable_for_reduction =
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, VariableT>(
          queue);
  *variable_for_reduction = initial_value;
  queue.submit([&](sycl::handler &cgh) {
    auto reduction =
        sycl::reduction(variable_for_reduction.get(), identity_value,
                        op_without_identity<VariableT>());

    auto inputValues =
        input_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda =
        reduction_get_lambda::get_lambda<VariableT, RangeT,
                                         reduction_get_lambda::with_combine,
                                         op_without_identity<VariableT>>(
            inputValues);
    cgh.parallel_for<kernel<VariableT, RangeT, 1>>(range_param, reduction,
                                                   lambda);
  });
  queue.wait_and_throw();
  CHECK(*variable_for_reduction == expected_value);
}

template <typename VariableT, typename RangeT>
void run_test_for_buffer(RangeT &range_param, sycl::queue &queue) {
  sycl::buffer<VariableT> input_buf{range};
  fill_buffer<VariableT>(input_buf);
  VariableT identity_value(identity);
  VariableT initial_value(initial);
  VariableT expected_value = get_expected_value(
      op_without_identity<VariableT>(), input_buf, initial_value);

  sycl::buffer<VariableT> reduction_buffer(&initial_value, 1);

  queue.submit([&](sycl::handler &cgh) {
    auto reduction = sycl::reduction(reduction_buffer, cgh, identity_value,
                                     op_without_identity<VariableT>());

    auto inputValues =
        input_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda =
        reduction_get_lambda::get_lambda<VariableT, RangeT,
                                         reduction_get_lambda::with_combine,
                                         op_without_identity<VariableT>>(
            inputValues);
    cgh.parallel_for<kernel<VariableT, RangeT, 2>>(range_param, reduction,
                                                   lambda);
  });
  queue.wait_and_throw();
  CHECK(reduction_buffer.get_host_access()[0] == expected_value);
}

template <typename VariableT, typename RangeT>
void run_test_for_span(RangeT &range_param, sycl::queue &queue) {
  check_usm_shared_aspect(queue);

  sycl::buffer<VariableT> input_buf{range};
  fill_buffer<VariableT>(input_buf);

  VariableT identity_value(identity);

  std::vector<VariableT> expected_values(number_elements);
  for (int i = 0; i < number_elements; i++) {
    expected_values[i] = get_expected_value(op_without_identity<VariableT>(),
                                            input_buf, VariableT(initial + i));
  }

  auto allocated_memory =
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, VariableT>(
          queue, number_elements);

  for (int i = 0; i < number_elements; i++) {
    allocated_memory.get()[i] = VariableT(initial + i);
  }

  queue.submit([&](sycl::handler &cgh) {
    sycl::span<VariableT, number_elements> span(allocated_memory.get(),
                                                number_elements);
    auto reduction =
        sycl::reduction(span, identity_value, op_without_identity<VariableT>());

    auto inputValues =
        input_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda = reduction_get_lambda::get_lambda_for_span<
        VariableT, RangeT, reduction_get_lambda::with_combine,
        op_without_identity<VariableT>>(inputValues, number_elements);
    cgh.parallel_for<kernel<VariableT, RangeT, 3>>(range_param, reduction,
                                                   lambda);
  });
  queue.wait_and_throw();
  for (int i = 0; i < number_elements; i++) {
    CHECK(allocated_memory.get()[i] == expected_values[i]);
  }
}

template <typename VariableT, typename RangeT>
void run_test_for_value_ptr_property_list(RangeT &range_param,
                                          sycl::queue &queue) {
  check_usm_shared_aspect(queue);
  sycl::buffer<VariableT> input_buf{range};
  fill_buffer<VariableT>(input_buf);
  VariableT identity_value(identity);
  VariableT initial_value(initial);

  VariableT expected_value = get_expected_value(
      op_without_identity<VariableT>(), input_buf, identity_value);

  auto variable_for_reduction =
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, VariableT>(
          queue);
  *variable_for_reduction = initial_value;

  queue.submit([&](sycl::handler &cgh) {
    auto reduction =
        sycl::reduction(variable_for_reduction.get(), identity_value,
                        op_without_identity<VariableT>(),
                        {sycl::property::reduction::initialize_to_identity()});

    auto inputValues =
        input_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda =
        reduction_get_lambda::get_lambda<VariableT, RangeT,
                                         reduction_get_lambda::with_combine,
                                         op_without_identity<VariableT>>(
            inputValues);
    cgh.parallel_for<kernel<VariableT, RangeT, 4>>(range_param, reduction,
                                                   lambda);
  });
  queue.wait_and_throw();
  CHECK(*variable_for_reduction == expected_value);
}

template <typename VariableT, typename RangeT>
void run_test_for_buffer_property_list(RangeT &range_param,
                                       sycl::queue &queue) {
  sycl::buffer<VariableT> input_buf{range};
  fill_buffer<VariableT>(input_buf);
  VariableT identity_value(identity);
  VariableT initial_value(initial);
  VariableT expected_value = get_expected_value(
      op_without_identity<VariableT>(), input_buf, identity_value);

  sycl::buffer<VariableT> reduction_buffer(&initial_value, 1);

  queue.submit([&](sycl::handler &cgh) {
    auto reduction = sycl::reduction(
        reduction_buffer, cgh, identity_value, op_without_identity<VariableT>(),
        {sycl::property::reduction::initialize_to_identity()});

    auto inputValues =
        input_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda =
        reduction_get_lambda::get_lambda<VariableT, RangeT,
                                         reduction_get_lambda::with_combine,
                                         op_without_identity<VariableT>>(
            inputValues);
    cgh.parallel_for<kernel<VariableT, RangeT, 5>>(range_param, reduction,
                                                   lambda);
  });
  queue.wait_and_throw();
  CHECK(reduction_buffer.get_host_access()[0] == expected_value);
}

template <typename VariableT, typename RangeT>
void run_test_for_span_property_list(RangeT &range_param, sycl::queue &queue) {
  check_usm_shared_aspect(queue);
  sycl::buffer<VariableT> input_buf{range};
  fill_buffer<VariableT>(input_buf);

  VariableT identity_value(identity);

  VariableT expected_value = get_expected_value(
      op_without_identity<VariableT>(), input_buf, identity_value);

  auto allocated_memory =
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, VariableT>(
          queue, number_elements);

  queue.submit([&](sycl::handler &cgh) {
    sycl::span<VariableT, number_elements> span(allocated_memory.get(),
                                                number_elements);
    auto reduction =
        sycl::reduction(span, identity_value, op_without_identity<VariableT>(),
                        {sycl::property::reduction::initialize_to_identity()});

    auto inputValues =
        input_buf.template get_access<sycl::access_mode::read>(cgh);
    auto lambda = reduction_get_lambda::get_lambda_for_span<
        VariableT, RangeT, reduction_get_lambda::with_combine,
        op_without_identity<VariableT>>(inputValues, number_elements);
    cgh.parallel_for<kernel<VariableT, RangeT, 6>>(range_param, reduction,
                                                   lambda);
  });
  queue.wait_and_throw();
  for (int i = 0; i < number_elements; i++) {
    CHECK(allocated_memory.get()[i] == expected_value);
  }
}

template <typename VariableT>
struct run_test_for_type {
  void operator()(sycl::queue &queue, const std::string &type_name) {
    run_test_for_value_ptr<VariableT>(range, queue);
    run_test_for_value_ptr<VariableT>(nd_range, queue);
    run_test_for_buffer<VariableT>(range, queue);
    run_test_for_buffer<VariableT>(nd_range, queue);
    run_test_for_span<VariableT>(range, queue);
    run_test_for_span<VariableT>(nd_range, queue);

    run_test_for_value_ptr_property_list<VariableT>(range, queue);
    run_test_for_value_ptr_property_list<VariableT>(nd_range, queue);
    run_test_for_buffer_property_list<VariableT>(range, queue);
    run_test_for_buffer_property_list<VariableT>(nd_range, queue);
    run_test_for_span_property_list<VariableT>(range, queue);
    run_test_for_span_property_list<VariableT>(nd_range, queue);
  }
};

}  // namespace reduction_with_identity_param
#endif  // __SYCL_CTS_TEST_REDUCTION_WITH_IDENTITY_PARAM_H
