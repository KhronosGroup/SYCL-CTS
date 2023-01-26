/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for reduction without property list tests
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_REDUCTION_WITHOUT_IDENTITY_PARAM_COMMON_H
#define __SYCL_CTS_TEST_REDUCTION_WITHOUT_IDENTITY_PARAM_COMMON_H

#include "../common/common.h"
#include "../../util/type_traits.h"
#include "../../util/usm_helper.h"
#include "reduction_common.h"
#include "reduction_get_lambda.h"
#include <cstddef>

namespace reduction_without_identity {

static constexpr size_t number_elements{2};
template <typename VariableT, int TestCase>
class kernel;

constexpr int init_value_without_property_case{99};

using run_test_with_property = std::true_type;
using run_test_without_property = std::false_type;

/** @brief Provide std::string with common error message
 *  @tparam VariableT Variable type from type coverage
 *  @param underlying_type_name String with underlying type
 *  @param got Real value after test running
 *  @param expected Expected value after test running
 *  @retval std::string with common error message
 */
template <typename VariableT>
static std::string get_fail_message(const std::string &underlying_type_name,
                                    VariableT &got, VariableT &expected) {
  std::string fail_message{"Test for the reduction constructor failed for \"" +
                          underlying_type_name +
                          "\" underlying type. got: " + std::to_string(got) +
                          " but expected " + std::to_string(expected)};
  return fail_message;
}

/** @brief Construct reducer with USM pointer to variable
 *  @tparam UsePropertyFlag bool flag that let switch between using and don't
 *          using sycl::property::reduction::initialize_to_identity
 *  @tparam PtrForVariableT Pointer type to variable that used in
 *          sycl::reduction
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @param ptr_for_variable USM pointer to variable that use in sycl::reduction
 *  @param functor The functor (plus, multiplies e.t.c) with which the test runs
 *  @retval Reducer with pointer to variable and chosen functor
 */
template <bool UsePropertyFlag, typename PtrForVariableT, typename FunctorT>
auto get_reduction_for_value_ptr(PtrForVariableT ptr_for_variable,
                                FunctorT functor) {
  if constexpr (UsePropertyFlag) {
    return sycl::reduction(
        ptr_for_variable, functor,
        {sycl::property::reduction::initialize_to_identity()});
  } else {
    return sycl::reduction(ptr_for_variable, functor);
  }
}

/** @brief Construct reducer with sycl::buffer
 *  @tparam UsePropertyFlag bool flag that let switch between using and don't
 *          using sycl::property::reduction::initialize_to_identity
 *  @tparam BufferT sycl::buffer type
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @param buffer sycl::buffer object
 *  @param cgh sycl::handler object
 *  @param functor The functor (plus, multiplies e.t.c) with which the test runs
 *  @retval Reducer with sycl::buffer and chosen functor
 */
template <bool UsePropertyFlag, typename BufferT, typename FunctorT>
auto get_reduction_for_buffer(BufferT &buffer, sycl::handler &cgh,
                              FunctorT functor) {
  if constexpr (UsePropertyFlag) {
    return sycl::reduction(
        buffer, cgh, functor,
        {sycl::property::reduction::initialize_to_identity()});
  } else {
    return sycl::reduction(buffer, cgh, functor);
  }
}

/** @brief Construct reducer with sycl::span
 *  @tparam UsePropertyFlag bool flag that let switch between using and don't
 *          using sycl::property::reduction::initialize_to_identity
 *  @tparam SpanT sycl::span type
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @param span sycl::span object
 *  @param functor The functor (plus, multiplies e.t.c) with which the test runs
 *  @retval Reducer with sycl::span and chosen functor
 */
template <bool UsePropertyFlag, typename SpanT, typename FunctorT>
auto get_reduction_for_span(SpanT &span, FunctorT functor) {
  if constexpr (UsePropertyFlag) {
    return sycl::reduction(
        span, functor, {sycl::property::reduction::initialize_to_identity()});
  } else {
    return sycl::reduction(span, functor);
  }
}

/** @brief Run test for reduction that receive ptr to a variable for construct
 *         reducer object
 *  @tparam VariableT Variable type from type coverage
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam UsePropertyFlag bool flag that let switch between using and don't
 *          using sycl::property::reduction::initialize_to_identity
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @param functor The functor (plus, multiplies e.t.c) with which the test runs
 *  @param range sycl::range or sycl::nd_range type object
 *  @param queue sycl::queue class object
 *  @param type_name a string representing the currently tested type
 */
template <typename VariableT, bool UseCombineFlagT, bool UsePropertyFlag,
          typename FunctorT, typename RangeT>
void run_test_for_value_ptr(FunctorT &functor, RangeT &range,
                            sycl::queue &queue,
                            const std::string &type_name) {
  reduction_common::check_usm_shared_aspect(queue);
  
  sycl::buffer<VariableT> initial_buf{
      reduction_common::get_buffer<VariableT>()};
  VariableT expected_value{reduction_common::get_expected_value(
      functor, initial_buf,
      reduction_common::get_init_value_for_expected_value<VariableT, FunctorT,
                                                          UsePropertyFlag>())};
  auto variable_for_reduction{
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, VariableT>(
          queue)};
  *variable_for_reduction.get() =
      reduction_common::get_init_value_for_reduction<VariableT, FunctorT,
                                                    UsePropertyFlag>();
  queue.submit([&](sycl::handler &cgh) {
    auto reduction{get_reduction_for_value_ptr<UsePropertyFlag>(
        variable_for_reduction.get(), functor)};
    auto lambda{reduction_get_lambda::get_lambda<VariableT, RangeT,
                                                UseCombineFlagT, FunctorT>(
        initial_buf.template get_access<sycl::access_mode::read>(cgh))};

    cgh.parallel_for<kernel<VariableT, 1>>(range, reduction, lambda);
  });

  
  INFO(get_fail_message(type_name, *variable_for_reduction.get(),
                            expected_value) << __LINE__);
  CHECK(*variable_for_reduction.get() == expected_value);
}

/** @brief Run test for reduction that receive sycl::buffer for construct
 *         reducer object
 *  @tparam VariableT Variable type from type coverage
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam UsePropertyFlag bool flag that let switch between using and don't
 *          using sycl::property::reduction::initialize_to_identity
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @param functor The functor (plus, multiplies e.t.c) with which the test runs
 *  @param range sycl::range or sycl::nd_range type object
 *  @param queue sycl::queue class object
 *  @param type_name a string representing the currently tested type
 */
template <typename VariableT, bool UseCombineFlagT, bool UsePropertyFlag,
          typename FunctorT, typename RangeT>
void run_test_for_buffer(FunctorT functor, RangeT range, sycl::queue &queue,
                        const std::string &type_name) {
  sycl::buffer<VariableT> initial_buf{
      reduction_common::get_buffer<VariableT>()};
  VariableT expected_value{reduction_common::get_expected_value(
      functor, initial_buf,
      reduction_common::get_init_value_for_expected_value<VariableT,
                                                          FunctorT>())};
  VariableT output_result{
      reduction_common::get_init_value_for_reduction<VariableT, FunctorT,
                                                    UsePropertyFlag>()};
  sycl::buffer<VariableT> output_buffer{&output_result, 1};

  queue.submit([&](sycl::handler &cgh) {
    auto reduction{
        get_reduction_for_buffer<UsePropertyFlag>(output_buffer, cgh, functor)};
    auto lambda{reduction_get_lambda::get_lambda<VariableT, RangeT,
                                                UseCombineFlagT, FunctorT>(
        initial_buf.template get_access<sycl::access_mode::read>(cgh))};
    cgh.parallel_for<kernel<VariableT, 2>>(range, reduction, lambda);
  });
  
  INFO(get_fail_message(type_name, output_buffer.get_host_access()[0],
                              expected_value) << __LINE__);
  CHECK(output_buffer.get_host_access()[0] == expected_value);
}


/** @brief Run test for reduction that receive sycl::span for construct
 *         reducer object
 *  @tparam VariableT Variable type from type coverage
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam UsePropertyFlag bool flag that let switch between using and don't
 *          using sycl::property::reduction::initialize_to_identity
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @param functor The functor (plus, multiplies e.t.c) with which the test runs
 *  @param range sycl::range or sycl::nd_range type object
 *  @param queue sycl::queue class object
 *  @param type_name a string representing the currently tested type
 */
template <typename VariableT, bool UseCombineFlagT, bool UsePropertyFlag,
          typename FunctorT, typename RangeT>
void run_test_for_span(FunctorT functor, RangeT range, sycl::queue &queue,
                      const std::string &type_name) {
  reduction_common::check_usm_shared_aspect(queue);
  
  sycl::buffer<VariableT> initial_buf{
      reduction_common::get_buffer<VariableT>()};
  VariableT expected_value{reduction_common::get_expected_value(
      functor, initial_buf,
      reduction_common::get_init_value_for_expected_value<VariableT,
                                                          FunctorT>())};
  auto allocated_memory{
      usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, VariableT>(
          queue, number_elements)};
  auto value_for_filling{
      reduction_common::get_init_value_for_reduction<VariableT, FunctorT,
                                                    UsePropertyFlag>()};
  for (size_t i = 0; i < number_elements; i++) {
    allocated_memory.get()[i] = value_for_filling;
  }

  queue.submit([&](sycl::handler &cgh) {
    sycl::span<VariableT, number_elements> span(allocated_memory.get(),
                                                number_elements);
    auto reduction{get_reduction_for_span<UsePropertyFlag>(span, functor)};
    auto lambda{
        reduction_get_lambda::get_lambda_for_span<VariableT, RangeT,
                                                  UseCombineFlagT, FunctorT>(
            initial_buf.template get_access<sycl::access_mode::read>(cgh),
            number_elements)};
    cgh.parallel_for<kernel<VariableT, 3>>(range, reduction, lambda);
  });
  for (size_t i = 0; i < number_elements; i++) {
    INFO(get_fail_message(type_name, allocated_memory.get()[i],
                                expected_value) << __LINE__);
    CHECK(allocated_memory.get()[i] == expected_value);
  }
}

/** @brief Dummy functor that use in type coverage for scalar type variables
 *  @tparam VariableT Variable type from type coverage
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam UsePropertyFlag bool flag that let switch between using and don't
 *          using sycl::property::reduction::initialize_to_identity
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @param functor The functor (plus, multiplies e.t.c) with which the test runs
 *  @param range sycl::range or sycl::nd_range type object
 *  @param queue sycl::queue class object
 *  @param type_name a string representing the currently tested type
 */
template <typename VariableT, bool UseCombineFlagT, bool UsePropertyFlag,
          typename FunctorT, typename RangeT>
void run_test_for_all_reductions_types(FunctorT functor, RangeT &range,
                                      sycl::queue &queue,
                                      const std::string &type_name) {
  if constexpr (is_sycl_floating_point<VariableT>::value &&
                (std::is_same<FunctorT, sycl::bit_and<VariableT>>::value ||
                std::is_same<FunctorT, sycl::bit_or<VariableT>>::value ||
                std::is_same<FunctorT, sycl::bit_xor<VariableT>>::value)) {
    WARN(
        "Test skipped due to floating point variable cannot be used with " +
        std::string(typeid(FunctorT).name()) + " functor");
  } else {
    run_test_for_value_ptr<VariableT, UseCombineFlagT, UsePropertyFlag>(
        functor, range, queue, type_name);
    run_test_for_buffer<VariableT, UseCombineFlagT, UsePropertyFlag>(
        functor, range, queue, type_name);
    run_test_for_span<VariableT, UseCombineFlagT, UsePropertyFlag>(
        functor, range, queue, type_name);
  }
}

/** @brief Dummy functor that use in type coverage for scalar type variables
 *  @tparam VariableT Variable type from type coverage
 *  @tparam UsePropertyFlagT std::integral_constant type that
 *          let switch between using and don't using
 *          sycl::property::reduction::initialize_to_identity
 *  @param queue sycl::queue class object
 *  @param type_name a string representing the currently tested type
 */
template <typename VariableT, typename UsePropertyFlagT>
struct run_tests_for_all_functors {
  template <typename RangeT>
  void operator()(RangeT &range, sycl::queue &queue, const std::string &type_name) {
    constexpr bool use_lambda_without_combine{
        reduction_get_lambda::without_combine};
    constexpr bool use_lambda_with_combine{reduction_get_lambda::with_combine};
    constexpr bool use_property_flag{UsePropertyFlagT::value};

    // for functors that can be called by .combine() or overloaded operator()
    // for functors that can be called by .combine() or overloaded operator()
    // test will be called twice using operator +, *, ^= e.t.c.  and .combine()
    run_test_for_all_reductions_types<VariableT, use_lambda_without_combine,
                                      use_property_flag>(
        sycl::plus<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_with_combine,
                                      use_property_flag>(
        sycl::multiplies<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_without_combine,
                                      use_property_flag>(
        sycl::multiplies<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_with_combine,
                                      use_property_flag>(
        sycl::bit_and<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_without_combine,
                                      use_property_flag>(
        sycl::bit_and<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_with_combine,
                                      use_property_flag>(
        sycl::bit_or<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_without_combine,
                                      use_property_flag>(
        sycl::bit_or<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_with_combine,
                                      use_property_flag>(
        sycl::bit_xor<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_without_combine,
                                      use_property_flag>(
        sycl::bit_xor<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_with_combine,
                                      use_property_flag>(
        sycl::minimum<VariableT>(), range, queue, type_name);
    run_test_for_all_reductions_types<VariableT, use_lambda_with_combine,
                                      use_property_flag>(
        sycl::maximum<VariableT>(), range, queue, type_name);
  }
};
} // namespace reduction_without_identity

#endif  // __SYCL_CTS_TEST_REDUCTION_WITHOUT_IDENTITY_PARAM_COMMON_H
