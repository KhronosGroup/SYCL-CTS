/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for interaction with lambdas
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_REDUCTION_GET_LAMBDA_H
#define __SYCL_CTS_TEST_REDUCTION_GET_LAMBDA_H

#include "reduction_common.h"
#include <functional>
// to use size_t
#include <cstddef>

namespace reduction_get_lambda {

/** @brief Apply provided functor to with provided values
 *  @tparam FunctorT The type of the functor that will be applied
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FirstValueT The first value type that will be used in functor
 *  @tparam SecondValueT The second value type that will be used in functor
 *  @param first_val First value that will be used in functor
 *  @param second_val Second value that will be used in functor
 *  @retval Result value after applying chosen functor to two provided values
 */
template <typename VariableT, typename FunctorT, typename FirstValueT,
          typename SecondValueT>
auto apply_chosen_functor(FirstValueT &first_val, SecondValueT &second_val) {
  if constexpr (std::is_same_v<FunctorT, sycl::multiplies<VariableT>>) {
    first_val *= second_val;
  } else if constexpr (std::is_same_v<FunctorT, sycl::plus<VariableT>>) {
    first_val += second_val;
  } else if constexpr (std::is_same_v<FunctorT, sycl::bit_or<VariableT>>) {
    first_val |= second_val;
  } else if constexpr (std::is_same_v<FunctorT, sycl::bit_and<VariableT>>) {
    first_val &= second_val;
  } else if constexpr (std::is_same_v<FunctorT, sycl::bit_xor<VariableT>>) {
    first_val ^= second_val;
  }
}

template <typename VariableT, typename FunctorT, typename FirstValueT,
          typename SecondValueT>
auto apply_chosen_functor_span(FirstValueT &first_val, SecondValueT &second_val, size_t number_elements) {
  for (size_t i = 0; i < number_elements; i++) {
    if constexpr (std::is_same_v<FunctorT, sycl::multiplies<VariableT>>) {
      first_val[i] *= second_val;
    } else if constexpr (std::is_same_v<FunctorT, sycl::plus<VariableT>>) {
      first_val[i] += second_val;
    } else if constexpr (std::is_same_v<FunctorT, sycl::bit_or<VariableT>>) {
      first_val[i] |= second_val;
    } else if constexpr (std::is_same_v<FunctorT, sycl::bit_and<VariableT>>) {
      first_val[i] &= second_val;
    } else if constexpr (std::is_same_v<FunctorT, sycl::bit_xor<VariableT>>) {
      first_val[i] ^= second_val;
    }
  }
}

/** @brief Construct lambda for interacting with reducer while
 *         sycl::handler.parallel_for uses sycl::range
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UseCombineFlagT std::integral_constant type that lets switch between
 *          calling .combine() function or operator +, *, ^=, etc.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
constexpr bool with_combine{true};
constexpr bool without_combine{false};
template <typename VariableT, typename FunctorT, bool UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_range(AccessorT accessor) {
  if constexpr (UseCombineFlagT) {
    return
        [=](sycl::id<1> idx, auto &reducer) { reducer.combine(accessor[idx]); };
  } else {
    return [=](sycl::id<1> idx, auto &reducer) {
      apply_chosen_functor<VariableT, FunctorT>(reducer, accessor[idx]);
    };
  }
}

/** @brief Construct lambda for interacting with reducer while
 *         sycl::handler.parallel_for uses sycl::nd_range
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UseCombineFlagT std::integral_constant type that lets switch
 *          between calling .combine() function or operator +, *, ^=, etc.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename FunctorT, bool UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_nd_range(AccessorT accessor, size_t number_elements = 0) {
  if constexpr (UseCombineFlagT) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      reducer.combine(accessor[nd_item.get_global_id()]);
    };
  } else {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      apply_chosen_functor<VariableT, FunctorT>(
          reducer, accessor[nd_item.get_global_id()]);
    };
  }
}

/** @brief Construct lambda for interacting with reducer in
 *         sycl::handler.parallel_for
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @tparam UseCombineFlagT std::integral_constant type that lets switch
 *          between calling .combine() function or operator +, *, ^=, etc.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename RangeT, bool UseCombineFlagT,
          typename FunctorT = void, typename AccessorT>
auto get_lambda(AccessorT accessor) {
  if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
    return get_lambda_with_range<VariableT, FunctorT, UseCombineFlagT>(
        accessor);
  } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
    return get_lambda_with_nd_range<VariableT, FunctorT, UseCombineFlagT>(
        accessor);
  }
}

/** @brief Construct lambda for interacting with reducer while
 *         sycl::handler.parallel_for uses sycl::range
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UseCombineFlagT std::integral_constant type that lets switch
 *          between calling .combine() function or operator +, *, ^=, etc.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename FunctorT, bool UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_range_for_span(AccessorT accessor,
                                    size_t number_elements) {
  if constexpr (UseCombineFlagT) {
    return [=](sycl::id<1> idx, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i].combine(accessor[idx]);
      }
    };
  } else {
    return [=](sycl::id<1> idx, auto &reducer) {
        apply_chosen_functor_span<VariableT, FunctorT>(
          reducer, accessor[idx], number_elements);
    };
  }
}

/** @brief Construct lambda for interacting with reducer while
 *         sycl::handler.parallel_for uses sycl::nd_range
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UseCombineFlagT std::integral_constant type that lets switch
 *          between calling .combine() function or operator +, *, ^=, etc.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename FunctorT, bool UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_nd_range_for_span(AccessorT accessor,
                                       size_t number_elements) {
  if constexpr (UseCombineFlagT) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i].combine(accessor[nd_item.get_global_id()]);
      }
    };
  } else {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
        apply_chosen_functor_span<VariableT, FunctorT>(
          reducer, accessor[nd_item.get_global_id()], number_elements);
    };
  }
}

/** @brief Construct lambda for interacting with reducer in
 *         sycl::handler.parallel_for with using sycl::span
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @tparam UseCombineFlagT std::integral_constant type that lets switch
 *          between calling .combine() function or operator +, *, ^=, etc.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename RangeT, bool UseCombineFlagT,
          typename FunctorT = void, typename AccessorT>
auto get_lambda_for_span(AccessorT accessor, size_t number_elements) {
  if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
    return get_lambda_with_range_for_span<VariableT, FunctorT, UseCombineFlagT>(
        accessor, number_elements);
  } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
    return get_lambda_with_nd_range_for_span<VariableT, FunctorT,
                                             UseCombineFlagT>(accessor,
                                                              number_elements);
  }
}

}  // namespace reduction_get_lambda

#endif  // __SYCL_CTS_TEST_REDUCTION_GET_LAMBDA_H
