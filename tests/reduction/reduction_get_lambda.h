/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
auto apply_chosen_functor_span(FirstValueT &first_val, SecondValueT &second_val,
                               size_t number_elements) {
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
  return [=](sycl::id<1> idx, auto& reducer) {
    if constexpr (UseCombineFlagT)
      reducer.combine(accessor[idx]);
    else
      apply_chosen_functor<VariableT, FunctorT>(reducer, accessor[idx]);
  };
}

/** @brief Construct lambda for interacting each even work item with reducer
 * while sycl::handler.parallel_for uses sycl::range
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
template <typename VariableT, typename FunctorT, bool UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_range_even_item(AccessorT accessor) {
  return [=](sycl::id<1> idx, auto& reducer) {
    size_t num = idx;
    if (num & 1) {
      if constexpr (UseCombineFlagT)
        reducer.combine(accessor[idx]);
      else
        apply_chosen_functor<VariableT, FunctorT>(reducer, accessor[idx]);
    }
  };
}

/** @brief Construct lambda for interacting no one work item with reducer while
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
template <typename VariableT, typename FunctorT, bool UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_range_no_one_item(AccessorT accessor) {
  return [=](sycl::id<1> idx, auto& reducer) {};
}

/** @brief Construct lambda for interacting each work item twice with reducer
 * while sycl::handler.parallel_for uses sycl::range
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
template <typename VariableT, typename FunctorT, bool UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_range_item_twice(AccessorT accessor) {
  return [=](sycl::id<1> idx, auto& reducer) {
    if constexpr (UseCombineFlagT) {
      reducer.combine(accessor[idx]).combine(accessor[idx]);
    } else {
      apply_chosen_functor<VariableT, FunctorT>(reducer, accessor[idx]);
      apply_chosen_functor<VariableT, FunctorT>(reducer, accessor[idx]);
    }
  };
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
  return [=](sycl::nd_item<1> nd_item, auto& reducer) {
    if constexpr (UseCombineFlagT)
      reducer.combine(accessor[nd_item.get_global_id()]);
    else
      apply_chosen_functor<VariableT, FunctorT>(
          reducer, accessor[nd_item.get_global_id()]);
  };
}

/** @brief Construct lambda for interacting each even work item with reducer
 * while sycl::handler.parallel_for uses sycl::nd_range
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
auto get_lambda_with_nd_range_even_item(AccessorT accessor,
                                        size_t number_elements = 0) {
  return [=](sycl::nd_item<1> nd_item, auto& reducer) {
    size_t num = nd_item.get_global_id();
    if (num & 1) {
      if constexpr (UseCombineFlagT)
        reducer.combine(accessor[nd_item.get_global_id()]);
      else
        apply_chosen_functor<VariableT, FunctorT>(
            reducer, accessor[nd_item.get_global_id()]);
    }
  };
}

/** @brief Construct lambda for interacting no one work item with reducer while
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
auto get_lambda_with_nd_range_no_one_item(AccessorT accessor,
                                          size_t number_elements = 0) {
  return [=](sycl::nd_item<1> nd_item, auto& reducer) {};
}

/** @brief Construct lambda for interacting each work item twice with reducer
 * while sycl::handler.parallel_for uses sycl::nd_range
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
auto get_lambda_with_nd_range_item_twice(AccessorT accessor,
                                         size_t number_elements = 0) {
  return [=](sycl::nd_item<1> nd_item, auto& reducer) {
    if constexpr (UseCombineFlagT) {
      reducer.combine(accessor[nd_item.get_global_id()])
          .combine(accessor[nd_item.get_global_id()]);
    } else {
      apply_chosen_functor<VariableT, FunctorT>(
          reducer, accessor[nd_item.get_global_id()]);
      apply_chosen_functor<VariableT, FunctorT>(
          reducer, accessor[nd_item.get_global_id()]);
    }
  };
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
          typename FunctorT = void, reduction_common::test_case_type TestCaseT,
          typename AccessorT>
auto get_lambda(AccessorT accessor) {
  if constexpr (TestCaseT == reduction_common::test_case_type::each_work_item) {
    if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
      return get_lambda_with_range<VariableT, FunctorT, UseCombineFlagT>(
          accessor);
    } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
      return get_lambda_with_nd_range<VariableT, FunctorT, UseCombineFlagT>(
          accessor);
    }
  } else if constexpr (TestCaseT ==
                       reduction_common::test_case_type::each_even_work_item) {
    if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
      return get_lambda_with_range_even_item<VariableT, FunctorT,
                                             UseCombineFlagT>(accessor);
    } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
      return get_lambda_with_nd_range_even_item<VariableT, FunctorT,
                                                UseCombineFlagT>(accessor);
    }
  } else if constexpr (TestCaseT ==
                       reduction_common::test_case_type::no_one_work_item) {
    if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
      return get_lambda_with_range_no_one_item<VariableT, FunctorT,
                                               UseCombineFlagT>(accessor);
    } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
      return get_lambda_with_nd_range_no_one_item<VariableT, FunctorT,
                                                  UseCombineFlagT>(accessor);
    }
  } else if constexpr (TestCaseT ==
                       reduction_common::test_case_type::each_work_item_twice) {
    if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
      return get_lambda_with_range_item_twice<VariableT, FunctorT,
                                              UseCombineFlagT>(accessor);
    } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
      return get_lambda_with_nd_range_item_twice<VariableT, FunctorT,
                                                 UseCombineFlagT>(accessor);
    }
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
      apply_chosen_functor_span<VariableT, FunctorT>(reducer, accessor[idx],
                                                     number_elements);
    };
  }
}

/** @brief Construct lambda for interacting each even item with reducer while
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
auto get_lambda_with_range_for_span_even_item(AccessorT accessor,
                                              size_t number_elements) {
  if constexpr (UseCombineFlagT) {
    return [=](sycl::id<1> idx, auto& reducer) {
      size_t num = idx;
      if (num & 1) {
        for (size_t i = 0; i < number_elements; i++) {
          reducer[i].combine(accessor[idx]);
        }
      }
    };
  } else {
    return [=](sycl::id<1> idx, auto& reducer) {
      size_t num = idx;
      if (num & 1) {
        apply_chosen_functor_span<VariableT, FunctorT>(reducer, accessor[idx],
                                                       number_elements);
      }
    };
  }
}

/** @brief Construct lambda for interacting no one item with reducer while
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
auto get_lambda_with_range_for_span_no_one_item(AccessorT accessor,
                                                size_t number_elements) {
  return [=](sycl::id<1> idx, auto& reducer) {};
}

/** @brief Construct lambda for interacting each item twice with reducer while
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
auto get_lambda_with_range_for_span_item_twice(AccessorT accessor,
                                               size_t number_elements) {
  if constexpr (UseCombineFlagT) {
    return [=](sycl::id<1> idx, auto& reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i].combine(accessor[idx]).combine(accessor[idx]);
      }
    };
  } else {
    return [=](sycl::id<1> idx, auto& reducer) {
      apply_chosen_functor_span<VariableT, FunctorT>(reducer, accessor[idx],
                                                     number_elements);
      apply_chosen_functor_span<VariableT, FunctorT>(reducer, accessor[idx],
                                                     number_elements);
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

/** @brief Construct lambda for interacting each even item with reducer while
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
auto get_lambda_with_nd_range_for_span_even_item(AccessorT accessor,
                                                 size_t number_elements) {
  if constexpr (UseCombineFlagT) {
    return [=](sycl::nd_item<1> nd_item, auto& reducer) {
      size_t num = nd_item.get_global_id();
      if (num & 1) {
        for (size_t i = 0; i < number_elements; i++) {
          reducer[i].combine(accessor[nd_item.get_global_id()]);
        }
      }
    };
  } else {
    return [=](sycl::nd_item<1> nd_item, auto& reducer) {
      size_t num = nd_item.get_global_id();
      if (num & 1) {
        apply_chosen_functor_span<VariableT, FunctorT>(
            reducer, accessor[nd_item.get_global_id()], number_elements);
      }
    };
  }
}

/** @brief Construct lambda for interacting no one item with reducer while
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
auto get_lambda_with_nd_range_for_span_no_one_item(AccessorT accessor,
                                                   size_t number_elements) {
  return [=](sycl::nd_item<1> nd_item, auto& reducer) {};
}

/** @brief Construct lambda for interacting each item twice with reducer while
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
auto get_lambda_with_nd_range_for_span_item_twice(AccessorT accessor,
                                                  size_t number_elements) {
  if constexpr (UseCombineFlagT) {
    return [=](sycl::nd_item<1> nd_item, auto& reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i]
            .combine(accessor[nd_item.get_global_id()])
            .combine(accessor[nd_item.get_global_id()]);
      }
    };
  } else {
    return [=](sycl::nd_item<1> nd_item, auto& reducer) {
      apply_chosen_functor_span<VariableT, FunctorT>(
          reducer, accessor[nd_item.get_global_id()], number_elements);
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
          typename FunctorT = void, reduction_common::test_case_type TestCaseT,
          typename AccessorT>
auto get_lambda_for_span(AccessorT accessor, size_t number_elements) {
  if constexpr (TestCaseT == reduction_common::test_case_type::each_work_item) {
    if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
      return get_lambda_with_range_for_span<VariableT, FunctorT,
                                            UseCombineFlagT>(accessor,
                                                             number_elements);
    } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
      return get_lambda_with_nd_range_for_span<VariableT, FunctorT,
                                               UseCombineFlagT>(
          accessor, number_elements);
    }
  } else if constexpr (TestCaseT ==
                       reduction_common::test_case_type::each_even_work_item) {
    if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
      return get_lambda_with_range_for_span_even_item<VariableT, FunctorT,
                                                      UseCombineFlagT>(
          accessor, number_elements);
    } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
      return get_lambda_with_nd_range_for_span_even_item<VariableT, FunctorT,
                                                         UseCombineFlagT>(
          accessor, number_elements);
    }
  } else if constexpr (TestCaseT ==
                       reduction_common::test_case_type::no_one_work_item) {
    if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
      return get_lambda_with_range_for_span_no_one_item<VariableT, FunctorT,
                                                        UseCombineFlagT>(
          accessor, number_elements);
    } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
      return get_lambda_with_nd_range_for_span_no_one_item<VariableT, FunctorT,
                                                           UseCombineFlagT>(
          accessor, number_elements);
    }
  } else if constexpr (TestCaseT ==
                       reduction_common::test_case_type::each_work_item_twice) {
    if constexpr (std::is_same_v<RangeT, sycl::range<1>>) {
      return get_lambda_with_range_for_span_item_twice<VariableT, FunctorT,
                                                       UseCombineFlagT>(
          accessor, number_elements);
    } else if constexpr (std::is_same_v<RangeT, sycl::nd_range<1>>) {
      return get_lambda_with_nd_range_for_span_item_twice<VariableT, FunctorT,
                                                          UseCombineFlagT>(
          accessor, number_elements);
    }
  }
}

}  // namespace reduction_get_lambda

#endif  // __SYCL_CTS_TEST_REDUCTION_GET_LAMBDA_H
