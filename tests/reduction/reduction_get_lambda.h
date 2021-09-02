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

/** @brief Construct lambda for interacting with reducer while
 *         sycl::handler.parallel_for used sycl::range
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
using with_combine = std::true_type;
using without_combine = std::false_type;
template <typename VariableT, typename FunctorT, typename UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_range(AccessorT accessor) {
  if constexpr (UseCombineFlagT::value) {
    return
        [=](sycl::id<1> idx, auto &reducer) { reducer.combine(accessor[idx]); };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::multiplies<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) { reducer *= accessor[idx]; };
  } else if constexpr (std::is_same<FunctorT, sycl::plus<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) { reducer += accessor[idx]; };
  } else if constexpr (std::is_same<FunctorT, sycl::bit_or<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) { reducer |= accessor[idx]; };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::bit_and<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) { reducer &= accessor[idx]; };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::bit_xor<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) { reducer ^= accessor[idx]; };
  }
}

/** @brief Construct lambda for interacting with reducer while
 *         sycl::handler.parallel_for used sycl::nd_range
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename FunctorT, typename UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_nd_range(AccessorT accessor, size_t number_elements = 0) {
  if constexpr (UseCombineFlagT::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      reducer.combine(accessor[nd_item.get_global_id()]);
    };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::multiplies<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      reducer *= accessor[nd_item.get_global_id()];
    };
  } else if constexpr (std::is_same<FunctorT, sycl::plus<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      reducer += accessor[nd_item.get_global_id()];
    };
  } else if constexpr (std::is_same<FunctorT, sycl::bit_or<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      reducer |= accessor[nd_item.get_global_id()];
    };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::bit_and<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      reducer &= accessor[nd_item.get_global_id()];
    };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::bit_xor<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      reducer ^= accessor[nd_item.get_global_id()];
    };
  }
}

/** @brief Construct lambda for interacting with reducer in
 *         sycl::handler.parallel_for
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename RangeT, typename UseCombineFlagT,
          typename FunctorT = void, typename AccessorT>
auto get_lambda(AccessorT accessor) {
  if constexpr (std::is_same<RangeT, sycl::range<1>>::value) {
    return get_lambda_with_range<VariableT, FunctorT, UseCombineFlagT>(
        accessor);
  } else if constexpr (std::is_same<RangeT, sycl::nd_range<1>>::value) {
    return get_lambda_with_nd_range<VariableT, FunctorT, UseCombineFlagT>(
        accessor);
  }
}

/** @brief Construct lambda for interacting with reducer while
 *         sycl::handler.parallel_for used sycl::range
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename FunctorT, typename UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_range_for_span(AccessorT accessor,
                                    size_t number_elements) {
  if constexpr (UseCombineFlagT::value) {
    return [=](sycl::id<1> idx, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i].combine(accessor[idx]);
      }
    };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::multiplies<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] *= accessor[idx];
      }
    };
  } else if constexpr (std::is_same<FunctorT, sycl::plus<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] += accessor[idx];
      }
    };
  } else if constexpr (std::is_same<FunctorT, sycl::bit_or<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] |= accessor[idx];
      }
    };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::bit_and<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] &= accessor[idx];
      }
    };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::bit_xor<VariableT>>::value) {
    return [=](sycl::id<1> idx, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] ^= accessor[idx];
      }
    };
  }
}

/** @brief Construct lambda for interacting with reducer while
 *         sycl::handler.parallel_for used sycl::nd_range
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename FunctorT, typename UseCombineFlagT,
          typename AccessorT>
auto get_lambda_with_nd_range_for_span(AccessorT accessor,
                                       size_t number_elements) {
  if constexpr (UseCombineFlagT::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i].combine(accessor[nd_item.get_global_id()]);
      }
    };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::multiplies<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] *= accessor[nd_item.get_global_id()];
      }
    };
  } else if constexpr (std::is_same<FunctorT, sycl::plus<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] += accessor[nd_item.get_global_id()];
      }
    };
  } else if constexpr (std::is_same<FunctorT, sycl::bit_or<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] |= accessor[nd_item.get_global_id()];
      }
    };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::bit_and<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] &= accessor[nd_item.get_global_id()];
      }
    };
  } else if constexpr (std::is_same<FunctorT,
                                    sycl::bit_xor<VariableT>>::value) {
    return [=](sycl::nd_item<1> nd_item, auto &reducer) {
      for (size_t i = 0; i < number_elements; i++) {
        reducer[i] ^= accessor[nd_item.get_global_id()];
      }
    };
  }
}

/** @brief Construct lambda for interacting with reducer in
 *         sycl::handler.parallel_for with using sycl::span
 *  @tparam VariableT Variable type from type coverage
 *  @tparam FunctorT The type of the functor with which the test runs
 *  @tparam RangeT sycl::range or sycl::nd_range type
 *  @tparam UseCombineFlagT std::integral_constant type that let switch between
 *          calling .combine() function or operator +, *, ^= e.t.c.
 *  @tparam AccessorT buffer accessor type
 *  @param accessor Accessor to the buffer
 *  @param number_elements Number elements in sycl::span, used only when
 *         constructing reducer with span
 *  @retval Lambda with chosen operator
 */
template <typename VariableT, typename RangeT, typename UseCombineFlagT,
          typename FunctorT = void, typename AccessorT>
auto get_lambda_for_span(AccessorT accessor, size_t number_elements) {
  if constexpr (std::is_same<RangeT, sycl::range<1>>::value) {
    return get_lambda_with_range_for_span<VariableT, FunctorT, UseCombineFlagT>(
        accessor, number_elements);
  } else if constexpr (std::is_same<RangeT, sycl::nd_range<1>>::value) {
    return get_lambda_with_nd_range_for_span<VariableT, FunctorT,
                                             UseCombineFlagT>(accessor,
                                                              number_elements);
  }
}

}  // namespace reduction_get_lambda

#endif  // __SYCL_CTS_TEST_REDUCTION_GET_LAMBDA_H
