/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_IDENTITY_HELPER_H
#define __SYCL_CTS_TEST_IDENTITY_HELPER_H

#include "../../util/type_traits.h"

#include <limits>
#include <type_traits>

/** plus */
template <
    typename AccumulatorT, typename OperatorT,
    std::enable_if_t<std::is_same_v<OperatorT, sycl::plus<AccumulatorT>> &&
                         std::is_arithmetic_v<AccumulatorT>,
                     bool> = true>
AccumulatorT get_identity() {
  return {};
}

/** multiplies */
template <typename AccumulatorT, typename OperatorT,
          std::enable_if_t<
              std::is_same_v<OperatorT, sycl::multiplies<AccumulatorT>> &&
                  std::is_arithmetic_v<AccumulatorT>,
              bool> = true>
AccumulatorT get_identity() {
  return {1};
}

/** bit_and */
template <
    typename AccumulatorT, typename OperatorT,
    std::enable_if_t<std::is_same_v<OperatorT, sycl::bit_and<AccumulatorT>> &&
                         std::is_integral_v<AccumulatorT>,
                     bool> = true>
AccumulatorT get_identity() {
  return ~AccumulatorT{};
}

/** bit_or */
template <
    typename AccumulatorT, typename OperatorT,
    std::enable_if_t<std::is_same_v<OperatorT, sycl::bit_or<AccumulatorT>> &&
                         std::is_integral_v<AccumulatorT>,
                     bool> = true>
AccumulatorT get_identity() {
  return {};
}

/** bit_xor */
template <
    typename AccumulatorT, typename OperatorT,
    std::enable_if_t<std::is_same_v<OperatorT, sycl::bit_xor<AccumulatorT>> &&
                         std::is_integral_v<AccumulatorT>,
                     bool> = true>
AccumulatorT get_identity() {
  return {};
}

/** logical_and */
template <typename AccumulatorT, typename OperatorT,
          std::enable_if_t<
              std::is_same_v<OperatorT, sycl::logical_and<AccumulatorT>> &&
                  std::is_same_v<std::remove_cv_t<AccumulatorT>, bool>,
              bool> = true>
bool get_identity() {
  return true;
}

/** logical_or */
template <typename AccumulatorT, typename OperatorT,
          std::enable_if_t<
              std::is_same_v<OperatorT, sycl::logical_or<AccumulatorT>> &&
                  std::is_same_v<std::remove_cv_t<AccumulatorT>, bool>,
              bool> = true>
bool get_identity() {
  return false;
}

/** minimum (integral) */
template <
    typename AccumulatorT, typename OperatorT,
    std::enable_if_t<std::is_same_v<OperatorT, sycl::minimum<AccumulatorT>> &&
                         std::is_integral_v<AccumulatorT>,
                     bool> = true>
AccumulatorT get_identity() {
  return std::numeric_limits<AccumulatorT>::max();
}

/** minimum (floating point) */
template <
    typename AccumulatorT, typename OperatorT,
    std::enable_if_t<std::is_same_v<OperatorT, sycl::minimum<AccumulatorT>> &&
                         is_sycl_scalar_floating_point_v<AccumulatorT>,
                     bool> = true>
AccumulatorT get_identity() {
  return std::numeric_limits<AccumulatorT>::infinity();
}

/** maximum (integral) */
template <
    typename AccumulatorT, typename OperatorT,
    std::enable_if_t<std::is_same_v<OperatorT, sycl::maximum<AccumulatorT>> &&
                         std::is_integral_v<AccumulatorT>,
                     bool> = true>
AccumulatorT get_identity() {
  return std::numeric_limits<AccumulatorT>::lowest();
}

/** maximum (floating point) */
template <typename AccumulatorT, typename OperatorT,
          std::enable_if_t<
              std::is_same_v<OperatorT, sycl::maximum<AccumulatorT>> &&
                  (std::is_floating_point_v<AccumulatorT> ||
                   std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>),
              bool> = true>
AccumulatorT get_identity() {
  return -std::numeric_limits<AccumulatorT>::infinity();
}

#endif  // __SYCL_CTS_TEST_IDENTITY_HELPER_H
