/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2020-2022 The Khronos Group Inc.
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
//  This file contains helper functions for modifying and comparing values,
//  arrays, and objects that implements operator[]
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_H
#define __SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_H
#include "../../util/type_traits.h"

#include <cassert>
#include <tuple>
#include <utility>
#include <variant>

namespace value_operations {

namespace detail {

template <typename T, size_t N>
using ArrayT = T[N];

/**
 * @brief Helper function assigns value in general case
 * and assigns true if value is even number and false otherwise
 * in case of bool types
 *
 * @tparam T Type of variable for assignment
 * @tparam U Type of right operand for assignment operation
 * @param left Variable for assignment
 * @param right New value to assign
 */
template <typename T, typename U>
void assign_value_or_even(T& left, const U& right) {
  if constexpr (std::is_same_v<T, bool> ||
                std::is_same_v<T, std::optional<bool>>)
    left = (right % 2 != 0);
  else
    left = right;
}

/**
 * @brief Helper function checks if values are equal in general case
 * and checks if right operand is even number in case of bool types
 *
 * @tparam T Type of left operand for comparison
 * @tparam U Type of right operand for comparison
 * @param left Left operand for comparison
 * @param right Right operand for comparison
 */
template <typename T, typename U>
bool are_equal_value_or_even(const T& left, const U& right) {
  if constexpr (std::is_same_v<T, bool> ||
                std::is_same_v<T, std::optional<bool>>)
    return left == (right % 2 != 0);
  else
    return left == right;
}

/**
 * @brief Function allows to make assignment operations to elements of the
 * container using std::get and std::index_sequence
 *
 * @tparam ContainerT Type of container for assignment
 * @tparam U Right operand for assignment operation
 * @tparam I Indexes for assignment
 * @param left Container for assignment
 * @param right New value to assign
 */
template <typename ContainerT, typename U, std::size_t... I>
void assign_by_index_sequence(ContainerT& left, const U& right,
                              std::index_sequence<I...>) {
  ((assign_value_or_even(std::get<I>(left), right)), ...);
}

/**
 * @brief Function allows to compare elements of the
 * container using std::get and std::index_sequence
 *
 * @tparam ContainerT Type of container for assignment
 * @tparam U Right operand for assignment operation
 * @tparam I Indexes for assignment
 * @param left Left operand of comparison
 * @param right Right operand of comparison
 * @return Function returns true if all elements of the left operand are equal
 * to the right operand. False otherwise
 */
template <typename ContainerT, typename U, std::size_t... I>
bool are_equal_by_index_sequence(const ContainerT& left, const U& right,
                                 std::index_sequence<I...>) {
  bool result = true;
  ((result &= are_equal_value_or_even(std::get<I>(left), right)), ...);
  return result;
}

}  // namespace detail

// Modify functions
template <typename T, size_t N>
inline void assign(detail::ArrayT<T, N>& left, const T& right) {
  for (size_t i = 0; i < N; ++i) {
    detail::assign_value_or_even(left[i], right);
  }
}

template <typename LeftArrT, size_t LeftArrN, typename RightArrT,
          size_t RightArrN>
inline void assign(detail::ArrayT<LeftArrT, LeftArrN>& left,
                   const detail::ArrayT<RightArrT, RightArrN>& right) {
  static_assert(LeftArrN == RightArrN, "Arrays have to be the same size");
  for (size_t i = 0; i < LeftArrN; ++i) {
    detail::assign_value_or_even(left[i], right[i]);
  }
}

template <typename LeftArrT, typename RightNonArrT>
inline typename std::enable_if_t<has_subscript_and_size_v<LeftArrT> &&
                                 !has_subscript_and_size_v<RightNonArrT>>
assign(LeftArrT& left, const RightNonArrT& right) {
  for (size_t i = 0; i < left.size(); ++i) {
    detail::assign_value_or_even(left[i], right);
  }
}

template <typename LeftArrT, typename RightArrT>
inline typename std::enable_if_t<has_subscript_and_size_v<LeftArrT> &&
                                 has_subscript_and_size_v<RightArrT>>
assign(LeftArrT& left, const RightArrT& right) {
  assert((left.size() == right.size()) && "Arrays have to be the same size");
  for (size_t i = 0; i < left.size(); ++i) {
    detail::assign_value_or_even(left[i], right[i]);
  }
}

template <typename LeftNonArrT, typename RightNonArrT = LeftNonArrT>
inline typename std::enable_if_t<!has_subscript_and_size_v<LeftNonArrT> &&
                                 !has_subscript_and_size_v<RightNonArrT>>
assign(LeftNonArrT& left, const RightNonArrT& right) {
  detail::assign_value_or_even(left, right);
}

template <typename... Types, typename U>
void assign(std::tuple<Types...>& left, const U& right) {
  using tuple_t = std::remove_reference_t<decltype(left)>;
  using indexes = std::make_index_sequence<std::tuple_size<tuple_t>::value>;
  detail::assign_by_index_sequence(left, right, indexes());
}

template <typename FirstT, typename SecondT = FirstT, typename U>
void assign(std::pair<FirstT, SecondT>& left, const U& right) {
  constexpr size_t pair_size = 2;
  using indexes = std::make_index_sequence<pair_size>;
  detail::assign_by_index_sequence(left, right, indexes());
}
/////////////////////////// Modify functions

// Compare functions
template <typename T, size_t N>
inline bool are_equal(const detail::ArrayT<T, N>& left, const T& right) {
  for (size_t i = 0; i < N; ++i) {
    if (!detail::are_equal_value_or_even(left[i], right)) return false;
  }
  return true;
}

template <typename LeftArrT, size_t LeftArrN, typename RightArrT,
          size_t RightArrN>
inline bool are_equal(const detail::ArrayT<LeftArrT, LeftArrN>& left,
                      const detail::ArrayT<RightArrT, RightArrN>& right) {
  static_assert(LeftArrN == RightArrN, "Arrays have to be the same size");
  for (size_t i = 0; i < LeftArrN; ++i) {
    if (!detail::are_equal_value_or_even(left[i], right[i])) return false;
  }
  return true;
}

template <typename LeftArrT, typename RightNonArrT>
inline typename std::enable_if_t<has_subscript_and_size_v<LeftArrT> &&
                                     !has_subscript_and_size_v<RightNonArrT>,
                                 bool>
are_equal(const LeftArrT& left, const RightNonArrT& right) {
  for (size_t i = 0; i < left.size(); ++i) {
    if (!detail::are_equal_value_or_even(left[i], right)) return false;
  }
  return true;
}

template <typename LeftArrT, typename RightArrT>
inline typename std::enable_if_t<has_subscript_and_size_v<LeftArrT> &&
                                     has_subscript_and_size_v<RightArrT>,
                                 bool>
are_equal(const LeftArrT& left, const RightArrT& right) {
  assert((left.size() == right.size()) && "Arrays have to be the same size");
  for (size_t i = 0; i < left.size(); ++i) {
    if (!detail::are_equal_value_or_even(left[i], right[i])) return false;
  }
  return true;
}

template <typename LeftNonArrT, typename RightNonArrT = LeftNonArrT>
inline typename std::enable_if_t<!has_subscript_and_size_v<LeftNonArrT> &&
                                     !has_subscript_and_size_v<RightNonArrT>,
                                 bool>
are_equal(const LeftNonArrT& left, const RightNonArrT& right) {
  return detail::are_equal_value_or_even(left, right);
}

template <typename... Types, typename U>
std::enable_if_t<!std::is_same_v<std::tuple<Types...>, U>, bool> are_equal(
    const std::tuple<Types...>& left, const U& right) {
  using tuple_t = std::remove_reference_t<decltype(left)>;
  using indexes = std::make_index_sequence<std::tuple_size<tuple_t>::value>;
  return detail::are_equal_by_index_sequence(left, right, indexes());
}

template <typename FirstT, typename SecondT = FirstT, typename U>
typename std::enable_if_t<!std::is_same_v<std::pair<FirstT, SecondT>, U>, bool>
are_equal(const std::pair<FirstT, SecondT>& left, const U& right) {
  constexpr size_t pair_size = 2;
  using indexes = std::make_index_sequence<pair_size>;
  return detail::are_equal_by_index_sequence(left, right, indexes());
}

template <typename... Types, typename U>
typename std::enable_if_t<!std::is_same_v<std::variant<Types...>, U>, bool>
are_equal(const std::variant<Types...>& left, const U& right) {
  return std::get<U>(left) == right;
}
//////////////////////////// Compare functions

template <typename T>
inline constexpr auto init(int val) {
  std::remove_const_t<T> data;
  assign(data, val);
  return data;
}

}  // namespace value_operations
#endif  //__SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_H
