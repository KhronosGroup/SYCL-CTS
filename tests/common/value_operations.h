/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2020-2022 The Khronos Group Inc.
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
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

template <typename ContainerT, typename U, std::size_t... I>
void assign_by_index_sequence(ContainerT& left, const U& right,
                              std::index_sequence<I...>) {
  ((std::get<I>(left) = right), ...);
}

template <typename ContainerT, typename U, std::size_t... I>
bool are_equal_by_index_sequence(const ContainerT& left, const U& right,
                                 std::index_sequence<I...>) {
  bool result = true;
  ((result &= std::get<I>(left) == right), ...);
  return result;
}

}  // namespace detail

// Modify functions
template <typename T, size_t N>
inline void assign(detail::ArrayT<T, N>& left, const T& right) {
  for (size_t i = 0; i < N; ++i) {
    left[i] = right;
  }
}

template <typename LeftArrT, size_t LeftArrN, typename RightArrT,
          size_t RightArrN>
inline void assign(detail::ArrayT<LeftArrT, LeftArrN>& left,
                   const detail::ArrayT<RightArrT, RightArrN>& right) {
  static_assert(LeftArrN == RightArrN, "Arrays have to be the same size");
  for (size_t i = 0; i < LeftArrN; ++i) {
    left[i] = right[i];
  }
}

template <typename LeftArrT, typename RightNonArrT>
inline typename std::enable_if_t<has_subscript_operator_v<LeftArrT> &&
                                 !has_subscript_operator_v<RightNonArrT>>
assign(LeftArrT& left, const RightNonArrT& right) {
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] = right;
  }
}

template <typename LeftArrT, typename RightArrT>
inline typename std::enable_if_t<has_subscript_operator_v<LeftArrT> &&
                                 has_subscript_operator_v<RightArrT>>
assign(LeftArrT& left, const RightArrT& right) {
  assert((left.size() == right.size()) && "Arrays have to be the same size");
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] = right[i];
  }
}

template <typename LeftNonArrT, typename RightNonArrT = LeftNonArrT>
inline typename std::enable_if_t<!has_subscript_operator_v<LeftNonArrT> &&
                                 !has_subscript_operator_v<RightNonArrT>>
assign(LeftNonArrT& left, const RightNonArrT& right) {
  left = right;
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
    if (left[i] != right) return false;
  }
  return true;
}

template <typename LeftArrT, size_t LeftArrN, typename RightArrT,
          size_t RightArrN>
inline bool are_equal(const detail::ArrayT<LeftArrT, LeftArrN>& left,
                      const detail::ArrayT<RightArrT, RightArrN>& right) {
  static_assert(LeftArrN == RightArrN, "Arrays have to be the same size");
  for (size_t i = 0; i < LeftArrN; ++i) {
    if (left[i] != right[i]) return false;
  }
  return true;
}

template <typename LeftArrT, typename RightNonArrT>
inline typename std::enable_if_t<has_subscript_operator_v<LeftArrT> &&
                                     !has_subscript_operator_v<RightNonArrT>,
                                 bool>
are_equal(const LeftArrT& left, const RightNonArrT& right) {
  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i] != right) return false;
  }
  return true;
}

template <typename LeftArrT, typename RightArrT>
inline typename std::enable_if_t<has_subscript_operator_v<LeftArrT> &&
                                     has_subscript_operator_v<RightArrT>,
                                 bool>
are_equal(const LeftArrT& left, const RightArrT& right) {
  assert((left.size() == right.size()) && "Arrays have to be the same size");
  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i] != right[i]) return false;
  }
  return true;
}

template <typename LeftNonArrT, typename RightNonArrT = LeftNonArrT>
inline typename std::enable_if_t<!has_subscript_operator_v<LeftNonArrT> &&
                                     !has_subscript_operator_v<RightNonArrT>,
                                 bool>
are_equal(const LeftNonArrT& left, const RightNonArrT& right) {
  return (left == right);
}

template <typename... Types, typename U>
bool are_equal(std::tuple<Types...>& left, const U& right) {
  using tuple_t = std::remove_reference_t<decltype(left)>;
  using indexes = std::make_index_sequence<std::tuple_size<tuple_t>::value>;
  return detail::are_equal_by_index_sequence(left, right, indexes());
}

template <typename FirstT, typename SecondT = FirstT, typename U>
bool are_equal(std::pair<FirstT, SecondT>& left, const U& right) {
  constexpr size_t pair_size = 2;
  using indexes = std::make_index_sequence<pair_size>;
  return detail::are_equal_by_index_sequence(left, right, indexes());
}

template <typename... Types, typename U>
bool are_equal(std::variant<Types...>& left, const U& right) {
  return std::get<U>(left) == right;
}
//////////////////////////// Compare functions

}  // namespace value_operations
#endif  //__SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_H
