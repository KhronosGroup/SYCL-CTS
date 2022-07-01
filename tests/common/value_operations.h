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

namespace value_operations {

template <typename T, size_t N>
using ArrayT = T[N];

// Modify functions
template <typename T, size_t N>
inline void assign(ArrayT<T, N>& left, const T& right) {
  for (size_t i = 0; i < N; ++i) {
    left[i] = right;
  }
}

template <typename LeftArrT, size_t LeftArrN, typename RightArrT,
          size_t RightArrN>
inline void assign(ArrayT<LeftArrT, LeftArrN>& left,
                   const ArrayT<RightArrT, RightArrN>& right) {
  static_assert(LeftArrN == RightArrN, "Arrays have to be the same size");
  for (size_t i = 0; i < LeftArrN; ++i) {
    left[i] = right[i];
  }
}

template <typename LeftArrT, typename RightNonArrT>
inline typename std::enable_if_t<has_subscript_and_size_v<LeftArrT> &&
                                 !has_subscript_and_size_v<RightNonArrT>>
assign(LeftArrT& left, const RightNonArrT& right) {
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] = right;
  }
}

template <typename LeftArrT, typename RightArrT>
inline typename std::enable_if_t<has_subscript_and_size_v<LeftArrT> &&
                                 has_subscript_and_size_v<RightArrT>>
assign(LeftArrT& left, const RightArrT& right) {
  assert((left.size() == right.size()) && "Arrays have to be the same size");
  for (size_t i = 0; i < left.size(); ++i) {
    left[i] = right[i];
  }
}

template <typename LeftNonArrT, typename RightNonArrT = LeftNonArrT>
inline typename std::enable_if_t<!has_subscript_and_size_v<LeftNonArrT> &&
                                 !has_subscript_and_size_v<RightNonArrT>>
assign(LeftNonArrT& left, const RightNonArrT& right) {
  left = right;
}
/////////////////////////// Modify functions

// Compare functions
template <typename T, size_t N>
inline bool are_equal(const ArrayT<T, N>& left, const T& right) {
  for (size_t i = 0; i < N; ++i) {
    if (left[i] != right) return false;
  }
  return true;
}

template <typename LeftArrT, size_t LeftArrN, typename RightArrT,
          size_t RightArrN>
inline bool are_equal(const ArrayT<LeftArrT, LeftArrN>& left,
                      const ArrayT<RightArrT, RightArrN>& right) {
  static_assert(LeftArrN == RightArrN, "Arrays have to be the same size");
  for (size_t i = 0; i < LeftArrN; ++i) {
    if (left[i] != right[i]) return false;
  }
  return true;
}

template <typename LeftArrT, typename RightNonArrT>
inline typename std::enable_if_t<has_subscript_and_size_v<LeftArrT> &&
                                     !has_subscript_and_size_v<RightNonArrT>,
                                 bool>
are_equal(const LeftArrT& left, const RightNonArrT& right) {
  for (size_t i = 0; i < left.size(); ++i) {
    if (left[i] != right) return false;
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
    if (left[i] != right[i]) return false;
  }
  return true;
}

template <typename LeftNonArrT, typename RightNonArrT = LeftNonArrT>
inline typename std::enable_if_t<!has_subscript_and_size_v<LeftNonArrT> &&
                                     !has_subscript_and_size_v<RightNonArrT>,
                                 bool>
are_equal(const LeftNonArrT& left, const RightNonArrT& right) {
  return (left == right);
}
//////////////////////////// Compare functions

}  // namespace value_operations
#endif  //__SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_H
