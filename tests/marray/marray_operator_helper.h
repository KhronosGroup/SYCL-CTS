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
*******************************************************************************/

#ifndef SYCL_CTS_TEST_MARRAY_MARRAY_OPERATOR_HELPER_H
#define SYCL_CTS_TEST_MARRAY_MARRAY_OPERATOR_HELPER_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/type_list.h"

#include <valarray>

#define OPERATOR_BINARY(id, op)                                              \
  struct op_##id {                                                           \
    template <typename DataT, std::size_t NumElements>                       \
    sycl::marray<DataT, NumElements> operator()(                             \
        const sycl::marray<DataT, NumElements>& lhs,                         \
        const sycl::marray<DataT, NumElements>& rhs) {                       \
      return lhs op rhs;                                                     \
    }                                                                        \
    template <typename DataT, std::size_t NumElements>                       \
    sycl::marray<DataT, NumElements> operator()(                             \
        const sycl::marray<DataT, NumElements>& lhs, const DataT& rhs) {     \
      return lhs op rhs;                                                     \
    }                                                                        \
    template <typename DataT, std::size_t NumElements>                       \
    sycl::marray<DataT, NumElements> operator()(                             \
        const DataT& lhs, const sycl::marray<DataT, NumElements>& rhs) {     \
      return sycl::marray<DataT, NumElements>(lhs) op rhs;                   \
    }                                                                        \
    template <typename T>                                                    \
    std::valarray<T> operator()(const std::valarray<T>& lhs,                 \
                                const std::valarray<T>& rhs) {               \
      return lhs op rhs;                                                     \
    }                                                                        \
    template <typename T>                                                    \
    std::valarray<T> operator()(const std::valarray<T>& lhs, const T& rhs) { \
      return lhs op rhs;                                                     \
    }                                                                        \
    template <typename T>                                                    \
    std::valarray<T> operator()(const T& lhs, const std::valarray<T>& rhs) { \
      return lhs op rhs;                                                     \
    }                                                                        \
  }

OPERATOR_BINARY(add, +);
OPERATOR_BINARY(sub, -);
OPERATOR_BINARY(mul, *);
OPERATOR_BINARY(div, /);
OPERATOR_BINARY(mod, %);
OPERATOR_BINARY(bor, |);
OPERATOR_BINARY(band, &);
OPERATOR_BINARY(bxor, ^);
OPERATOR_BINARY(sl, <<);
OPERATOR_BINARY(sr, >>);

#define OPERATOR_BINARY_RELATIONAL(id, op)                               \
  struct op_##id {                                                       \
    template <typename DataT, std::size_t NumElements>                   \
    sycl::marray<bool, NumElements> operator()(                          \
        const sycl::marray<DataT, NumElements>& lhs,                     \
        const sycl::marray<DataT, NumElements>& rhs) {                   \
      return lhs op rhs;                                                 \
    }                                                                    \
    template <typename DataT, std::size_t NumElements>                   \
    sycl::marray<bool, NumElements> operator()(                          \
        const sycl::marray<DataT, NumElements>& lhs, const DataT& rhs) { \
      return lhs op rhs;                                                 \
    }                                                                    \
    template <typename DataT, std::size_t NumElements>                   \
    sycl::marray<bool, NumElements> operator()(                          \
        const DataT& lhs, const sycl::marray<DataT, NumElements>& rhs) { \
      return lhs op rhs;                                                 \
    }                                                                    \
    template <typename T>                                                \
    std::valarray<bool> operator()(const std::valarray<T>& lhs,          \
                                   const std::valarray<T>& rhs) {        \
      return lhs op rhs;                                                 \
    }                                                                    \
    template <typename T>                                                \
    std::valarray<bool> operator()(const std::valarray<T>& lhs,          \
                                   const T& rhs) {                       \
      return lhs op rhs;                                                 \
    }                                                                    \
    template <typename T>                                                \
    std::valarray<bool> operator()(const T& lhs,                         \
                                   const std::valarray<T>& rhs) {        \
      return lhs op rhs;                                                 \
    }                                                                    \
  }

OPERATOR_BINARY_RELATIONAL(land, &&);
OPERATOR_BINARY_RELATIONAL(lor, ||);

#define OPERATOR_BINARY_ASSIGN(id, op)                                  \
  struct op_assign_##id {                                               \
    template <typename DataT, std::size_t NumElements>                  \
    sycl::marray<DataT, NumElements>& operator()(                       \
        sycl::marray<DataT, NumElements>& lhs,                          \
        const sycl::marray<DataT, NumElements>& rhs) {                  \
      return lhs op rhs;                                                \
    }                                                                   \
    template <typename T>                                               \
    std::valarray<T>& operator()(std::valarray<T>& lhs,                 \
                                 const std::valarray<T>& rhs) {         \
      return lhs op rhs;                                                \
    }                                                                   \
    template <typename DataT, std::size_t NumElements>                  \
    sycl::marray<DataT, NumElements>& operator()(                       \
        sycl::marray<DataT, NumElements>& lhs, const DataT& rhs) {      \
      return lhs op rhs;                                                \
    }                                                                   \
    template <typename T>                                               \
    std::valarray<T>& operator()(std::valarray<T>& lhs, const T& rhs) { \
      return lhs op rhs;                                                \
    }                                                                   \
  }

OPERATOR_BINARY_ASSIGN(add, +=);
OPERATOR_BINARY_ASSIGN(sub, -=);
OPERATOR_BINARY_ASSIGN(mul, *=);
OPERATOR_BINARY_ASSIGN(div, /=);
OPERATOR_BINARY_ASSIGN(mod, %=);
OPERATOR_BINARY_ASSIGN(bor, |=);
OPERATOR_BINARY_ASSIGN(band, &=);
OPERATOR_BINARY_ASSIGN(bxor, ^=);
OPERATOR_BINARY_ASSIGN(sl, <<=);
OPERATOR_BINARY_ASSIGN(sr, >>=);

#define OPERATOR_UNARY(id, op)                         \
  struct op_##id {                                     \
    template <typename DataT, std::size_t NumElements> \
    sycl::marray<DataT, NumElements> operator()(       \
        const sycl::marray<DataT, NumElements>& v) {   \
      return op v;                                     \
    }                                                  \
    template <typename T>                              \
    std::valarray<T> operator()(std::valarray<T>& v) { \
      return op v;                                     \
    }                                                  \
  }

OPERATOR_UNARY(upos, +);
OPERATOR_UNARY(uneg, -);
OPERATOR_UNARY(bnot, ~);

struct op_lnot {
  template <typename DataT, std::size_t NumElements>
  sycl::marray<bool, NumElements> operator()(
      const sycl::marray<DataT, NumElements>& v) {
    return !v;
  }
  template <typename T>
  std::valarray<bool> operator()(const std::valarray<T>& v) {
    return !v;
  }
};

#define OPERATOR_UNARY_PRE(id, op, op_assign)          \
  struct op_##id {                                     \
    template <typename DataT, std::size_t NumElements> \
    sycl::marray<DataT, NumElements> operator()(       \
        sycl::marray<DataT, NumElements>& v) {         \
      return op v;                                     \
    }                                                  \
    template <typename T>                              \
    std::valarray<T> operator()(std::valarray<T>& v) { \
      return v op_assign T{1};                         \
    }                                                  \
  }

OPERATOR_UNARY_PRE(pre_inc, ++, +=);
OPERATOR_UNARY_PRE(pre_dec, --, -=);

#define OPERATOR_UNARY_POST(id, op, op_assign)         \
  struct op_##id {                                     \
    template <typename DataT, std::size_t NumElements> \
    sycl::marray<DataT, NumElements> operator()(       \
        sycl::marray<DataT, NumElements>& v) {         \
      return v op;                                     \
    }                                                  \
    template <typename T>                              \
    std::valarray<T> operator()(std::valarray<T>& v) { \
      std::valarray<T> tmp = v;                        \
      v op_assign T{1};                                \
      return tmp;                                      \
    }                                                  \
  }

OPERATOR_UNARY_POST(post_inc, ++, +=);
OPERATOR_UNARY_POST(post_dec, --, -=);

#endif  // SYCL_CTS_TEST_MARRAY_MARRAY_OPERATOR_HELPER_H
