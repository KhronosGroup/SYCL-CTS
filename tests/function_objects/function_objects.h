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

#include "../common/type_coverage.h"

#include <type_traits>

#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
#define TYPES                                                                 \
  bool, char, signed char, unsigned char, short int, unsigned short int, int, \
      unsigned int, long int, unsigned long int, long long int,               \
      unsigned long long int, float
#define TYPE_NAMES                                             \
  "bool", "char", "signed char", "unsigned char", "short int", \
      "unsigned short int", "int", "unsigned int", "long int", \
      "unsigned long int", "long long int", "unsigned long long int", "float"
#define TYPES_VECTOR                                                           \
  sycl::marray<bool, 1>, sycl::marray<char, 2>,                                \
      sycl::marray<unsigned short, 3>, sycl::marray<int, 4>,                   \
      sycl::marray<unsigned long int, 5>, sycl::marray<long long int, 6>,      \
      sycl::marray<float, 7>, sycl::vec<bool, 1>, sycl::vec<unsigned char, 2>, \
      sycl::vec<short, 3>, sycl::vec<unsigned int, 4>, sycl::vec<long, 8>,     \
      sycl::vec<unsigned long long, 16>, sycl::vec<float, 1>
#define TYPE_NAMES_VECTOR                                                     \
  "sycl::marray<bool, 1>", "sycl::marray<char, 2>",                           \
      "sycl::marray<unsigned short, 3>", "sycl::marray<int, 4>",              \
      "sycl::marray<unsigned long int, 5>", "sycl::marray<long long int, 6>", \
      "sycl::marray<float, 7>", "sycl::vec<bool, 1>",                         \
      "sycl::vec<unsigned char, 2>", "sycl::vec<short, 3>",                   \
      "sycl::vec<unsigned int, 4>", "sycl::vec<long, 8>",                     \
      "sycl::vec<unsigned long long, 16>", "sycl::vec<float, 1>"
#else
#define TYPES \
  bool, char, unsigned short int, int, unsigned long int, long long int, float
#define TYPE_NAMES                                                  \
  "bool", "char", "unsigned short int", "int", "unsigned long int", \
      "long long int", "float"
#define TYPES_VECTOR                                                           \
  sycl::marray<bool, 1>, sycl::marray<int, 4>, sycl::marray<long long int, 6>, \
      sycl::marray<float, 7>, sycl::vec<unsigned char, 2>,                     \
      sycl::vec<unsigned int, 4>, sycl::vec<float, 1>
#define TYPE_NAMES_VECTOR                                          \
  "sycl::marray<bool, 1>", "sycl::marray<int, 4>",                 \
      "sycl::marray<long long int, 6>", "sycl::marray<float, 7>",  \
      "sycl::vec<unsigned char, 2>", "sycl::vec<unsigned int, 4>", \
      "sycl::vec<float, 1>"
#endif

/** Gets all SYCL function object void specializations. */
inline auto get_op_types() {
  static const auto types =
      named_type_pack<sycl::plus<>, sycl::multiplies<>, sycl::bit_and<>,
                      sycl::bit_or<>, sycl::bit_xor<>, sycl::logical_and<>,
                      sycl::logical_or<>, sycl::minimum<>,
                      sycl::maximum<>>::generate("plus", "multiplies",
                                                 "bit_and", "bit_or", "bit_xor",
                                                 "logical_and", "logical_or",
                                                 "minimum", "maximum");
  return types;
}

/** Checks whether \p T is a void-specialized bitwise SYCL operator type. */
template <typename T>
struct is_void_bitwise_op
    : std::integral_constant<bool, std::is_same_v<T, sycl::bit_and<>> ||
                                       std::is_same_v<T, sycl::bit_or<>> ||
                                       std::is_same_v<T, sycl::bit_xor<>>> {};

/** Checks whether \p T is a sycl::vec type and its data type is integral. */
template <typename T>
struct is_integral_vec {
  static constexpr bool value = false;
};

template <typename DataT, int NumElements>
struct is_integral_vec<sycl::vec<DataT, NumElements>> {
  static constexpr bool value = std::is_integral_v<DataT>;
};

/** Checks whether \p T is a sycl::marray type and its data type is integral. */
template <typename T>
struct is_integral_marray {
  static constexpr bool value = false;
};

template <typename DataT, std::size_t NumElements>
struct is_integral_marray<sycl::marray<DataT, NumElements>> {
  static constexpr bool value = std::is_integral_v<DataT>;
};

/**
 * For \p T sycl::vec or sycl::marray, checks whether the data type is
 * integral. */
template <typename T>
struct is_integral_vector
    : std::integral_constant<bool, is_integral_vec<T>::value ||
                                       is_integral_marray<T>::value> {};

/**
 * Translates a SYCL function object \p OpT to a C++ built-in operator
 * and infers the type of executing that operator on
 * types \p LhsT and \p RhsT. */
template <typename OpT, typename LhsT, typename RhsT>
struct builtin_return_t;

template <typename LhsT, typename RhsT>
struct builtin_return_t<sycl::plus<>, LhsT, RhsT> {
  using type = decltype(LhsT{} + RhsT{});
};

template <typename LhsT, typename RhsT>
struct builtin_return_t<sycl::multiplies<>, LhsT, RhsT> {
  using type = decltype(LhsT{} * RhsT{});
};

template <typename LhsT, typename RhsT>
struct builtin_return_t<sycl::bit_and<>, LhsT, RhsT> {
  using type = decltype(LhsT{} & RhsT{});
};

template <typename LhsT, typename RhsT>
struct builtin_return_t<sycl::bit_or<>, LhsT, RhsT> {
  using type = decltype(LhsT{} | RhsT{});
};

template <typename LhsT, typename RhsT>
struct builtin_return_t<sycl::bit_xor<>, LhsT, RhsT> {
  using type = decltype(LhsT{} ^ RhsT{});
};

template <typename LhsT, typename RhsT>
struct builtin_return_t<sycl::logical_and<>, LhsT, RhsT> {
  using type = decltype(LhsT{} && RhsT{});
};

template <typename LhsT, typename RhsT>
struct builtin_return_t<sycl::logical_or<>, LhsT, RhsT> {
  using type = decltype(LhsT{} || RhsT{});
};

template <typename LhsT, typename RhsT>
struct builtin_return_t<sycl::minimum<>, LhsT, RhsT> {
  // obtain common type using ternary operator, condition has no effect on type
  using type = decltype(false ? LhsT{} : RhsT{});
};

template <typename LhsT, typename RhsT>
struct builtin_return_t<sycl::maximum<>, LhsT, RhsT> {
  // obtain common type using ternary operator, condition has no effect on type
  using type = decltype(false ? LhsT{} : RhsT{});
};

/**
 * Checks that void-specialized SYCL function object \p OpT has the same
 * return type as calling the built-in operator on the operands. */
template <typename OpT, typename LhsT, typename RhsT>
void check_return_type() {
  using expc_return_t = typename builtin_return_t<OpT, LhsT, RhsT>::type;

  OpT op{};
  using sycl_return_t = decltype(op(LhsT{}, RhsT{}));

  STATIC_CHECK(std::is_same_v<expc_return_t, sycl_return_t>);
}

/**
 * Checks that void-specialized SYCL function object \p OpT has the
 * correct return type given operand types \p LhsT and \p RhsT. */
template <typename OpT, typename LhsT, typename RhsT, typename Enable = void>
struct check_scalar_return_type {
  void operator()(const std::string &type_name_op,
                  const std::string &type_name_lhs,
                  const std::string &type_name_rhs) {}
};

template <typename OpT, typename LhsT, typename RhsT>
struct check_scalar_return_type<
    OpT, LhsT, RhsT,
    typename std::enable_if_t<!is_void_bitwise_op<OpT>::value ||
                              (std::is_integral_v<LhsT> &&
                               std::is_integral_v<RhsT>)>> {
  void operator()(const std::string &type_name_op,
                  const std::string &type_name_lhs,
                  const std::string &type_name_rhs) {
    INFO("" << type_name_op << "(" << type_name_lhs << ", " << type_name_rhs
            << ")");

    check_return_type<OpT, LhsT, RhsT>();
  }
};

/**
 * Checks that void-specialized SYCL function object \p OpT has the
 * correct return type given vector operand type \p OperandT for
 * left- and right-hand side. */
template <typename OpT, typename OperandT, typename Enable = void>
struct check_vector_return_type {
  void operator()(const std::string &type_name_op,
                  const std::string &type_name_operand) {}
};

template <typename OpT, typename OperandT>
struct check_vector_return_type<
    OpT, OperandT,
    typename std::enable_if_t<!is_void_bitwise_op<OpT>::value ||
                              is_integral_vector<OperandT>::value>> {
  void operator()(const std::string &type_name_op,
                  const std::string &type_name_operand) {
    INFO("" << type_name_op << "(" << type_name_operand << ", "
            << type_name_operand << ")");

    check_return_type<OpT, OperandT, OperandT>();
  }
};
