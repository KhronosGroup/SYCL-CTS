/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef SYCLCTS_TESTS_MARRAY_MARRAY_OPERATOR_H
#define SYCLCTS_TESTS_MARRAY_MARRAY_OPERATOR_H

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "../common/section_name_builder.h"
#include "marray_common.h"
#include "marray_operator_helper.h"

#include <valarray>

namespace marray_operators {

template <typename DataT, typename NumElementsT>
struct operators_helper {
  static constexpr std::size_t NumElements = NumElementsT::value;
  using marray_t = sycl::marray<DataT, NumElements>;
  using varray_t = std::valarray<DataT>;

  template <typename init_func, typename array_type>
  static void init(array_type &ma) {
    for (std::size_t i = 0; i < NumElements; i++) {
      ma[i] = init_func::template init<DataT>(i);
    }
  }

  template <typename init_func>
  static void init(DataT &d) {
    d = init_func::template init<DataT>();
  }
};

/**
 * @brief Define several sequences to initialize array instances. */

struct seq_inc {
  template <typename T>
  static T init(std::size_t i) {
    return T(i + 1);
  }
};

template <std::size_t NumElements>
struct seq_dec {
  template <typename T>
  static T init(std::size_t i) {
    return T(NumElements - i);
  }
};

struct seq_one {
  template <typename T>
  static T init(std::size_t) {
    return T(1);
  }
};

struct seq_two {
  template <typename T>
  static T init(std::size_t) {
    return T(2);
  }
};

template <std::size_t NumElements>
inline auto get_sequences() {
  return named_type_pack<seq_inc, seq_dec<NumElements>, seq_one,
                         seq_two>::generate("incrementing sequence",
                                            "decrementing sequence",
                                            "sequences of ones",
                                            "sequence of twos");
}

/**
 * @brief Define several constants to initialize scalar instances. */

struct sca_one {
  template <typename T>
  static T init() {
    return T(1);
  }
};

struct sca_two {
  template <typename T>
  static T init() {
    return T(2);
  }
};

inline auto get_scalars() {
  return named_type_pack<sca_one, sca_two>::generate("one (1)", "two (2)");
}

template <typename DataT, typename NumElementsT, typename OpT,
          typename SequenceT>
class run_unary_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

 public:
  void operator()(const std::string &function_name) {
    INFO("for input (sequence) \"" << function_name << "\": ");

    OpT op{};

    typename helper::varray_t val_expected(helper::NumElements);
    helper::template init<SequenceT>(val_expected);
    auto res_expected = op(val_expected);

    typename helper::marray_t val_actual;
    helper::template init<SequenceT>(val_actual);
    auto res_actual = op(val_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename enable = void>
class run_unary {
 public:
  void operator()(const std::string &) {}
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_unary<DataT, NumElementsT, OpT,
                std::enable_if_t<std::is_integral_v<DataT> ||
                                 (std::is_floating_point_v<DataT> &&
                                  !std::is_same_v<OpT, op_bnot>)>> {
 public:
  void operator()(const std::string &operator_name) {
    INFO("for operator \"" << operator_name << "\": ");

    const auto functions = get_sequences<NumElementsT::value>();
    for_all_combinations<run_unary_sequence, DataT, NumElementsT, OpT>(
        functions);
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename SequenceT>
class run_unary_post_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

 public:
  void operator()(const std::string &function_name) {
    INFO("for input (sequence) \"" << function_name << "\": ");

    OpT op{};

    typename helper::varray_t val_expected(helper::NumElements);
    helper::template init<SequenceT>(val_expected);
    auto res_expected = op(val_expected);

    typename helper::marray_t val_actual;
    helper::template init<SequenceT>(val_actual);
    auto res_actual = op(val_actual);

    // check the returned output
    CHECK(value_operations::are_equal(res_expected, res_actual));
    // check the modified input
    CHECK(value_operations::are_equal(val_expected, val_actual));
  }
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_unary_post {
 public:
  void operator()(const std::string &operator_name) {
    INFO("for operator \"" << operator_name << "\": ");

    const auto functions = get_sequences<NumElementsT::value>();
    for_all_combinations<run_unary_post_sequence, DataT, NumElementsT, OpT>(
        functions);
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename SequenceT, typename ScalarT>
class run_binary_sequence_scalar {
  using helper = operators_helper<DataT, NumElementsT>;

 public:
  void operator()(const std::string &function_name,
                  const std::string &constant_name) {
    INFO("for lhs (sequence) \"" << function_name << "\": ");
    INFO("for rhs (scalar) \"" << constant_name << "\": ");

    OpT op{};

    typename helper::varray_t lhs_expected(helper::NumElements);
    helper::template init<SequenceT>(lhs_expected);
    DataT rhs_expected;
    helper::template init<ScalarT>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    typename helper::marray_t lhs_actual;
    helper::template init<SequenceT>(lhs_actual);
    DataT rhs_actual;
    helper::template init<ScalarT>(rhs_actual);
    auto res_actual = op(lhs_actual, rhs_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }
};

template <typename DataT, typename NumElementsT, typename OpT, typename ScalarT,
          typename SequenceT>
class run_binary_scalar_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

 public:
  void operator()(const std::string &constant_name,
                  const std::string &function_name) {
    INFO("for lhs (scalar) \"" << constant_name << "\": ");
    INFO("for rhs (sequence) \"" << function_name << "\": ");

    OpT op{};

    DataT lhs_expected;
    helper::template init<ScalarT>(lhs_expected);
    typename helper::varray_t rhs_expected(helper::NumElements);
    helper::template init<SequenceT>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    DataT lhs_actual;
    helper::template init<ScalarT>(lhs_actual);
    typename helper::marray_t rhs_actual;
    helper::template init<SequenceT>(rhs_actual);
    auto res_actual = op(lhs_actual, rhs_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename SequenceT1, typename SequenceT2>
class run_binary_sequence_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

 public:
  void operator()(const std::string &function_name_1,
                  const std::string &function_name_2) {
    INFO("for lhs (sequence) \"" << function_name_1 << "\": ");
    INFO("for rhs (sequence) \"" << function_name_2 << "\": ");

    OpT op{};

    typename helper::varray_t lhs_expected(helper::NumElements);
    helper::template init<SequenceT1>(lhs_expected);
    typename helper::varray_t rhs_expected(helper::NumElements);
    helper::template init<SequenceT2>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    typename helper::marray_t lhs_actual;
    helper::template init<SequenceT1>(lhs_actual);
    typename helper::marray_t rhs_actual;
    helper::template init<SequenceT2>(rhs_actual);
    auto res_actual = op(lhs_actual, rhs_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename enable = void>
class run_binary {
 public:
  void operator()(const std::string &) {}
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_binary<
    DataT, NumElementsT, OpT,
    typename std::enable_if_t<
        std::is_integral_v<DataT> ||
        (std::is_floating_point_v<DataT> &&
         !(std::is_same_v<OpT, op_mod> || std::is_same_v<OpT, op_band> ||
           std::is_same_v<OpT, op_bor> || std::is_same_v<OpT, op_bxor> ||
           std::is_same_v<OpT, op_sl> || std::is_same_v<OpT, op_sr>))>> {
 public:
  void operator()(const std::string &operator_name) {
    INFO("for operator \"" << operator_name << "\": ");

    const auto constants = get_scalars();
    const auto functions = get_sequences<NumElementsT::value>();
    for_all_combinations<run_binary_sequence_scalar, DataT, NumElementsT, OpT>(
        functions, constants);
    for_all_combinations<run_binary_scalar_sequence, DataT, NumElementsT, OpT>(
        constants, functions);
    for_all_combinations<run_binary_sequence_sequence, DataT, NumElementsT,
                         OpT>(functions, functions);
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename SequenceT, typename ScalarT>
class run_binary_assignment_sequence_scalar {
  using helper = operators_helper<DataT, NumElementsT>;

 public:
  void operator()(const std::string &function_name,
                  const std::string &constant_name) {
    INFO("for lhs (sequence) \"" << function_name << "\": ");
    INFO("for rhs (scalar) \"" << constant_name << "\": ");

    OpT op{};

    typename helper::varray_t lhs_expected(helper::NumElements);
    helper::template init<SequenceT>(lhs_expected);
    DataT rhs_expected;
    helper::template init<ScalarT>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    typename helper::marray_t lhs_actual;
    helper::template init<SequenceT>(lhs_actual);
    DataT rhs_actual;
    helper::template init<ScalarT>(rhs_actual);
    auto res_actual = op(lhs_actual, rhs_actual);

    // check the returned output
    CHECK(value_operations::are_equal(res_expected, res_actual));
    // check the modified input
    CHECK(value_operations::are_equal(lhs_expected, lhs_actual));
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename SequenceT1, typename SequenceT2>
class run_binary_assignment_sequence_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

 public:
  void operator()(const std::string &function_name_1,
                  const std::string &function_name_2) {
    INFO("for lhs (sequence) \"" << function_name_1 << "\": ");
    INFO("for rhs (sequence) \"" << function_name_2 << "\": ");

    OpT op{};

    typename helper::varray_t lhs_expected(helper::NumElements);
    helper::template init<SequenceT1>(lhs_expected);
    typename helper::varray_t rhs_expected(helper::NumElements);
    helper::template init<SequenceT2>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    typename helper::marray_t lhs_actual;
    helper::template init<SequenceT1>(lhs_actual);
    typename helper::marray_t rhs_actual;
    helper::template init<SequenceT2>(rhs_actual);
    auto res_actual = op(lhs_actual, rhs_actual);

    // check the returned output
    CHECK(value_operations::are_equal(res_expected, res_actual));
    // check the modified input
    CHECK(value_operations::are_equal(lhs_expected, lhs_actual));
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename enable = void>
class run_binary_assignment {
 public:
  void operator()(const std::string &) {}
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_binary_assignment<
    DataT, NumElementsT, OpT,
    typename std::enable_if_t<std::is_integral_v<DataT> ||
                              (std::is_floating_point_v<DataT> &&
                               !(std::is_same_v<OpT, op_assign_mod> ||
                                 std::is_same_v<OpT, op_assign_bor> ||
                                 std::is_same_v<OpT, op_assign_band> ||
                                 std::is_same_v<OpT, op_assign_bxor> ||
                                 std::is_same_v<OpT, op_assign_sl> ||
                                 std::is_same_v<OpT, op_assign_sr>))>> {
 public:
  void operator()(const std::string &operator_name) {
    INFO("for operator \"" << operator_name << "\": ");

    const auto constants = get_scalars();
    const auto functions = get_sequences<NumElementsT::value>();
    for_all_combinations<run_binary_assignment_sequence_scalar, DataT,
                         NumElementsT, OpT>(functions, constants);
    for_all_combinations<run_binary_assignment_sequence_sequence, DataT,
                         NumElementsT, OpT>(functions, functions);
  }
};

template <typename DataT>
class check_marray_operators_for_type {
 public:
  void operator()(const std::string &type_name) {
    INFO("for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();

#ifdef SYCL_CTS_COMPILING_WITH_COMPUTECPP
    WARN(
        "ComputeCPP gives runtime error for binary negation (~)."
        "Skipping the test case for binary negation.");
#endif

    static const auto unary_operators =
        named_type_pack<op_upos, op_uneg, op_pre_inc, op_pre_dec, op_lnot
    // ~ gives segmentation fault with char
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
                        ,
                        op_bnot
#endif
                        >::generate("unary +", "unary -", "pre ++", "pre --",
                                    "!"
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP
                                    ,
                                    "~"
#endif
        );
    for_all_combinations<run_unary, DataT>(num_elements, unary_operators);

    static const auto unary_post_operators =
        named_type_pack<op_post_inc, op_post_dec>::generate("post ++",
                                                            "post --");
    for_all_combinations<run_unary_post, DataT>(num_elements,
                                                unary_post_operators);

#if defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP) || \
    defined(SYCL_CTS_COMPILING_WITH_DPCPP)
    WARN(
        "ComputeCPP and DPCPP do not compile for logical AND (&&) and local OR "
        "(||). Skipping the test case.");
#endif

    static const auto binary_operators =
        named_type_pack<op_add, op_sub, op_mul, op_div, op_mod, op_bor, op_band,
                        op_bxor, op_sl, op_sr
    // && and || are not defined for floating-point types
    // && and || are ambiguous for 'const bool' and
    //  'const cl::sycl::marray<bool, 1>'
    // && and || for any? type 'static_assert failed due to
    //  requirement 'num_elements<bool>() == 2UL'
#if !(defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP) || \
      defined(SYCL_CTS_COMPILING_WITH_DPCPP))
                        ,
                        op_land, op_lor
#endif
                        >::generate("+", "-", "*", "/", "%", "|", "&", "^",
                                    "<<", ">>"
#if !(defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP) || \
      defined(SYCL_CTS_COMPILING_WITH_DPCPP))
                                    ,
                                    "&&", "||"
#endif
        );

    for_all_combinations<run_binary, DataT>(num_elements, binary_operators);

    static const auto binary_assignment_operators =
        named_type_pack<op_assign_add, op_assign_sub, op_assign_mul,
                        op_assign_div, op_assign_mod, op_assign_bor,
                        op_assign_band, op_assign_bxor, op_assign_sl,
                        op_assign_sr>::generate("+=", "-=", "*=", "/=", "%=",
                                                "|=", "&=", "^=", "<<=", ">>=");
    for_all_combinations<run_binary_assignment, DataT>(
        num_elements, binary_assignment_operators);
  }
};

}  // namespace marray_operators

#endif  // SYCLCTS_TESTS_MARRAY_MARRAY_OPERATOR_H
