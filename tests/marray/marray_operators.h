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

#ifndef SYCLCTS_TESTS_MARRAY_MARRAY_OPERATOR_H
#define SYCLCTS_TESTS_MARRAY_MARRAY_OPERATOR_H

#include "../common/common.h"
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
  static void init(array_type& ma) {
    for (std::size_t i = 0; i < NumElements; i++) {
      ma[i] = init_func::template init<DataT>(i);
    }
  }

  template <typename init_func>
  static void init(DataT& d) {
    d = init_func::template init<DataT>();
  }
};

// Division operations on floating points make no precision guarantees, so
// similar to native::divide we can skip checking them.
template <typename OpT, typename ElemT>
struct skip_result_check
    : std::bool_constant<
          (std::is_same_v<OpT, op_div> || std::is_same_v<OpT, op_assign_div>)&&(
              std::is_same_v<ElemT, float> || std::is_same_v<ElemT, double> ||
              std::is_same_v<ElemT, sycl::half>)> {};

template <typename OpT, typename ElemT>
constexpr bool skip_result_check_v = skip_result_check<OpT, ElemT>::value;

template <typename OpT, typename ElemT, typename T1, typename T2>
bool are_equal_ignore_division(const T1& lhs, const T1& rhs) {
  // Division operations on floating points make no precision guarantees, so
  // similar to native::divide we can skip checking them here.
  constexpr bool is_div =
      std::is_same_v<OpT, op_div> || std::is_same_v<OpT, op_assign_div>;
  constexpr bool is_sycl_floating_point = std::is_same_v<ElemT, float> ||
                                          std::is_same_v<ElemT, double> ||
                                          std::is_same_v<ElemT, sycl::half>;
  if constexpr (is_div && is_sycl_floating_point) return true;
  return value_operations::are_equal(lhs, rhs);
}

/**
 * @brief Define several sequences to initialize array instances. */

struct seq_inc {
  template <typename DataT>
  static DataT init(std::size_t i) {
    return DataT(i + 1);
  }
};

template <std::size_t NumElements>
struct seq_dec {
  template <typename DataT>
  static DataT init(std::size_t i) {
    return DataT(NumElements - i);
  }
};

struct seq_one {
  template <typename DataT>
  static DataT init(std::size_t) {
    return {1};
  }
};

struct seq_two {
  template <typename DataT>
  static DataT init(std::size_t) {
    return DataT(2);
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
  template <typename DataT>
  static DataT init() {
    return DataT(1);
  }
};

struct sca_two {
  template <typename DataT>
  static DataT init() {
    return DataT(2);
  }
};

inline auto get_scalars() {
  return named_type_pack<sca_one, sca_two>::generate("one (1)", "two (2)");
}

template <typename DataT, typename NumElementsT, typename OpT,
          typename SequenceT>
class run_unary_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t val_actual;
    helper::template init<SequenceT>(val_actual);
    auto res_actual = op(val_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

  template <typename ResT>
  static void run_on_device(const std::valarray<ResT>& res_expected) {
    INFO("validation on device");

    auto queue = sycl_cts::util::get_cts_object::queue();

    sycl::marray<ResT, helper::NumElements> res_actual;
    {
      sycl::buffer<decltype(res_actual), 1> res_actual_buff{&res_actual,
                                                            sycl::range<1>{1}};

      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor res_actual_acc{res_actual_buff, cgh,
                                          sycl::write_only};
            cgh.single_task([=]() {
              OpT op;
              typename helper::marray_t val_actual;
              helper::template init<SequenceT>(val_actual);
              res_actual_acc[0] = op(val_actual);
            });
          })
          .wait_and_throw();
    }

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

 public:
  void operator()(const std::string& function_name) {
    INFO("for input (sequence) \"" << function_name << "\": ");

    OpT op;

    typename helper::varray_t val_expected(helper::NumElements);
    helper::template init<SequenceT>(val_expected);
    auto res_expected = op(val_expected);

    run_on_host(res_expected);

    run_on_device(res_expected);
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename enable = void>
class run_unary {
 public:
  void operator()(const std::string&) {}
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_unary<DataT, NumElementsT, OpT,
                std::enable_if_t<std::is_integral_v<DataT> ||
                                 (std::is_floating_point_v<DataT> &&
                                  !std::is_same_v<OpT, op_bnot>)>> {
 public:
  void operator()(const std::string& operator_name) {
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

  template <typename ResT>
  static void run_on_host(const typename helper::varray_t& val_expected,
                          const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t val_actual;
    helper::template init<SequenceT>(val_actual);
    auto res_actual = op(val_actual);

    // check the returned output
    CHECK(value_operations::are_equal(res_expected, res_actual));
    // check the modified input
    CHECK(value_operations::are_equal(val_expected, val_actual));
  }

  template <typename ResT>
  static void run_on_device(const typename helper::varray_t& val_expected,
                            const std::valarray<ResT>& res_expected) {
    INFO("validation on device");

    auto queue = sycl_cts::util::get_cts_object::queue();

    typename helper::marray_t val_actual;
    sycl::marray<ResT, helper::NumElements> res_actual;
    {
      sycl::buffer<decltype(val_actual), 1> val_actual_buff{&val_actual,
                                                            sycl::range<1>{1}};
      sycl::buffer<decltype(res_actual), 1> res_actual_buff{&res_actual,
                                                            sycl::range<1>{1}};

      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor val_actual_acc{val_actual_buff, cgh,
                                          sycl::read_write};
            sycl::accessor res_actual_acc{res_actual_buff, cgh,
                                          sycl::write_only};
            cgh.single_task([=]() {
              OpT op;
              helper::template init<SequenceT>(val_actual_acc[0]);
              res_actual_acc[0] = op(val_actual_acc[0]);
            });
          })
          .wait_and_throw();
    }

    // check the returned output
    CHECK(value_operations::are_equal(res_expected, res_actual));
    // check the modified input
    CHECK(value_operations::are_equal(val_expected, val_actual));
  }

 public:
  void operator()(const std::string& function_name) {
    INFO("for input (sequence) \"" << function_name << "\": ");

    OpT op;

    typename helper::varray_t val_expected(helper::NumElements);
    helper::template init<SequenceT>(val_expected);
    auto res_expected = op(val_expected);

    run_on_host(val_expected, res_expected);

    run_on_device(val_expected, res_expected);
  }
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_unary_post {
 public:
  void operator()(const std::string& operator_name) {
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

  template <typename ResT>
  static void run_on_host(const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t lhs_actual;
    helper::template init<SequenceT>(lhs_actual);
    DataT rhs_actual;
    helper::template init<ScalarT>(rhs_actual);
    auto res_actual = op(lhs_actual, rhs_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

  template <typename ResT>
  static void run_on_device(const std::valarray<ResT>& res_expected) {
    INFO("validation on device");

    auto queue = sycl_cts::util::get_cts_object::queue();

    sycl::marray<ResT, helper::NumElements> res_actual;
    {
      sycl::buffer<decltype(res_actual), 1> res_actual_buff{&res_actual,
                                                            sycl::range<1>{1}};

      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor res_actual_acc{res_actual_buff, cgh,
                                          sycl::write_only};
            cgh.single_task([=]() {
              OpT op;
              typename helper::marray_t lhs_actual;
              helper::template init<SequenceT>(lhs_actual);
              DataT rhs_actual;
              helper::template init<ScalarT>(rhs_actual);
              res_actual_acc[0] = op(lhs_actual, rhs_actual);
            });
          })
          .wait_and_throw();
    }

    if constexpr (!skip_result_check_v<OpT, ResT>)
      CHECK(value_operations::are_equal(res_expected, res_actual));
  }

 public:
  void operator()(const std::string& function_name,
                  const std::string& constant_name) {
    INFO("for lhs (sequence) \"" << function_name << "\": ");
    INFO("for rhs (scalar) \"" << constant_name << "\": ");

    OpT op;

    typename helper::varray_t lhs_expected(helper::NumElements);
    helper::template init<SequenceT>(lhs_expected);
    DataT rhs_expected;
    helper::template init<ScalarT>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    run_on_host(res_expected);

    run_on_device(res_expected);
  }
};

template <typename InitSeqForRhs, std::size_t seq_el_num>
inline constexpr bool init_seq_contains_too_big_values_for_shift_op(
    std::size_t max_shift_wo_undef_behavior) {
  return (std::is_same_v<InitSeqForRhs, seq_inc> ||
          std::is_same_v<InitSeqForRhs, seq_dec<seq_el_num>>)&&seq_el_num >
         max_shift_wo_undef_behavior;
}

/**
  @brief The function checks that for particular test case there is no
  undefined behavior for left or right shift operation. In case of shift
  operation too large value of right hand side argument can lead to undefined
  behavior. For initialization sequences: seq_inc, seq_dec the right hand side
  argument can take values up to 64. So we should exclude test cases for such
  big shift values. Left shift operation by 8 bits is guaranteed legal for our
  tests because maximum value after shift operation will be 64 * 2^8 = 2^14 (64
  is maximum value in initialization sequence) that can be stored inside int
  type (which should be at least 16 bits width) and inside wider types. It is
  also valid for shift operation with small integral types (such as char)
  because of its integral promotions to int before the operation execution. For
  right shift operation with right hand side argument is greater or equal to
  the number of bits in the promoted left operand, the behavior is undefined.
  So right shift operation by N bits with N is less than sizeof(int) - 1 is
  guaranteed legal for type int, wider types and for small integral types
  because of its integral promotions to int. */
template <typename OpT, typename InitSeqForRhs, std::size_t seq_el_num>
inline constexpr bool test_case_is_invalid_for_shift_op() {
  constexpr int max_left_shift_wo_undef_behavior = 8;
  constexpr int max_right_shift_wo_undef_behavior = sizeof(int) - 1;
  if constexpr (std::is_same_v<OpT, op_sl> || std::is_same_v<OpT, op_assign_sl>)
    return init_seq_contains_too_big_values_for_shift_op<InitSeqForRhs,
                                                         seq_el_num>(
        max_left_shift_wo_undef_behavior);
  if constexpr (std::is_same_v<OpT, op_sr> || std::is_same_v<OpT, op_assign_sr>)
    return init_seq_contains_too_big_values_for_shift_op<InitSeqForRhs,
                                                         seq_el_num>(
        max_right_shift_wo_undef_behavior);
  return false;
}

template <typename OpT>
inline constexpr bool is_shift_op() {
  return std::is_same_v<OpT, op_sl> || std::is_same_v<OpT, op_sr> ||
         std::is_same_v<OpT, op_assign_sl> || std::is_same_v<OpT, op_assign_sr>;
}

template <typename DataT, typename NumElementsT, typename OpT, typename ScalarT,
          typename SequenceT>
class run_binary_scalar_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    DataT lhs_actual;
    helper::template init<ScalarT>(lhs_actual);
    typename helper::marray_t rhs_actual;
    helper::template init<SequenceT>(rhs_actual);
    auto res_actual = op(lhs_actual, rhs_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

  template <typename ResT>
  static void run_on_device(const std::valarray<ResT>& res_expected) {
    INFO("validation on device");

    auto queue = sycl_cts::util::get_cts_object::queue();

    sycl::marray<ResT, helper::NumElements> res_actual;
    {
      sycl::buffer<decltype(res_actual), 1> res_actual_buff{&res_actual,
                                                            sycl::range<1>{1}};

      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor res_actual_acc{res_actual_buff, cgh,
                                          sycl::write_only};
            cgh.single_task([=]() {
              OpT op;
              DataT lhs_actual;
              helper::template init<ScalarT>(lhs_actual);
              typename helper::marray_t rhs_actual;
              helper::template init<SequenceT>(rhs_actual);
              res_actual_acc[0] = op(lhs_actual, rhs_actual);
            });
          })
          .wait_and_throw();
    }

    if constexpr (!skip_result_check_v<OpT, ResT>)
      CHECK(value_operations::are_equal(res_expected, res_actual));
  }

 public:
  void operator()(const std::string& constant_name,
                  const std::string& function_name) {
    if constexpr (is_shift_op<OpT>() &&
                  test_case_is_invalid_for_shift_op<OpT, SequenceT,
                                                    NumElementsT::value>()) {
      return;
    }

    INFO("for lhs (scalar) \"" << constant_name << "\": ");
    INFO("for rhs (sequence) \"" << function_name << "\": ");

    OpT op;

    DataT lhs_expected;
    helper::template init<ScalarT>(lhs_expected);
    typename helper::varray_t rhs_expected(helper::NumElements);
    helper::template init<SequenceT>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    run_on_host(res_expected);

    run_on_device(res_expected);
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename SequenceT1, typename SequenceT2>
class run_binary_sequence_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t lhs_actual;
    helper::template init<SequenceT1>(lhs_actual);
    typename helper::marray_t rhs_actual;
    helper::template init<SequenceT2>(rhs_actual);
    auto res_actual = op(lhs_actual, rhs_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

  template <typename ResT>
  static void run_on_device(const std::valarray<ResT>& res_expected) {
    INFO("validation on device");

    auto queue = sycl_cts::util::get_cts_object::queue();

    sycl::marray<ResT, helper::NumElements> res_actual;
    {
      sycl::buffer<decltype(res_actual), 1> res_actual_buff{&res_actual,
                                                            sycl::range<1>{1}};

      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor res_actual_acc{res_actual_buff, cgh,
                                          sycl::write_only};
            cgh.single_task([=]() {
              OpT op;
              typename helper::marray_t lhs_actual;
              helper::template init<SequenceT1>(lhs_actual);
              typename helper::marray_t rhs_actual;
              helper::template init<SequenceT2>(rhs_actual);
              res_actual_acc[0] = op(lhs_actual, rhs_actual);
            });
          })
          .wait_and_throw();
    }

    if constexpr (!skip_result_check_v<OpT, ResT>)
      CHECK(value_operations::are_equal(res_expected, res_actual));
  }

 public:
  void operator()(const std::string& function_name_1,
                  const std::string& function_name_2) {
    if constexpr (is_shift_op<OpT>() &&
                  test_case_is_invalid_for_shift_op<OpT, SequenceT2,
                                                    NumElementsT::value>()) {
      return;
    }

    INFO("for lhs (sequence) \"" << function_name_1 << "\": ");
    INFO("for rhs (sequence) \"" << function_name_2 << "\": ");

    OpT op;

    typename helper::varray_t lhs_expected(helper::NumElements);
    helper::template init<SequenceT1>(lhs_expected);
    typename helper::varray_t rhs_expected(helper::NumElements);
    helper::template init<SequenceT2>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    run_on_host(res_expected);

    run_on_device(res_expected);
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename enable = void>
class run_binary {
 public:
  void operator()(const std::string&) {}
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
  void operator()(const std::string& operator_name) {
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

  template <typename ResT>
  static void run_on_host(const typename helper::varray_t& lhs_expected,
                          const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

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

  template <typename ResT>
  static void run_on_device(const typename helper::varray_t& lhs_expected,
                            const std::valarray<ResT>& res_expected) {
    INFO("validation on device");

    auto queue = sycl_cts::util::get_cts_object::queue();

    typename helper::marray_t lhs_actual;
    sycl::marray<ResT, helper::NumElements> res_actual;
    {
      sycl::buffer<decltype(lhs_actual), 1> lhs_actual_buff{&lhs_actual,
                                                            sycl::range<1>{1}};
      sycl::buffer<decltype(res_actual), 1> res_actual_buff{&res_actual,
                                                            sycl::range<1>{1}};

      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor lhs_actual_acc{lhs_actual_buff, cgh,
                                          sycl::read_write};
            sycl::accessor res_actual_acc{res_actual_buff, cgh,
                                          sycl::write_only};
            cgh.single_task([=]() {
              OpT op;
              typename helper::marray_t lhs_actual;
              helper::template init<SequenceT>(lhs_actual_acc[0]);
              DataT rhs_actual;
              helper::template init<ScalarT>(rhs_actual);
              res_actual_acc[0] = op(lhs_actual_acc[0], rhs_actual);
            });
          })
          .wait_and_throw();
    }

    if constexpr (!skip_result_check_v<OpT, ResT>) {
      // check the returned output
      CHECK(value_operations::are_equal(res_expected, res_actual));
      // check the modified input
      CHECK(value_operations::are_equal(lhs_expected, lhs_actual));
    }
  }

 public:
  void operator()(const std::string& function_name,
                  const std::string& constant_name) {
    INFO("for lhs (sequence) \"" << function_name << "\": ");
    INFO("for rhs (scalar) \"" << constant_name << "\": ");

    OpT op;

    typename helper::varray_t lhs_expected(helper::NumElements);
    helper::template init<SequenceT>(lhs_expected);
    DataT rhs_expected;
    helper::template init<ScalarT>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    run_on_host(lhs_expected, res_expected);

    run_on_device(lhs_expected, res_expected);
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename SequenceT1, typename SequenceT2>
class run_binary_assignment_sequence_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(const typename helper::varray_t& lhs_expected,
                          const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

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

  template <typename ResT>
  static void run_on_device(const typename helper::varray_t& lhs_expected,
                            const std::valarray<ResT>& res_expected) {
    INFO("validation on device");

    auto queue = sycl_cts::util::get_cts_object::queue();

    typename helper::marray_t lhs_actual;
    sycl::marray<ResT, helper::NumElements> res_actual;
    {
      sycl::buffer<decltype(lhs_actual), 1> lhs_actual_buff{&lhs_actual,
                                                            sycl::range<1>{1}};
      sycl::buffer<decltype(res_actual), 1> res_actual_buff{&res_actual,
                                                            sycl::range<1>{1}};

      queue
          .submit([&](sycl::handler& cgh) {
            sycl::accessor lhs_actual_acc{lhs_actual_buff, cgh,
                                          sycl::read_write};
            sycl::accessor res_actual_acc{res_actual_buff, cgh,
                                          sycl::write_only};
            cgh.single_task([=]() {
              OpT op;
              typename helper::marray_t lhs_actual;
              helper::template init<SequenceT1>(lhs_actual_acc[0]);
              typename helper::marray_t rhs_actual;
              helper::template init<SequenceT2>(rhs_actual);
              res_actual_acc[0] = op(lhs_actual_acc[0], rhs_actual);
            });
          })
          .wait_and_throw();
    }

    if constexpr (!skip_result_check_v<OpT, ResT>) {
      // check the returned output
      CHECK(value_operations::are_equal(res_expected, res_actual));
      // check the modified input
      CHECK(value_operations::are_equal(lhs_expected, lhs_actual));
    }
  }

 public:
  void operator()(const std::string& function_name_1,
                  const std::string& function_name_2) {
    if constexpr (is_shift_op<OpT>() &&
                  test_case_is_invalid_for_shift_op<OpT, SequenceT2,
                                                    NumElementsT::value>()) {
      return;
    }

    INFO("for lhs (sequence) \"" << function_name_1 << "\": ");
    INFO("for rhs (sequence) \"" << function_name_2 << "\": ");

    OpT op;

    typename helper::varray_t lhs_expected(helper::NumElements);
    helper::template init<SequenceT1>(lhs_expected);
    typename helper::varray_t rhs_expected(helper::NumElements);
    helper::template init<SequenceT2>(rhs_expected);
    auto res_expected = op(lhs_expected, rhs_expected);

    run_on_host(lhs_expected, res_expected);

    run_on_device(lhs_expected, res_expected);
  }
};

template <typename DataT, typename NumElementsT, typename OpT,
          typename enable = void>
class run_binary_assignment {
 public:
  void operator()(const std::string&) {}
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
  void operator()(const std::string& operator_name) {
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
class check_marray_pre_unary_operators_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("prefix unary operators for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();

    static const auto unary_operators =
        named_type_pack<op_upos, op_uneg, op_pre_inc, op_pre_dec, op_lnot,
                        op_bnot>::generate("unary +", "unary -", "pre ++",
                                           "pre --", "!", "~");
    for_all_combinations<run_unary, DataT>(num_elements, unary_operators);
  }
};

template <typename DataT>
class check_marray_post_unary_operators_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("suffix unary operators for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();

    static const auto unary_post_operators =
        named_type_pack<op_post_inc, op_post_dec>::generate("post ++",
                                                            "post --");
    for_all_combinations<run_unary_post, DataT>(num_elements,
                                                unary_post_operators);
  }
};

template <typename DataT>
class check_marray_arithmetic_binary_operators_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("arithmetic binary operators for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();

    static const auto binary_operators =
        named_type_pack<op_add, op_sub, op_mul, op_div, op_mod>::generate(
            "+", "-", "*", "/", "%");

    for_all_combinations<run_binary, DataT>(num_elements, binary_operators);
  }
};

template <typename DataT>
class check_marray_bitwise_binary_operators_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("bitwise binary operators for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();

    static const auto binary_operators =
        named_type_pack<op_bor, op_band, op_bxor, op_sl, op_sr>::generate(
            "|", "&", "^", "<<", ">>");

    for_all_combinations<run_binary, DataT>(num_elements, binary_operators);
  }
};

template <typename DataT>
class check_marray_relational_binary_operators_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("relational binary operators for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();

    static const auto binary_operators =
        named_type_pack<op_eq, op_not_eq, op_less, op_grater, op_less_eq,
                        op_grater_eq, op_land, op_lor>::generate("==", "!=",
                                                                 "<", ">", "<=",
                                                                 ">=", "&&",
                                                                 "||");

    for_all_combinations<run_binary, DataT>(num_elements, binary_operators);
  }
};

template <typename DataT>
class check_marray_arithmetic_assignment_operators_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("arithmetic assignment for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();

    static const auto binary_assignment_operators =
        named_type_pack<op_assign_add, op_assign_sub, op_assign_mul,
                        op_assign_div, op_assign_mod>::generate("+=", "-=",
                                                                "*=", "/=",
                                                                "%=");
    for_all_combinations<run_binary_assignment, DataT>(
        num_elements, binary_assignment_operators);
  }
};

template <typename DataT>
class check_marray_bitwise_assignment_operators_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("bitwise assignment for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();

    static const auto binary_assignment_operators =
        named_type_pack<op_assign_bor, op_assign_band, op_assign_bxor,
                        op_assign_sl, op_assign_sr>::generate("|=", "&=", "^=",
                                                              "<<=", ">>=");
    for_all_combinations<run_binary_assignment, DataT>(
        num_elements, binary_assignment_operators);
  }
};

}  // namespace marray_operators

#endif  // SYCLCTS_TESTS_MARRAY_MARRAY_OPERATOR_H
