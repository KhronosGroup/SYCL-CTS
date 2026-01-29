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

#include "../../util/type_traits.h"
#include "../common/common.h"
#include "../common/section_name_builder.h"
#include "marray_common.h"
#include "marray_operator_helper.h"

#include <string>
#include <valarray>

namespace marray_operators {

/**
 * @brief Define several sequences to initialize array instances. */

enum class init_sequence { inc, dec, ones, twos };

inline constexpr init_sequence all_init_sequences[] = {
    init_sequence::inc, init_sequence::dec, init_sequence::ones,
    init_sequence::twos};

template <typename DataT>
inline constexpr DataT get_seq_val(std::size_t num_elements, init_sequence seq,
                                   std::size_t i) {
  switch (seq) {
    case init_sequence::inc:
      return DataT(i + 1);
    case init_sequence::dec:
      return DataT(num_elements - i);
    case init_sequence::ones:
      return DataT(1);
    case init_sequence::twos:
      return DataT(2);
  }
}

inline std::string get_sequence_name(init_sequence seq) {
  switch (seq) {
    case init_sequence::inc:
      return "incrementing sequence";
    case init_sequence::dec:
      return "decrementing sequence";
    case init_sequence::ones:
      return "sequence of ones";
    case init_sequence::twos:
      return "sequence of twos";
  }
}

/**
 * @brief Define several constants to initialize scalar instances. */

enum class init_scalar { one, two };

static constexpr init_scalar all_init_scalars[] = {init_scalar::one,
                                                   init_scalar::two};

template <typename DataT>
inline DataT get_scalar_val(init_scalar sca) {
  switch (sca) {
    case init_scalar::one:
      return DataT(1);
    case init_scalar::two:
      return DataT(2);
  }
}

inline std::string get_scalar_name(init_scalar sca) {
  switch (sca) {
    case init_scalar::one:
      return "one (1)";
    case init_scalar::two:
      return "two (2)";
  }
}

template <typename DataT, typename NumElementsT>
struct operators_helper {
  static constexpr std::size_t NumElements = NumElementsT::value;
  using marray_t = sycl::marray<DataT, NumElements>;
  using varray_t = std::valarray<DataT>;

  template <typename array_type>
  static void init(array_type& ma, init_sequence seq) {
    for (std::size_t i = 0; i < NumElements; i++) {
      ma[i] = get_seq_val<DataT>(NumElements, seq, i);
    }
  }

  static void init(DataT& d, init_scalar sca) {
    d = get_scalar_val<DataT>(sca);
  }
};

// Division operations on floating points make no precision guarantees, so
// similar to native::divide we can skip checking them.
template <typename OpT, typename ElemT>
struct skip_result_check
    : std::bool_constant<(
          std::is_same_v<OpT, op_div> ||
          std::is_same_v<
              OpT, op_assign_div>)&&is_sycl_scalar_floating_point_v<ElemT>> {};

template <typename OpT, typename ElemT>
constexpr bool skip_result_check_v = skip_result_check<OpT, ElemT>::value;

template <typename OpT, typename ElemT, typename T1, typename T2>
bool are_equal_ignore_division(const T1& lhs, const T1& rhs) {
  // Division operations on floating points make no precision guarantees, so
  // similar to native::divide we can skip checking them here.
  constexpr bool is_div =
      std::is_same_v<OpT, op_div> || std::is_same_v<OpT, op_assign_div>;
  constexpr bool is_sycl_scalar_floating_point =
      std::is_same_v<ElemT, float> || std::is_same_v<ElemT, double> ||
      std::is_same_v<ElemT, sycl::half>;
  if constexpr (is_div && is_sycl_scalar_floating_point) return true;
  return value_operations::are_equal(lhs, rhs);
}

template <typename DataT, typename NumElementsT, typename OpT>
class run_unary_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(init_sequence seq, const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t val_actual;
    helper::init(val_actual, seq);
    auto res_actual = op(val_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

  template <typename ResT>
  static void run_on_device(init_sequence seq,
                            const std::valarray<ResT>& res_expected) {
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
              helper::init(val_actual, seq);
              res_actual_acc[0] = op(val_actual);
            });
          })
          .wait_and_throw();
    }

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

 public:
  void operator()() {
    for (const init_sequence seq : all_init_sequences) {
      INFO("for input (sequence) \"" << get_sequence_name(seq) << "\": ");

      OpT op;

      typename helper::varray_t val_expected(helper::NumElements);
      helper::init(val_expected, seq);
      auto res_expected = op(val_expected);

      run_on_host(seq, res_expected);

      run_on_device(seq, res_expected);
    }
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

    run_unary_sequence<DataT, NumElementsT, OpT>{}();
  }
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_unary_post_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(init_sequence seq,
                          const typename helper::varray_t& val_expected,
                          const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t val_actual;
    helper::init(val_actual, seq);
    auto res_actual = op(val_actual);

    // check the returned output
    CHECK(value_operations::are_equal(res_expected, res_actual));
    // check the modified input
    CHECK(value_operations::are_equal(val_expected, val_actual));
  }

  template <typename ResT>
  static void run_on_device(init_sequence seq,
                            const typename helper::varray_t& val_expected,
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
              helper::init(val_actual_acc[0], seq);
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
  void operator()() {
    for (const init_sequence seq : all_init_sequences) {
      INFO("for input (sequence) \"" << get_sequence_name(seq) << "\": ");

      OpT op;

      typename helper::varray_t val_expected(helper::NumElements);
      helper::init(val_expected, seq);
      auto res_expected = op(val_expected);

      run_on_host(seq, val_expected, res_expected);

      run_on_device(seq, val_expected, res_expected);
    }
  }
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_unary_post {
 public:
  void operator()(const std::string& operator_name) {
    INFO("for operator \"" << operator_name << "\": ");

    run_unary_post_sequence<DataT, NumElementsT, OpT>{}();
  }
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_binary_sequence_scalar {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(init_sequence seq, init_scalar sca,
                          const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t lhs_actual;
    helper::init(lhs_actual, seq);
    DataT rhs_actual;
    helper::init(rhs_actual, sca);
    auto res_actual = op(lhs_actual, rhs_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

  template <typename ResT>
  static void run_on_device(init_sequence seq, init_scalar sca,
                            const std::valarray<ResT>& res_expected) {
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
              helper::init(lhs_actual, seq);
              DataT rhs_actual;
              helper::init(rhs_actual, sca);
              res_actual_acc[0] = op(lhs_actual, rhs_actual);
            });
          })
          .wait_and_throw();
    }

    if constexpr (!skip_result_check_v<OpT, ResT>)
      CHECK(value_operations::are_equal(res_expected, res_actual));
  }

 public:
  void operator()() {
    for (const init_sequence seq : all_init_sequences) {
      for (const init_scalar sca : all_init_scalars) {
        INFO("for lhs (sequence) \"" << get_sequence_name(seq) << "\": ");
        INFO("for rhs (scalar) \"" << get_scalar_name(sca) << "\": ");

        OpT op;

        typename helper::varray_t lhs_expected(helper::NumElements);
        helper::init(lhs_expected, seq);
        DataT rhs_expected;
        helper::init(rhs_expected, sca);
        auto res_expected = op(lhs_expected, rhs_expected);

        run_on_host(seq, sca, res_expected);

        run_on_device(seq, sca, res_expected);
      }
    }
  }
};

inline constexpr bool init_seq_contains_too_big_values_for_shift_op(
    init_sequence seq, std::size_t seq_el_num,
    std::size_t max_shift_wo_undef_behavior) {
  return (seq == init_sequence::inc || seq == init_sequence::dec) &&
         seq_el_num > max_shift_wo_undef_behavior;
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
template <typename OpT>
inline constexpr bool test_case_is_invalid_for_shift_op(
    init_sequence rhs_seq, std::size_t seq_el_num) {
  constexpr int max_left_shift_wo_undef_behavior = 8;
  constexpr int max_right_shift_wo_undef_behavior = sizeof(int) - 1;
  if constexpr (std::is_same_v<OpT, op_sl> || std::is_same_v<OpT, op_assign_sl>)
    return init_seq_contains_too_big_values_for_shift_op(
        rhs_seq, seq_el_num, max_left_shift_wo_undef_behavior);
  if constexpr (std::is_same_v<OpT, op_sr> || std::is_same_v<OpT, op_assign_sr>)
    return init_seq_contains_too_big_values_for_shift_op(
        rhs_seq, seq_el_num, max_right_shift_wo_undef_behavior);
  return false;
}

template <typename OpT>
inline constexpr bool is_shift_op() {
  return std::is_same_v<OpT, op_sl> || std::is_same_v<OpT, op_sr> ||
         std::is_same_v<OpT, op_assign_sl> || std::is_same_v<OpT, op_assign_sr>;
}

template <typename DataT, typename NumElementsT, typename OpT>
class run_binary_scalar_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(init_scalar sca, init_sequence seq,
                          const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    DataT lhs_actual;
    helper::init(lhs_actual, sca);
    typename helper::marray_t rhs_actual;
    helper::init(rhs_actual, seq);
    auto res_actual = op(lhs_actual, rhs_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

  template <typename ResT>
  static void run_on_device(init_scalar sca, init_sequence seq,
                            const std::valarray<ResT>& res_expected) {
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
              helper::init(lhs_actual, sca);
              typename helper::marray_t rhs_actual;
              helper::init(rhs_actual, seq);
              res_actual_acc[0] = op(lhs_actual, rhs_actual);
            });
          })
          .wait_and_throw();
    }

    if constexpr (!skip_result_check_v<OpT, ResT>)
      CHECK(value_operations::are_equal(res_expected, res_actual));
  }

 public:
  void operator()() {
    for (const init_scalar sca : all_init_scalars) {
      for (const init_sequence seq : all_init_sequences) {
        if (is_shift_op<OpT>() &&
            test_case_is_invalid_for_shift_op<OpT>(seq, helper::NumElements))
          continue;

        INFO("for lhs (scalar) \"" << get_scalar_name(sca) << "\": ");
        INFO("for rhs (sequence) \"" << get_sequence_name(seq) << "\": ");

        OpT op;

        DataT lhs_expected;
        helper::init(lhs_expected, sca);
        typename helper::varray_t rhs_expected(helper::NumElements);
        helper::init(rhs_expected, seq);
        auto res_expected = op(lhs_expected, rhs_expected);

        run_on_host(sca, seq, res_expected);

        run_on_device(sca, seq, res_expected);
      }
    }
  }
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_binary_sequence_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(init_sequence seq1, init_sequence seq2,
                          const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t lhs_actual;
    helper::init(lhs_actual, seq1);
    typename helper::marray_t rhs_actual;
    helper::init(rhs_actual, seq2);
    auto res_actual = op(lhs_actual, rhs_actual);

    CHECK(value_operations::are_equal(res_expected, res_actual));
  }

  template <typename ResT>
  static void run_on_device(init_sequence seq1, init_sequence seq2,
                            const std::valarray<ResT>& res_expected) {
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
              helper::init(lhs_actual, seq1);
              typename helper::marray_t rhs_actual;
              helper::init(rhs_actual, seq2);
              res_actual_acc[0] = op(lhs_actual, rhs_actual);
            });
          })
          .wait_and_throw();
    }

    if constexpr (!skip_result_check_v<OpT, ResT>)
      CHECK(value_operations::are_equal(res_expected, res_actual));
  }

 public:
  void operator()() {
    for (const init_sequence seq1 : all_init_sequences) {
      for (const init_sequence seq2 : all_init_sequences) {
        if (is_shift_op<OpT>() &&
            test_case_is_invalid_for_shift_op<OpT>(seq2, helper::NumElements))
          continue;

        INFO("for lhs (sequence) \"" << get_sequence_name(seq1) << "\": ");
        INFO("for rhs (sequence) \"" << get_sequence_name(seq2) << "\": ");

        OpT op;

        typename helper::varray_t lhs_expected(helper::NumElements);
        helper::init(lhs_expected, seq1);
        typename helper::varray_t rhs_expected(helper::NumElements);
        helper::init(rhs_expected, seq2);
        auto res_expected = op(lhs_expected, rhs_expected);

        run_on_host(seq1, seq2, res_expected);

        run_on_device(seq1, seq2, res_expected);
      }
    }
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

    run_binary_sequence_scalar<DataT, NumElementsT, OpT>{}();
    run_binary_scalar_sequence<DataT, NumElementsT, OpT>{}();
    run_binary_sequence_sequence<DataT, NumElementsT, OpT>{}();
  }
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_binary_assignment_sequence_scalar {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(init_sequence seq, init_scalar sca,
                          const typename helper::varray_t& lhs_expected,
                          const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t lhs_actual;
    helper::init(lhs_actual, seq);
    DataT rhs_actual;
    helper::init(rhs_actual, sca);
    auto res_actual = op(lhs_actual, rhs_actual);

    // check the returned output
    CHECK(value_operations::are_equal(res_expected, res_actual));
    // check the modified input
    CHECK(value_operations::are_equal(lhs_expected, lhs_actual));
  }

  template <typename ResT>
  static void run_on_device(init_sequence seq, init_scalar sca,
                            const typename helper::varray_t& lhs_expected,
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
              helper::init(lhs_actual_acc[0], seq);
              DataT rhs_actual;
              helper::init(rhs_actual, sca);
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
  void operator()() {
    for (const init_sequence seq : all_init_sequences) {
      for (const init_scalar sca : all_init_scalars) {
        INFO("for lhs (sequence) \"" << get_sequence_name(seq) << "\": ");
        INFO("for rhs (scalar) \"" << get_scalar_name(sca) << "\": ");

        OpT op;

        typename helper::varray_t lhs_expected(helper::NumElements);
        helper::init(lhs_expected, seq);
        DataT rhs_expected;
        helper::init(rhs_expected, sca);
        auto res_expected = op(lhs_expected, rhs_expected);

        run_on_host(seq, sca, lhs_expected, res_expected);

        run_on_device(seq, sca, lhs_expected, res_expected);
      }
    }
  }
};

template <typename DataT, typename NumElementsT, typename OpT>
class run_binary_assignment_sequence_sequence {
  using helper = operators_helper<DataT, NumElementsT>;

  template <typename ResT>
  static void run_on_host(init_sequence seq1, init_sequence seq2,
                          const typename helper::varray_t& lhs_expected,
                          const ResT& res_expected) {
    INFO("validation on host");

    OpT op;

    typename helper::marray_t lhs_actual;
    helper::init(lhs_actual, seq1);
    typename helper::marray_t rhs_actual;
    helper::init(rhs_actual, seq2);
    auto res_actual = op(lhs_actual, rhs_actual);

    // check the returned output
    CHECK(value_operations::are_equal(res_expected, res_actual));
    // check the modified input
    CHECK(value_operations::are_equal(lhs_expected, lhs_actual));
  }

  template <typename ResT>
  static void run_on_device(init_sequence seq1, init_sequence seq2,
                            const typename helper::varray_t& lhs_expected,
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
              helper::init(lhs_actual_acc[0], seq1);
              typename helper::marray_t rhs_actual;
              helper::init(rhs_actual, seq2);
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
  void operator()() {
    for (const init_sequence seq1 : all_init_sequences) {
      for (const init_sequence seq2 : all_init_sequences) {
        if (is_shift_op<OpT>() &&
            test_case_is_invalid_for_shift_op<OpT>(seq2, helper::NumElements))
          continue;

        INFO("for lhs (sequence) \"" << get_sequence_name(seq1) << "\": ");
        INFO("for rhs (sequence) \"" << get_sequence_name(seq2) << "\": ");

        OpT op;

        typename helper::varray_t lhs_expected(helper::NumElements);
        helper::init(lhs_expected, seq1);
        typename helper::varray_t rhs_expected(helper::NumElements);
        helper::init(rhs_expected, seq2);
        auto res_expected = op(lhs_expected, rhs_expected);

        run_on_host(seq1, seq2, lhs_expected, res_expected);

        run_on_device(seq1, seq2, lhs_expected, res_expected);
      }
    }
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

    run_binary_assignment_sequence_scalar<DataT, NumElementsT, OpT>{}();
    run_binary_assignment_sequence_sequence<DataT, NumElementsT, OpT>{}();
  }
};

template <typename DataT>
class check_marray_pre_unary_operators_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("prefix unary operators for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();
    static const auto unary_operators = [] {
      if constexpr (std::is_same_v<DataT, bool>) {
        // marray<bool> does not have operator++/--
        return named_type_pack<op_upos, op_uneg, op_lnot, op_bnot>::generate(
            "unary +", "unary -", "!", "~");
      } else {
        return named_type_pack<op_upos, op_uneg, op_pre_inc, op_pre_dec,
                               op_lnot, op_bnot>::generate("unary +", "unary -",
                                                           "pre ++", "pre --",
                                                           "!", "~");
      }
    }();
    for_all_combinations<run_unary, DataT>(num_elements, unary_operators);
  }
};

template <typename DataT>
class check_marray_post_unary_operators_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("suffix unary operators for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();

    // marray<bool> does not have operator++/--
    if constexpr (!std::is_same_v<DataT, bool>) {
      static const auto unary_post_operators =
          named_type_pack<op_post_inc, op_post_dec>::generate("post ++",
                                                              "post --");
      for_all_combinations<run_unary_post, DataT>(num_elements,
                                                  unary_post_operators);
    }
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
