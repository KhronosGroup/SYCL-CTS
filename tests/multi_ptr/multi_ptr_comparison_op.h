
/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides code for multi_ptr comparison operators
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_MULTI_PTR_COMPARISON_OP_H
#define __SYCLCTS_TESTS_MULTI_PTR_COMPARISON_OP_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

#include <array>    // for std::array
#include <cstddef>  // for std::size_t

namespace multi_ptr_comparison_op {

/**
 * @brief Provides functor for verification on multi_ptr comparison operators
 * @tparam T Current data type
 * @tparam AddrSpaceT sycl::access::address_space enumeration's field
 * @tparam IsDecoratedT sycl::access::decorated enumeration's field
 */
template <typename T, typename AddrSpaceT, typename IsDecoratedT>
class run_multi_ptr_comparison_op_test {
  static constexpr sycl::access::address_space space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;
  T low_value = 1;
  T great_value = 2;
  using multi_ptr_t = sycl::multi_ptr<T, space, decorated>;
  sycl::range m_r = sycl::range(1);

  template <typename TestActionA, std::size_t N>
  void run_test(sycl::queue &queue, TestActionA test_action,
                std::array<bool, N> &test_results) {
    sycl::buffer<T> low_value_buffer(&low_value, m_r);
    sycl::buffer<T> great_value_buffer(&great_value, m_r);
    sycl::buffer<bool> nullptr_nullptr_res_buffer(test_results.data(),
                                                  sycl::range(N));
    queue.submit([&](sycl::handler &cgh) {
      auto low_value_acc =
          low_value_buffer.template get_access<sycl::access_mode::read>(cgh);
      auto great_value_acc =
          great_value_buffer.template get_access<sycl::access_mode::read>(cgh);
      auto nullptr_nullptr_res_acc =
          nullptr_nullptr_res_buffer
              .template get_access<sycl::access_mode::write>(cgh);

      if constexpr (space == sycl::access::address_space::global_space) {
        cgh.single_task([=] {
          test_action(low_value_acc, great_value_acc, nullptr_nullptr_res_acc);
        });
      } else {
        cgh.parallel_for(sycl::nd_range(r, r), [=](sycl::nd_item item) {
          test_action(low_value_acc, great_value_acc, nullptr_nullptr_res_acc);
        });
      }
    });
  }

 public:
  /**
   * @param type_name Current data type string representation
   * @param address_space_name Current sycl::access::address_space string
   *        representation
   * @param is_decorated_name Current sycl::access::decorated string
   *        representation
   */
  void operator()(const std::string &type_name,
                  const std::string &address_space_name,
                  const std::string &is_decorated_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();
    SECTION(
        section_name(
            "Check multi_ptr operator==(const multi_ptr&, const multi_ptr&)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Expected value that will be returned after "lower_mptr == lower_mptr"
      // operator will be called
      bool first_first_mptr_expected_val = true;
      // Expected value that will be returned after "lower_mptr == greater_mptr"
      // operator will be called
      bool first_second_mptr_expected_val = false;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "lower_mptr == lower_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "lower_mptr == greater_mptr" operator will be called
      std::array<bool, 2> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t mptr_1(low_value_acc);
        multi_ptr_t mptr_2(great_value_acc);

        result_acc[0] = mptr_1 == mptr_1;
        result_acc[1] = mptr_1 == mptr_2;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == first_first_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == first_second_mptr_expected_val);
    }

    SECTION(
        section_name(
            "Check multi_ptr operator!=(const multi_ptr&, const multi_ptr&)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Expected value that will be returned after "lower_mptr != lower_mptr"
      // operator will be called
      bool first_first_mptr_expected_val = false;
      // Expected value that will be returned after "lower_mptr != greater_mptr"
      // operator will be called
      bool first_second_mptr_expected_val = true;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "lower_mptr != lower_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "lower_mptr != greater_mptr" operator will be called
      std::array<bool, 2> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t mptr_1(low_value_acc);
        multi_ptr_t mptr_2(great_value_acc);

        result_acc[0] = mptr_1 != mptr_1;
        result_acc[1] = mptr_1 != mptr_2;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == first_first_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == first_second_mptr_expected_val);
    }
    SECTION(section_name(
                "Check multi_ptr operator<(const multi_ptr&, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Expected value that will be returned after "lower_mptr < lower_mptr"
      // operator will be called
      bool first_first_mptr_expected_val = false;
      // Expected value that will be returned after "lower_mptr < greater_mptr"
      // operator will be called
      bool first_second_mptr_expected_val = true;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "lower_mptr < lower_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "lower_mptr < greater_mptr" operator will be called
      std::array<bool, 2> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t mptr_1(low_value_acc);
        multi_ptr_t mptr_2(great_value_acc);

        result_acc[0] = mptr_1 < mptr_1;
        result_acc[1] = mptr_1 < mptr_2;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == first_first_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == first_second_mptr_expected_val);
    }

    SECTION(section_name(
                "Check multi_ptr operator>(const multi_ptr&, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Expected value that will be returned after "lower_mptr > greater_mptr"
      // operator will be called
      bool first_first_mptr_expected_val = false;
      // Expected value that will be returned after "greater_mptr > lower_mptr"
      // operator will be called
      bool first_second_mptr_expected_val = true;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "lower_mptr > greater_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "greater_mptr > lower_mptr" operator will be called
      std::array<bool, 2> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t mptr_1(low_value_acc);
        multi_ptr_t mptr_2(great_value_acc);

        result_acc[0] = mptr_1 > mptr_2;
        result_acc[1] = mptr_2 > mptr_1;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == first_first_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == first_second_mptr_expected_val);
    }

    SECTION(
        section_name(
            "Check multi_ptr operator<=(const multi_ptr&, const multi_ptr&)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Expected value that will be returned after "lower_mptr <= lower_mptr"
      // operator will be called
      bool first_first_mptr_expected_val = true;
      // Expected value that will be returned after "lower_mptr <= greater_mptr"
      // operator will be called
      bool first_second_mptr_expected_val = true;
      // Expected value that will be returned after "greater_mptr <= lower_mptr"
      // operator will be called
      bool second_first_mptr_expected_val = false;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "lower_mptr <= lower_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "lower_mptr <= greater_mptr" operator will be called
      //   - At the third position will contains value that will be returned
      //     after "greater_mptr <= lower_mptr" operator will be called
      std::array<bool, 3> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t mptr_1(low_value_acc);
        multi_ptr_t mptr_2(great_value_acc);

        result_acc[0] = mptr_1 <= mptr_1;
        result_acc[1] = mptr_1 <= mptr_2;
        result_acc[2] = mptr_2 <= mptr_1;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == first_first_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == first_second_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[2] == second_first_mptr_expected_val);
    }
    SECTION(
        section_name(
            "Check multi_ptr operator>=(const multi_ptr&, const multi_ptr&)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Expected value that will be returned after "lower_mptr >= lower_mptr"
      // operator will be called
      bool first_first_mptr_expected_val = true;
      // Expected value that will be returned after "greater_mptr >= lower_mptr"
      // operator will be called
      bool second_first_mptr_expected_val = true;
      // Expected value that will be returned after "lower_mptr >= greater_mptr"
      // operator will be called
      bool first_second_mptr_expected_val = false;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "lower_mptr >= lower_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "greater_mptr >= lower_mptr" operator will be called
      //   - At the third position will contains value that will be returned
      //     after "lower_mptr >= greater_mptr" operator will be called
      std::array<bool, 3> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t mptr_1(low_value_acc);
        multi_ptr_t mptr_2(great_value_acc);

        result_acc[0] = mptr_1 >= mptr_1;
        result_acc[1] = mptr_2 >= mptr_1;
        result_acc[2] = mptr_1 >= mptr_2;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == first_first_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == second_first_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[2] == first_second_mptr_expected_val);
    }

    SECTION(
        section_name(
            "Check multi_ptr operator==(const multi_ptr& lhs, std::nullptr_t)")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      // Expected value that will be returned after "nullptr_mptr == nullptr"
      // operator will be called
      bool nullptr_nullptr_mptr_expected_val = true;
      // Expected value that will be returned after "nullptr == nullptr_mptr"
      // operator will be called
      bool mptr_nullptr_nullptr_expected_val = true;
      // Expected value that will be returned after "nullptr == value_mptr"
      // operator will be called
      bool nullptr_value_expected_val = false;
      // Expected value that will be returned after "value_mptr == nullptr"
      // operator will be called
      bool value_nullptr_expected_val = false;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "nullptr_mptr == nullptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "value_mptr == nullptr" operator will be called
      std::array<bool, 4> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(low_value_acc);

        result_acc[0] = nullptr_mptr == nullptr;
        result_acc[1] = nullptr == nullptr_mptr;
        result_acc[2] = value_mptr == nullptr;
        result_acc[3] = nullptr == value_mptr;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == nullptr_nullptr_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == mptr_nullptr_nullptr_expected_val);
      CHECK(mpr_comparison_ops_results[2] == nullptr_value_expected_val);
      CHECK(mpr_comparison_ops_results[3] == value_nullptr_expected_val);
    }
    SECTION(section_name(
                "Check multi_ptr operator!=(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Expected value that will be returned after "nullptr_mptr != nullptr"
      // operator will be called
      bool nullptr_nullptr_mptr_expected_val = false;
      // Expected value that will be returned after "nullptr != nullptr_mptr"
      // operator will be called
      bool mptr_nullptr_nullptr_expected_val = false;
      // Expected value that will be returned after "nullptr != value_mptr"
      // operator will be called
      bool nullptr_value_expected_val = true;
      // Expected value that will be returned after "value_mptr != nullptr"
      // operator will be called
      bool value_nullptr_expected_val = false;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "nullptr_mptr != nullptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "nullptr != value_mptr" operator will be called
      std::array<bool, 4> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(low_value_acc);

        result_acc[0] = nullptr != nullptr_mptr;
        result_acc[1] = nullptr_mptr != nullptr;
        result_acc[3] = nullptr != value_mptr;
        result_acc[2] = value_mptr != nullptr;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == nullptr_nullptr_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == mptr_nullptr_nullptr_expected_val);
      CHECK(mpr_comparison_ops_results[2] == nullptr_value_expected_val);
      CHECK(mpr_comparison_ops_results[3] == value_nullptr_expected_val);
    }

    SECTION(section_name(
                "Check multi_ptr operator<(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Expected value that will be returned after "nullptr < nullptr_mptr"
      // operator will be called
      bool nullptr_nullptr_mptr_expected_val = false;
      // Expected value that will be returned after "nullptr_mptr < nullptr"
      // operator will be called
      bool nullptr_mptr_nullptr_expected_val = true;
      // Expected value that will be returned after "nullptr < value_mptr"
      // operator will be called
      bool nullptr_value_expected_val = false;
      // Expected value that will be returned after "value_mptr < nullptr"
      // operator will be called
      bool value_nullptr_expected_val = false;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "nullptr < nullptr_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "nullptr < value_mptr" operator will be called
      //   - At the third position will contains value that will be returned
      //     after "value_mptr < nullptr" operator will be called
      std::array<bool, 4> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(low_value_acc);

        result_acc[0] = nullptr < nullptr_mptr;
        result_acc[0] = nullptr_mptr < nullptr;
        result_acc[1] = nullptr < value_mptr;
        result_acc[2] = value_mptr < nullptr;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == nullptr_nullptr_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == nullptr_mptr_nullptr_expected_val);
      CHECK(mpr_comparison_ops_results[2] == nullptr_value_expected_val);
      CHECK(mpr_comparison_ops_results[3] == value_nullptr_expected_val);
    }
    SECTION(section_name(
                "Check multi_ptr operator>(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Expected value that will be returned after "nullptr < nullptr_mptr"
      // operator will be called
      bool nullptr_nullptr_mptr_expected_val = false;
      // Expected value that will be returned after "nullptr_mptr < nullptr"
      // operator will be called
      bool nullptr_mptr_nullptr_expected_val = false;
      // Expected value that will be returned after "nullptr < value_mptr"
      // operator will be called
      bool nullptr_value_expected_val = true;
      // Expected value that will be returned after "value_mptr < nullptr"
      // operator will be called
      bool value_nullptr_expected_val = false;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "nullptr > nullptr_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "nullptr > value_mptr" operator will be called
      //   - At the third position will contains value that will be returned
      //     after "value_mptr > nullptr" operator will be called
      std::array<bool, 4> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(low_value_acc);

        result_acc[0] = nullptr > nullptr_mptr;
        result_acc[0] = nullptr_mptr > nullptr;
        result_acc[1] = nullptr > value_mptr;
        result_acc[2] = value_mptr > nullptr;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == nullptr_nullptr_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == nullptr_mptr_nullptr_expected_val);
      CHECK(mpr_comparison_ops_results[2] == nullptr_value_expected_val);
      CHECK(mpr_comparison_ops_results[3] == value_nullptr_expected_val);
    }
    SECTION(section_name(
                "Check multi_ptr operator<=(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Expected value that will be returned after "nullptr <= nullptr_mptr"
      // operator will be called
      bool nullptr_nullptr_mptr_expected_val = true;
      // Expected value that will be returned after "nullptr_mptr <= nullptr"
      // operator will be called
      bool nullptr_mptr_nullptr_expected_val = true;
      // Expected value that will be returned after "nullptr <= value_mptr"
      // operator will be called
      bool nullptr_value_expected_val = false;
      // Expected value that will be returned after "value_mptr <= nullptr"
      // operator will be called
      bool value_nullptr_expected_val = false;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "nullptr <= nullptr_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "nullptr <= value_mptr" operator will be called
      //   - At the third position will contains value that will be returned
      //     after "value_mptr <= nullptr" operator will be called
      std::array<bool, 4> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(low_value_acc);

        result_acc[0] = nullptr <= nullptr_mptr;
        result_acc[1] = nullptr_mptr <= nullptr;
        result_acc[2] = nullptr <= value_mptr;
        result_acc[3] = value_mptr <= nullptr;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == nullptr_nullptr_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == nullptr_mptr_nullptr_expected_val);
      CHECK(mpr_comparison_ops_results[2] == nullptr_value_expected_val);
      CHECK(mpr_comparison_ops_results[3] == value_nullptr_expected_val);
    }

    SECTION(section_name(
                "Check multi_ptr operator>=(std::nullptr_t, const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      // Expected value that will be returned after "nullptr >= nullptr_mptr"
      // operator will be called
      bool nullptr_nullptr_mptr_expected_val = true;
      // Expected value that will be returned after "nullptr_mptr >= nullptr"
      // operator will be called
      bool nullptr_mptr_nullptr_expected_val = true;
      // Expected value that will be returned after "nullptr >= value_mptr"
      // operator will be called
      bool nullptr_value_expected_val = false;
      // Expected value that will be returned after "value_mptr >= nullptr"
      // operator will be called
      bool value_nullptr_expected_val = true;

      // Array that contained operator calling result:
      //   - At the first position will contains value that will be returned
      //     after "nullptr >= nullptr_mptr" operator will be called
      //   - At the second position will contains value that will be returned
      //     after "nullptr_mptr >= nullptr" operator will be called
      //   - At the third position will contains value that will be returned
      //     after "nullptr >= value_mptr" operator will be called
      //   - At the fourth position will contains value that will be returned
      //     after "value_mptr >= nullptr" operator will be called
      std::array<bool, 4> mpr_comparison_ops_results;
      const auto run_test_action = [](auto low_value_acc, auto great_value_acc,
                                      auto result_acc) {
        multi_ptr_t nullptr_mptr(nullptr);
        multi_ptr_t value_mptr(low_value_acc);

        result_acc[0] = nullptr >= nullptr_mptr;
        result_acc[1] = nullptr_mptr >= nullptr;
        result_acc[2] = nullptr >= value_mptr;
        result_acc[3] = value_mptr >= nullptr;
      };

      run_test(queue, run_test_action, mpr_comparison_ops_results);

      CHECK(mpr_comparison_ops_results[0] == nullptr_nullptr_mptr_expected_val);
      CHECK(mpr_comparison_ops_results[0] == nullptr_mptr_nullptr_expected_val);
      CHECK(mpr_comparison_ops_results[1] == nullptr_value_expected_val);
      CHECK(mpr_comparison_ops_results[2] == value_nullptr_expected_val);
    }
  }
};

template <typename T>
class check_multi_ptr_comparison_op_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto address_spaces = multi_ptr_common::get_address_spaces();
    const auto is_decorated = multi_ptr_common::get_decorated();
    // Run test
    for_all_combinations<run_multi_ptr_comparison_op_test, T>(
        address_spaces, is_decorated, type_name);
  }
};

}  // namespace multi_ptr_comparison_op

#endif  // __SYCLCTS_TESTS_MULTI_PTR_COMPARISON_OP_H
