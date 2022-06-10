
/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides code for multi_ptr arithmetic operators
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_MULTI_PTR_ARITHMETIC_OP_H
#define __SYCLCTS_TESTS_MULTI_PTR_ARITHMETIC_OP_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_prefetch_member {

constexpr int expected_val = 42;

/**
 * @brief Provides functor for verification on multi_ptr arithmetic operators
 * @tparam T Current data type
 * @tparam AddrSpaceT sycl::access::address_space enumeration's field
 * @tparam IsDecoratedT sycl::access::decorated enumeration's field
 */
template <typename T, typename AddrSpaceT, typename IsDecoratedT>
class run_multi_ptr_arithmetic_op_test {
  static constexpr sycl::access::address_space space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;
  using multi_ptr_t = sycl::multi_ptr<T, space, decorated>;

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
    constexpr size_t array_size = 10;
    // Array that will be used in multi_ptr
    T arr[array_size];
    for (size_t i = 0; i < array_size; ++i) {
      arr[i] = i;
    }
    size_t middle_elem_index = array_size / 2;
    T op_return_result_expected_val;
    T op_calling_expected_val;

    auto queue = sycl_cts::util::get_cts_object::queue();
    T value = user_def_types::get_init_value_helper<T>(expected_val);
    SECTION(section_name("Check multi_ptr preincrement operator")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[middle_elem_index + 1];
      op_calling_expected_val = arr[middle_elem_index + 1];
      T op_calling_result_val;
      T op_calling_val;
      {
        sycl::buffer<T> val_buffer(&arr[middle_elem_index], sycl::range<1>(1));
        sycl::buffer<T> op_return_res_val_buffer(&op_calling_result_val,
                                                 sycl::range<1>(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto op_return_res_val_acc =
              op_return_res_val_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            multi_ptr_t result_mptr = ++mptr;

            op_return_res_val_acc[0] = result_mptr[0];
            op_res_val_acc[0] = mptr[0];
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(section_name("Check multi_ptr postincrement operator")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[middle_elem_index];
      op_calling_expected_val = arr[middle_elem_index + 1];
      T op_calling_result_val;
      T op_calling_val;
      {
        sycl::buffer<T> val_buffer(&arr[middle_elem_index], sycl::range<1>(1));
        sycl::buffer<T> op_return_res_val_buffer(&op_calling_result_val,
                                                 sycl::range<1>(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto op_return_res_val_acc =
              op_return_res_val_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            multi_ptr_t result_mptr = mptr++;

            op_return_res_val_acc[0] = result_mptr[0];
            op_res_val_acc[0] = mptr[0];
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(section_name("Check multi_ptr predecrement operator")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[middle_elem_index - 1];
      op_calling_expected_val = arr[middle_elem_index - 1];
      T op_calling_result_val;
      T op_calling_val;
      {
        sycl::buffer<T> val_buffer(&arr[middle_elem_index], sycl::range<1>(1));
        sycl::buffer<T> op_return_res_val_buffer(&op_calling_result_val,
                                                 sycl::range<1>(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto op_return_res_val_acc =
              op_return_res_val_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            multi_ptr_t result_mptr = --mptr;

            op_return_res_val_acc[0] = result_mptr[0];
            op_res_val_acc[0] = mptr[0];
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(section_name("Check multi_ptr postdecrement operator")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[middle_elem_index];
      op_calling_expected_val = arr[middle_elem_index - 1];
      T op_calling_result_val;
      T op_calling_val;
      {
        sycl::buffer<T> val_buffer(&arr[middle_elem_index], sycl::range<1>(1));
        sycl::buffer<T> op_return_res_val_buffer(&op_calling_result_val,
                                                 sycl::range<1>(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto op_return_res_val_acc =
              op_return_res_val_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            multi_ptr_t result_mptr = mptr--;

            op_return_res_val_acc[0] = result_mptr[0];
            op_res_val_acc[0] = mptr[0];
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
      CHECK(op_calling_val == op_calling_expected_val);
    }

    using diff_t = multi_ptr_t::difference_type;
    diff_t shift = array_size / 3;

    SECTION(section_name("Check multi_ptr operator+=")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_calling_expected_val = arr[middle_elem_index + shift];
      T op_calling_val;
      {
        sycl::buffer<T> val_buffer(&arr[middle_elem_index], sycl::range<1>(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            mptr += shift;

            op_res_val_acc[0] = mptr[0];
          });
        });
      }
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(section_name("Check multi_ptr operator-=")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_calling_expected_val = arr[middle_elem_index - shift];
      T op_calling_val;
      {
        sycl::buffer<T> val_buffer(&arr[middle_elem_index], sycl::range<1>(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            mptr -= shift;

            op_res_val_acc[0] = mptr[0];
          });
        });
      }
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(section_name("Check multi_ptr operator+")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[middle_elem_index + shift];
      T op_calling_result_val;
      {
        sycl::buffer<T> val_buffer(&arr[middle_elem_index], sycl::range<1>(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_result_val,
                                          sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            const multi_ptr_t mptr(acc_for_mptr);
            multi_ptr_t result_mptr = mptr + shift;

            op_res_val_acc[0] = *result_mptr;
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
    }
    SECTION(section_name("Check multi_ptr operator-")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[middle_elem_index - shift];
      T op_calling_result_val;
      {
        sycl::buffer<T> val_buffer(&arr[middle_elem_index], sycl::range<1>(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_result_val,
                                          sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            const multi_ptr_t mptr(acc_for_mptr);
            multi_ptr_t result_mptr = mptr - shift;

            op_res_val_acc[0] = *result_mptr;
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
    }
    SECTION(section_name("Check multi_ptr dereference operator")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[middle_elem_index];
      T op_calling_result_val;
      {
        sycl::buffer<T> val_buffer(&arr[middle_elem_index], sycl::range<1>(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_result_val,
                                          sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            const multi_ptr_t mptr(acc_for_mptr);

            op_res_val_acc[0] = *mptr;
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
    }
  }
};

template <typename T>
class check_multi_ptr_arithmetic_op_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto address_spaces = multi_ptr_common::get_address_spaces();
    const auto is_decorated = multi_ptr_common::get_decorated();
    // Run test
    for_all_combinations<run_multi_ptr_arithmetic_op_test, T>(
        address_spaces, is_decorated, type_name);
  }
};

}  // namespace multi_ptr_prefetch_member

#endif  // __SYCLCTS_TESTS_MULTI_PTR_ARITHMETIC_OP_H
