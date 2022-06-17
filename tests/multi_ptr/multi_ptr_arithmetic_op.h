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
    // Expected value that will be returned after multi_ptr's operator will be
    // called
    T op_return_result_expected_val;
    // Expected value that will be stored into multi_ptr after
    // multi_ptr's operator will be called
    T op_calling_expected_val;

    auto queue = sycl_cts::util::get_cts_object::queue();
    SECTION(section_name("Check multi_ptr operator++(multi_ptr& mp)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[1];
      op_calling_expected_val = arr[1];
      // Variable that will be used to store the value that will be returned
      // after calling multi_ptr's operator
      T op_calling_result_val;
      // Variable that will be used to store the multi_ptr's value after
      // multi_ptr's operator will be called
      T op_calling_val;
      {
        sycl::buffer<T> buffer_for_mptr(arr, sycl::range(array_size));
        sycl::buffer<T> op_return_res_val_buffer(&op_calling_result_val,
                                                 sycl::range(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);
          // Accessor that will be used to store the value that will be returned
          // after calling multi_ptr's operator
          auto op_return_res_val_acc =
              op_return_res_val_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          // Accessor that will be used to store the multi_ptr's value after
          // multi_ptr's operator will be called
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            multi_ptr_t result_mptr = ++mptr;

            op_return_res_val_acc[0] = *result_mptr;
            op_res_val_acc[0] = *mptr;
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(section_name("Check multi_ptr operator++(multi_ptr&, int)")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[0];
      op_calling_expected_val = arr[1];
     // Variable that will be used to store the value that will be returned
      // after calling multi_ptr's operator
      T op_calling_result_val;
      // Variable that will be used to store the multi_ptr's value after
      // multi_ptr's operator will be called
      T op_calling_val;
      {
        sycl::buffer<T> buffer_for_mptr(arr, sycl::range(array_size));
        sycl::buffer<T> op_return_res_val_buffer(&op_calling_result_val,
                                                 sycl::range(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);
          // Accessor that will be used to store the value that will be returned
          // after calling multi_ptr's operator
          auto op_return_res_val_acc =
              op_return_res_val_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          // Accessor that will be used to store the multi_ptr's value after
          // multi_ptr's operator will be called
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            multi_ptr_t result_mptr = mptr++;

            op_return_res_val_acc[0] = *result_mptr;
            op_res_val_acc[0] = *mptr;
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(section_name("Check multi_ptr operator--(multi_ptr&)")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[middle_elem_index - 1];
      op_calling_expected_val = arr[middle_elem_index - 1];
      // Variable that will be used to store the value that will be returned
      // after calling multi_ptr's operator
      T op_calling_result_val;
      // Variable that will be used to store the multi_ptr's value after
      // multi_ptr's operator will be called
      T op_calling_val;
      {
        sycl::buffer<T> buffer_for_mptr(arr, sycl::range(array_size));
        sycl::buffer<T> op_return_res_val_buffer(&op_calling_result_val,
                                                 sycl::range(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);
          // Accessor that will be used to store the value that will be returned
          // after calling multi_ptr's operator
          auto op_return_res_val_acc =
              op_return_res_val_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          // Accessor that will be used to store the multi_ptr's value after
          // multi_ptr's operator will be called
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            // Shift multi_ptr that he is pointed to the middle element, to have
            // possibe decrease pointed value index
            mptr += middle_elem_index;
            multi_ptr_t result_mptr = --mptr;

            op_return_res_val_acc[0] = *result_mptr;
            op_res_val_acc[0] = *mptr;
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(section_name("Check multi_ptr operator--(multi_ptr&, int)")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[middle_elem_index];
      op_calling_expected_val = arr[middle_elem_index - 1];
      // Variable that will be used to store the value that will be returned
      // after calling multi_ptr's operator
      T op_calling_result_val;
      // Variable that will be used to store the multi_ptr's value after
      // multi_ptr's operator will be called
      T op_calling_val;
      {
        sycl::buffer<T> buffer_for_mptr(arr, sycl::range(array_size));
        sycl::buffer<T> op_return_res_val_buffer(&op_calling_result_val,
                                                 sycl::range(1));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);
          // Accessor that will be used to store the value that will be returned
          // after calling multi_ptr's operator
          auto op_return_res_val_acc =
              op_return_res_val_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          // Accessor that will be used to store the multi_ptr's value after
          // multi_ptr's operator will be called
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            // Shift multi_ptr that he is pointed to the middle element, to have
            // possibe decrease pointed value index
            mptr += middle_elem_index;
            multi_ptr_t result_mptr = mptr--;

            op_return_res_val_acc[0] = *result_mptr;
            op_res_val_acc[0] = *mptr;
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
      CHECK(op_calling_val == op_calling_expected_val);
    }

    using diff_t = multi_ptr_t::difference_type;
    diff_t shift = array_size / 3;
    SECTION(section_name("Check multi_ptr operator+=(multi_ptr&, diff_type)")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_calling_expected_val = arr[shift];
      // Variable that will be used to store the multi_ptr's value after
      // multi_ptr's operator will be called
      T op_calling_val;
      {
        sycl::buffer<T> buffer_for_mptr(arr, sycl::range(array_size));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);
          // Accessor that will be used to store the multi_ptr's value after
          // multi_ptr's operator will be called
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            mptr += shift;

            op_res_val_acc[0] = *mptr;
          });
        });
      }
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(section_name("Check multi_ptr operator-=(multi_ptr&, diff_type)")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_calling_expected_val = arr[middle_elem_index - shift];
      // Variable that will be used to store the multi_ptr's value after
      // multi_ptr's operator will be called
      T op_calling_val;
      {
        sycl::buffer<T> buffer_for_mptr(arr, sycl::range(array_size));
        sycl::buffer<T> op_res_val_buffer(&op_calling_val, sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);
          // Accessor that will be used to store the multi_ptr's value after
          // multi_ptr's operator will be called
          auto op_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr(acc_for_mptr);
            // Shift multi_ptr that he is pointed to the middle element, to have
            // possibe decrease pointed value index
            mptr += middle_elem_index;
            mptr -= shift;

            op_res_val_acc[0] = *mptr;
          });
        });
      }
      CHECK(op_calling_val == op_calling_expected_val);
    }
    SECTION(
        section_name("Check multi_ptr operator+(const multi_ptr&, diff_type)")
            .with("T", type_name)
            .with("address_space", "access::address_space::global_space")
            .with("decorated", is_decorated_name)
            .create()) {
      op_return_result_expected_val = arr[shift];
      // Variable that will be used to store the value that will be returned
      // after calling multi_ptr's operator
      T op_calling_result_val;
      {
        sycl::buffer<T> buffer_for_mptr(arr, sycl::range(array_size));
        sycl::buffer<T> op_res_val_buffer(&op_calling_result_val,
                                          sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);
          // Accessor that will be used to store the value that will be returned
          // after calling multi_ptr's operator
          auto op_return_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            const multi_ptr_t mptr(acc_for_mptr);
            multi_ptr_t result_mptr = mptr + shift;

            op_return_res_val_acc[0] = *result_mptr;
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
    }
    SECTION(
        section_name("Check multi_ptr operator-(const multi_ptr&, diff_type)")
            .with("T", type_name)
            .with("address_space", "access::address_space::global_space")
            .with("decorated", is_decorated_name)
            .create()) {
      op_return_result_expected_val = arr[middle_elem_index - shift];
      // Variable that will be used to store the value that will be returned
      // after calling multi_ptr's operator
      T op_calling_result_val;
      {
        sycl::buffer<T> buffer_for_mptr(arr, sycl::range(array_size));
        sycl::buffer<T> op_res_val_buffer(&op_calling_result_val,
                                          sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);
          // Accessor that will be used to store the value that will be returned
          // after calling multi_ptr's operator
          auto op_return_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            const multi_ptr_t mptr(acc_for_mptr);
            // Shift multi_ptr that he is pointed to the middle element, to have
            // possibe decrease pointed value index
            mptr += middle_elem_index;
            multi_ptr_t result_mptr = mptr - shift;

            op_return_res_val_acc[0] = *result_mptr;
          });
        });
      }
      CHECK(op_calling_result_val == op_return_result_expected_val);
    }
    SECTION(section_name("Check multi_ptr operator*(const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      op_return_result_expected_val = arr[0];
      // Variable that will be used to store the value that will be returned
      // after calling multi_ptr's operator
      T op_calling_result_val;
      {
        sycl::buffer<T> buffer_for_mptr(arr, sycl::range(array_size));
        sycl::buffer<T> op_res_val_buffer(&op_calling_result_val,
                                          sycl::range(1));
        queue.submit([&](sycl::handler &cgh) {
          auto acc_for_mptr =
              buffer_for_mptr.template get_access<sycl::access_mode::read>(cgh);
          // Accessor that will be used to store the value that will be returned
          // after calling multi_ptr's operator
          auto op_return_res_val_acc =
              op_res_val_buffer.template get_access<sycl::access_mode::write>(
                  cgh);
          cgh.single_task([=] {
            const multi_ptr_t mptr(acc_for_mptr);

            op_return_res_val_acc[0] = *mptr;
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
