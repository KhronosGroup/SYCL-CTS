
/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides code for multi_ptr access members
//
*******************************************************************************/
#ifndef __SYCLCTS_TESTS_MULTI_PTR_ACCESS_MEMBERS_OPS_H
#define __SYCLCTS_TESTS_MULTI_PTR_ACCESS_MEMBERS_OPS_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_access_members {

/**
 * @brief Provides functions for verification multi_ptr access operators that
 *        they returns correct data
 * @tparam T Current data type
 * @tparam AddrSpaceT sycl::access::address_space enumeration's field
 * @tparam IsDecoratedT sycl::access::decorated enumeration's field
 */
template <typename T, typename AddrSpaceT, typename IsDecoratedT>
class run_access_members_tests {
  static constexpr sycl::access::address_space space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;
  using multi_ptr_t = const sycl::multi_ptr<T, space, decorated>;

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
    constexpr int val_to_init = 42;
    T value = user_def_types::get_init_value_helper<T>(val_to_init);
    // Variables that will be used to check that access members returns correct
    // data type
    bool dereference_return_type_is_correct = false;
    bool dereference_op_return_type_is_correct = false;
    bool get_return_type_is_correct = false;
    bool get_raw_return_type_is_correct = false;
    bool get_decorated_return_type_is_correct = false;
    // Variables that will be used to check that access members returns correct
    // value
    T dereference_ret_value;
    T dereference_op_ret_value;
    T get_member_ret_value;
    T get_raw_member_ret_value;

    sycl::range r = sycl::range(1);
    {
      sycl::buffer<T> dereference_ret_value_buffer(&dereference_ret_value,
                                                   sycl::range(1));
      sycl::buffer<T> dereference_op_ret_value_buffer(&dereference_op_ret_value,
                                                      sycl::range(1));
      sycl::buffer<T> get_member_ret_value_buffer(&get_member_ret_value,
                                                  sycl::range(1));
      sycl::buffer<T> get_raw_member_ret_value_buffer(&get_raw_member_ret_value,
                                                      sycl::range(1));

      sycl::buffer<T> val_buffer(&value, sycl::range(1));
      sycl::buffer<bool> dereference_return_type_buf(
          &dereference_return_type_is_correct, sycl::range(1));
      sycl::buffer<bool> dereference_op_return_type_buf(
          &dereference_op_return_type_is_correct, sycl::range(1));
      sycl::buffer<bool> get_return_type_buf(&get_return_type_is_correct,
                                             sycl::range(1));
      sycl::buffer<bool> get_raw_return_type_buf(
          &get_raw_return_type_is_correct, sycl::range(1));
      sycl::buffer<bool> get_decorated_return_type_buf(
          &get_decorated_return_type_is_correct, sycl::range(1));

      queue.submit([&](sycl::handler &cgh) {
        auto dereference_ret_value_acc =
            dereference_ret_value_buffer
                .template get_access<sycl::access_mode::write>(cgh);
        auto dereference_op_ret_value_acc =
            dereference_op_ret_value_buffer
                .template get_access<sycl::access_mode::write>(cgh);
        auto get_member_ret_value_acc =
            get_member_ret_value_buffer
                .template get_access<sycl::access_mode::write>(cgh);
        auto get_raw_member_ret_value_acc =
            get_raw_member_ret_value_buffer
                .template get_access<sycl::access_mode::write>(cgh);

        auto dereference_return_type_acc =
            dereference_return_type_buf
                .template get_access<sycl::access_mode::write>(cgh);
        auto dereference_op_return_type_acc =
            dereference_op_return_type_buf
                .template get_access<sycl::access_mode::write>(cgh);
        auto get_return_type_acc =
            get_return_type_buf.template get_access<sycl::access_mode::write>(
                cgh);
        auto get_raw_return_type_acc =
            get_raw_return_type_buf
                .template get_access<sycl::access_mode::write>(cgh);
        auto get_decorated_return_type_acc =
            get_decorated_return_type_buf
                .template get_access<sycl::access_mode::write>(cgh);

        auto acc_for_multi_ptr = val_buffer.template get_access<
            user_def_types::get_init_value_helper::access_mode::read>(cgh);

        auto test_device_code = [=] {
          const multi_ptr_t multi_ptr(acc_for_multi_ptr);

          // Dereference and multi_ptr::operator->() available only when:
          // !std::is_void<sycl::multi_ptr::value_type>::value
          if constexpr (!std::is_void_v<multi_ptr_t::value_type>) {
            // Check dereference operator return value and type correctness
            dereference_return_type_acc[0] =
                std::is_same_v<decltype(*multi_ptr), multi_ptr_t::reference>;
            dereference_ret_value_acc[0] = *multi_ptr;
            // Check operator->() return value and type correctness
            dereference_op_return_type_acc[0] =
                std::is_same_v<decltype(multi_ptr.operator->()),
                               multi_ptr_t::pointer>;
            dereference_op_ret_value_acc[0] = *(multi_ptr.operator->());
          }

          // Check get() return value and type correctness
          get_return_type_acc[0] =
              std::is_same_v<decltype(multi_ptr.get()), multi_ptr_t::pointer>;
          // Skip verification if pointer is decorated
          if constexpr (decorated == sycl::access::decorated::yes) {
            get_member_ret_value_acc[0] = *(multi_ptr.get());
          }
          // Check get_raw() return value and type correctness
          get_raw_return_type_acc[0] =
              std::is_same_v<decltype(multi_ptr.get_raw()),
                             std::add_pointer_t<multi_ptr_t::value_type>>;
          get_raw_member_ret_value_acc[0] = *(multi_ptr.get_raw());
          // Check get_decorated() return type correctness
          get_decorated_return_type_acc[0] =
              std::is_pointer_v<decltype(multi_ptr.get_decorated())>;
        };

        if constexpr (space == sycl::access::address_space::global_space) {
          cgh.single_task([=] { test_device_code(); });
        } else {
          cgh.parallel_for(sycl::nd_range(r, r), [=](sycl::nd_item item) {
            if constexpr (space == sycl::access::address_space::local_space) {
              test_device_code();
            } else {
              test_device_code();
            }
          });
        }
      });
    }
    T expected_value = user_def_types::get_init_value_helper<T>(val_to_init);
    // Dereference and multi_ptr::operator->() available only when:
    // !std::is_void<sycl::multi_ptr::value_type>::value
    if constexpr (!std::is_void_v<multi_ptr_t::value_type>) {
      SECTION(section_name("Check dereference return value and type")
                  .with("T", type_name)
                  .with("address_space", address_space_name)
                  .with("decorated", is_decorated_name)
                  .create()) {
        CHECK(dereference_return_type_is_correct);
        CHECK(dereference_ret_value == expected_value);
      }
      SECTION(
          section_name(
              "Check dereference operator (operator->()) return value and type")
              .with("T", type_name)
              .with("address_space", address_space_name)
              .with("decorated", is_decorated_name)
              .create()) {
        CHECK(dereference_op_return_type_is_correct);
        CHECK(dereference_op_ret_value == expected_value);
      }
    }
    SECTION(section_name("Check get() return value and type")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(get_return_type_is_correct);
      // Skip verification if pointer is decorated
      if constexpr (decorated == sycl::access::decorated::yes) {
        CHECK(get_member_ret_value == expected_value);
      }
    }
    SECTION(section_name("Check get_raw() return value and type")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(get_raw_return_type_is_correct);
      CHECK(get_raw_member_ret_value == expected_value);
    }
    SECTION(section_name("Check that get_decorated() returns pointer")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(get_decorated_return_type_is_correct);
    }
  }
};

template <typename T>
class check_multi_ptr_access_members_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto address_spaces = multi_ptr_common::get_address_spaces();
    const auto is_decorated = multi_ptr_common::get_decorated();

    for_all_combinations<run_access_members_tests, T>(address_spaces,
                                                      is_decorated, type_name);
  }
};

}  // namespace multi_ptr_access_members

#endif  // __SYCLCTS_TESTS_MULTI_PTR_ACCESS_MEMBERS_OPS_H
