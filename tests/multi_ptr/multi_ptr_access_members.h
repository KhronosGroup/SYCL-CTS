
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
 * @brief Provides verification for multi_ptr access operators that they returns
 *        correct data types
 * @tparam T Current data type
 * @tparam AddrSpaceT sycl::access::address_space enumeration's field
 * @tparam IsDecorated sycl::access::decorated enumeration's field
 * @param type_name Current data type string representation
 * @param address_space_name Current sycl::access::address_space string
 *        representation
 * @param is_decorated_name Current sycl::access::decorated string
 *        representation
 */
template <typename T, typename AddrSpaceT, typename IsDecorated>
class run_access_members_tests {
  static constexpr sycl::access::address_space space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecorated::value;
  using multi_ptr_t = const sycl::multi_ptr<T, space, decorated>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &address_space_name,
                  const std::string &is_decorated_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();
    constexpr int val_to_init = 42;
    T value = user_def_types::get_init_value_helper<T>(val_to_init);
    bool dereference_result = false;
    bool dereference_op_result = false;
    bool get_result = false;
    bool get_raw_result = false;
    bool get_decorated_result = false;
    {
      sycl::buffer<T> val_buffer(&value, sycl::range<1>(1));
      sycl::buffer<bool, 1> dereference_result_buf(&dereference_result,
                                                   sycl::range<1>(1));
      sycl::buffer<bool, 1> dereference_op_result_buf(&dereference_op_result,
                                                      sycl::range<1>(1));
      sycl::buffer<bool, 1> get_result_buf(&dereference_op_result,
                                           sycl::range<1>(1));
      sycl::buffer<bool, 1> get_raw_result_buf(&dereference_op_result,
                                               sycl::range<1>(1));
      sycl::buffer<bool, 1> get_decorated_result_buf(&dereference_op_result,
                                                     sycl::range<1>(1));

      queue.submit([&](sycl::handler &cgh) {
        auto dereference_result_acc =
            dereference_result_buf
                .template get_access<sycl::access_mode::write>(cgh);
        auto dereference_op_result_acc =
            dereference_op_result_buf
                .template get_access<sycl::access_mode::write>(cgh);
        auto get_result_acc =
            get_result_buf.template get_access<sycl::access_mode::write>(cgh);
        auto get_raw_result_acc =
            get_raw_result_buf.template get_access<sycl::access_mode::write>(
                cgh);
        auto get_decorated_result_acc =
            get_decorated_result_buf
                .template get_access<sycl::access_mode::write>(cgh);

        auto acc_for_multi_ptr =
            val_buffer.template get_access<sycl::access_mode::read>(cgh);
        cgh.single_task([=] {
          const multi_ptr_t multi_ptr(acc_for_multi_ptr);

          // Check dereference operator return type correctness
          dereference_result_acc[0] =
              std::is_same_v<decltype(*multi_ptr), multi_ptr::reference>;
          // Check operator->() return type correctness
          dereference_op_result_acc[0] =
              std::is_same_v<decltype(multi_ptr.operator->()),
                             multi_ptr::pointer>;
          // Check get() return type correctness
          get_result_acc[0] =
              std::is_same_v<decltype(multi_ptr.get()), multi_ptr::pointer>;
          // Check get_raw() return type correctness
          get_raw_result_acc[0] =
              std::is_same_v<decltype(multi_ptr.get_raw()),
                             std::add_pointer_t<multi_ptr::value_type>>;
          // Check get_decorated() return type correctness
          get_decorated_result_acc[0] =
              std::is_pointer_v<decltype(multi_ptr.get_decorated())>;
        });
      });
    }
    SECTION(section_name("Check dereference return type")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(dereference_result);
    }
    SECTION(
        section_name("Check dereference operator (operator->()) return type")
            .with("T", type_name)
            .with("address_space", address_space_name)
            .with("decorated", is_decorated_name)
            .create()) {
      CHECK(dereference_op_result);
    }
    SECTION(section_name("Check get() return type")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(get_result);
    }
    SECTION(section_name("Check get_raw() return type")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(get_raw_result);
    }
    SECTION(section_name("Check that get_decorated() returns pointer")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(get_decorated_result);
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
