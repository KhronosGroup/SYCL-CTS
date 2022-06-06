
/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides code for multi_ptr common assignment operators
//
*******************************************************************************/
#ifndef __SYCLCTS_TESTS_MULTI_PTR_COMMON_ASSIGN_OPS_H
#define __SYCLCTS_TESTS_MULTI_PTR_COMMON_ASSIGN_OPS_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_common_assignment_ops {

constexpr int expected_val = 42;
template <typename T, typename AddrSpaceT, typename IsDecorated>
class run_common_assign_tests {
  static constexpr sycl::access::address_space space = AddrSpaceT::value;
  static constexpr sycl::access::decorated decorated = IsDecorated::value;
  using multi_ptr_t = sycl::multi_ptr<T, space, decorated>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &address_space_name,
                  const std::string &is_decorated_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();
    T value = user_def_types::get_init_value_helper<T>(expected_val);
    SECTION(section_name("Check &operator=(const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      bool res = false;
      {
        sycl::buffer<bool, 1> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<T> val_buffer(&value, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          auto val_acc =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task([=] {
            const multi_ptr_t mptr_in(val_acc);
            multi_ptr_t mptr_out;

            mptr_out = mptr_in;

            // Check that second mptr has the same value as first mptr
            res_acc[0] = *(mptr_out.get_raw()) == val_acc[0];
          });
        });
      }
      CHECK(res);
    }

    SECTION(section_name("Check &operator=(multi_ptr&&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      bool res = false;
      {
        sycl::buffer<bool, 1> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<T> val_buffer(&value, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          auto val_acc =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr_in(val_acc);
            multi_ptr_t mptr_out;

            mptr_out = std::move(mptr_in);

            // Check that second mptr has the same value as first mptr
            res_acc[0] = *(mptr_out.get_raw()) == val_acc[0];
          });
        });
      }
      CHECK(res);
    }

    SECTION(section_name("Check &operator=(std::nullptr_t)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      bool res = false;
      {
        sycl::buffer<bool, 1> res_buf(&res, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          cgh.single_task([=] {
            multi_ptr_t mptr_out;
            mptr_out = nullptr;

            // Check that new mptr is nullptr
            res_acc[0] = mptr_out.get_raw() == nullptr;
          });
        });
      }
      CHECK(res);
    }
  }
};

template <typename T>
class check_multi_ptr_common_assign_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto address_spaces = multi_ptr_common::get_address_spaces();
    const auto is_decorated = multi_ptr_common::get_decorated();

    for_all_combinations<run_common_assign_tests, T>(address_spaces,
                                                     is_decorated, type_name);
  }
};

}  // namespace multi_ptr_common_assignment_ops

#endif  // __SYCLCTS_TESTS_MULTI_PTR_COMMON_ASSIGN_OPS_H
