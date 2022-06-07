
/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides code for multi_ptr prefetch member
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_MULTI_PTR_PREFETCH_FUNC_H
#define __SYCLCTS_TESTS_MULTI_PTR_PREFETCH_FUNC_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_prefetch_member {

constexpr int expected_val = 42;

/**
 * @brief Provides functions for verification multi_ptr prefetch member
 * @tparam T Current data type
 * @tparam IsDecoratedT sycl::access::decorated enumeration's field
 */
template <typename T, typename IsDecoratedT>
class run_prefetch_test {
  static constexpr sycl::access::decorated decorated = IsDecoratedT::value;
  using multi_ptr_t =
      sycl::multi_ptr<T, sycl::access::address_space::global_space, decorated>;

 public:
  /**
   * @param type_name Current data type string representation
   * @param is_decorated_name Current sycl::access::decorated string
   *        representation
   */
  void operator()(const std::string &type_name,
                  const std::string &is_decorated_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();
    T value = user_def_types::get_init_value_helper<T>(expected_val);
    SECTION(section_name("Check multi_ptr::prefetch")
                .with("T", type_name)
                .with("address_space", "access::address_space::global_space")
                .with("decorated", is_decorated_name)
                .create()) {
      bool res = false;
      {
        sycl::buffer<bool> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<T> val_buffer(&value, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task([=] {
            const multi_ptr_t mptr(acc_for_mptr);

            // Check call and const correctness for multi_ptr::prefetch, then
            // verify that multi_ptr contained expected value
            mptr_in.prefetch(0);
            res_acc[0] = mptr_in[0] == acc_for_mptr[0];
          });
        });
      }
      CHECK(res);
    }
  }
};

template <typename T>
class check_multi_ptr_prefetch_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto is_decorated = multi_ptr_common::get_decorated();
    // Run test
    for_all_combinations<run_prefetch_test, T>(is_decorated, type_name);
  }
};

}  // namespace multi_ptr_prefetch_member

#endif  // __SYCLCTS_TESTS_MULTI_PTR_PREFETCH_FUNC_H
