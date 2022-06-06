
/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides code for multi_ptr convert assignment operators
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_MULTI_PTR_CONVERT_ASSIGN_OPS_H
#define __SYCLCTS_TESTS_MULTI_PTR_CONVERT_ASSIGN_OPS_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_convert_assignment_ops {

constexpr int expected_val = 42;

template <typename T, typename SrcAddrSpaceT, typename SrcIsDecorated,
          typename DstAddrSpaceT, typename DstIsDecorated>
class run_convert_assignment_operators_tests {
  static constexpr sycl::access::address_space src_space = SrcAddrSpaceT::value;
  static constexpr sycl::access::decorated src_decorated =
      SrcIsDecorated::value;
  static constexpr sycl::access::address_space dst_space = DstAddrSpaceT::value;
  static constexpr sycl::access::decorated dst_decorated =
      DstIsDecorated::value;
  using src_multi_ptr_t = sycl::multi_ptr<T, src_space, src_decorated>;
  using dst_multi_ptr_t = sycl::multi_ptr<T, dst_space, dst_decorated>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &src_address_space_name,
                  const std::string &src_is_decorated_name,
                  const std::string &dst_address_space_name,
                  const std::string &dst_is_decorated_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();
    T value = user_def_types::get_init_value_helper<T>(expected_val);
    SECTION(
        section_name(
            "Check &operator=(const multi_ptr<value_type, ASP, IsDecorated>&)")
            .with("T", type_name)
            .with("src address_space", src_address_space_name)
            .with("src decorated", src_is_decorated_name)
            .with("dst address_space", dst_address_space_name)
            .with("dst decorated", dst_is_decorated_name)
            .create()) {
      bool res = false;
      {
        sycl::buffer<bool, 1> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<T> val_buffer(&value, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task([=] {
            const src_multi_ptr_t mptr_in(acc_for_mptr);
            dst_multi_ptr_t mptr_out;

            mptr_out = mptr_in;

            // Check that second mptr has the same value as first mptr
            res_acc[0] = *(mptr_out.get_raw()) == acc_for_mptr[0];
          });
        });
      }
      CHECK(res);
    }

    SECTION(
        section_name(
            "Check &operator=(multi_ptr<value_type, ASP, IsDecorated>&)")
            .with("T", type_name)
            .with("src address_space", src_address_space_name)
            .with("src decorated", src_is_decorated_name)
            .with("dst address_space", dst_address_space_name)
            .with("dst decorated", dst_is_decorated_name)
            .create()) {
      bool res = false;
      {
        sycl::buffer<bool, 1> res_buf(&res, sycl::range<1>(1));
        sycl::buffer<T> val_buffer(&value, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          auto acc_for_mptr =
              val_buffer.template get_access<sycl::access_mode::read>(cgh);
          cgh.single_task([=] {
            src_multi_ptr_t mptr_in(acc_for_mptr);
            dst_multi_ptr_t mptr_out;

            mptr_out = mptr_in;

            // Check that second mptr has the same value as first mptr
            res_acc[0] = *(mptr_out.get_raw()) == acc_for_mptr[0];
          });
        });
      }
      CHECK(res);
    }
    }
};

template <typename T>
class check_multi_ptr_convert_assign_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto address_spaces = multi_ptr_common::get_address_spaces();
    const auto is_decorated = multi_ptr_common::get_decorated();

    // Run test with address_space and decorated for source multi_ptr type and
    // with address_space and decorated for destination  multi_ptr type
    for_all_combinations<run_convert_assignment_operators_tests, T>(
        address_spaces, is_decorated, address_spaces, is_decorated, type_name);
  }
};

}  // namespace multi_ptr_convert_assignment_ops

#endif  // __SYCLCTS_TESTS_MULTI_PTR_CONVERT_ASSIGN_OPS_H
