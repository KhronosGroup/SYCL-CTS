
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

template <typename T, typename multi_ptr_t, typename ResAcc>
void check(const multi_ptr_t &const_mptr_in, multi_ptr_t &mptr_in, T val,
           ResAcc &res_acc) {
  multi_ptr_t mptr_out1, mptr_out2, mptr_out3;

  mptr_out1 = const_mptr_in;
  mptr_out2 = std::move(mptr_in);
  mptr_out3 = nullptr;

  // Check that out mptr has the same value as in mptr
  res_acc[0] = *(mptr_out1.get_raw()) == val;
  res_acc[1] = *(mptr_out2.get_raw()) == val;
  res_acc[2] = mptr_out3.get_raw() == nullptr;
}

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
    sycl::range r(1);
    std::array<bool, 3> res;
    res.fill(false);
    {
      sycl::buffer<bool> res_buf(res.data(), sycl::range<1>(3));
      sycl::buffer<T> val_buffer(&value, sycl::range<1>(1));
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            res_buf.template get_access<sycl::access_mode::write>(cgh);
        auto val_acc =
            val_buffer.template get_access<sycl::access_mode::read>(cgh);
        if constexpr (space == sycl::access::address_space::global_space ||
                      space == sycl::access::address_space::generic_space) {
          cgh.single_task([=] {
            const multi_ptr_t const_mptr_in(val_acc);
            multi_ptr_t mptr_in(val_acc);

            check(const_mptr_in, mptr_in, val_acc[0], res_acc);
          });
        } else {
          sycl::local_accessor<T> local_acc(r, cgh);
          cgh.parallel_for(sycl::nd_range<1>(r, r), [=](sycl::nd_item<1> item) {
            if constexpr (space == sycl::access::address_space::local_space) {
              auto ref = local_acc[0];
              value_operations::assign(ref, expected_val);
              const multi_ptr_t const_mptr_in(local_acc);
              multi_ptr_t mptr_in(local_acc);

              check(const_mptr_in, mptr_in, ref, res_acc);
            } else {
              T private_val =
                  user_def_types::get_init_value_helper<T>(expected_val);

              multi_ptr_t mptr_in =
                  sycl::address_space_cast<space, decorated, T>(&private_val);
              const multi_ptr_t const_mptr_in =
                  sycl::address_space_cast<space, decorated, T>(&private_val);

              check(const_mptr_in, mptr_in, private_val, res_acc);
            }
          });
        }
      });
    }
    SECTION(section_name("Check &operator=(const multi_ptr&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(res[0]);
    }

    SECTION(section_name("Check &operator=(multi_ptr&&)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(res[1]);
    }

    SECTION(section_name("Check &operator=(std::nullptr_t)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("decorated", is_decorated_name)
                .create()) {
      CHECK(res[2]);
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
