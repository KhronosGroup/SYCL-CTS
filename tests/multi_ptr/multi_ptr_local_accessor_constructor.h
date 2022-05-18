/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for multi_ptr local_accessor constructors
//
*******************************************************************************/
#ifndef __SYCLCTS_TESTS_MULTI_PTR_LOCAL_ACCESSOR_CONSTRUCTORS_H
#define __SYCLCTS_TESTS_MULTI_PTR_LOCAL_ACCESSOR_CONSTRUCTORS_H

#include "../common/common.h"
#include "../common/section_name_builder.h"
#include "multi_ptr_common.h"

namespace multi_ptr_local_accessor_constructors {

constexpr int expected_val = 42;

template <typename T, typename Address_SpaceT, typename DimensionT>
class run_local_accessor_cnstr_tests {
  static constexpr sycl::access::address_space Space = Address_SpaceT::value;
  static constexpr int dims = DimensionT::value;
  using multi_ptr_t = sycl::multi_ptr<T, Space, sycl::access::decorated::no>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &address_space_name) {
    auto queue = sycl_cts::util::get_cts_object::queue();
    auto r = sycl_cts::util::get_cts_object::range<dims>::get(1, 1, 1);
    SECTION(section_name("Check multi_ptr(local_accessor<T, dims>)")
                .with("T", type_name)
                .with("address_space", address_space_name)
                .with("dimension", dims)
                .create()) {
      bool res = false;
      {
        sycl::buffer<bool, 1> res_buf(&res, sycl::range<1>(1));
        queue.submit([&](sycl::handler &cgh) {
          auto res_acc =
              res_buf.template get_access<sycl::access_mode::write>(cgh);
          sycl::local_accessor<T, dims> acc(r, cgh);
          cgh.parallel_for(sycl::nd_range<dims>(r, r),
                           [=](sycl::nd_item<dims> item) {
                             auto ref = acc[sycl::id<dims>()];
                             value_operations::assign(ref, expected_val);
                             // Creating multi_ptr object with local_accessor
                             // constructor
                             multi_ptr_t mptr(acc);

                             // Check that mptr points at same value as accessor
                             res_acc[0] = (*(mptr.get()) == ref);
                           });
        });
      }
      CHECK(res);
    }
  }
};

template <typename T>
class check_multi_ptr_local_accessor_cnstr_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto dimensions = integer_pack<1, 2, 3>::generate_unnamed();
    const auto address_spaces = value_pack<
        sycl::access::address_space, sycl::access::address_space::local_space,
        sycl::access::address_space::generic_space>::generate_named();

    for_all_combinations<run_local_accessor_cnstr_tests, T>(
        address_spaces, dimensions, type_name);
  }
};

}  // namespace multi_ptr_local_accessor_constructors

#endif  // __SYCLCTS_TESTS_MULTI_PTR_LOCAL_ACCESSOR_CONSTRUCTORS_H
