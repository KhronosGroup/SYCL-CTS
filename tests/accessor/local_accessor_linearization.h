/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for sycl::local_accessor correct linearization test
//  For multidimensional accessors the iterator linearizes the data according
//  to Section 3.11.1
//
*******************************************************************************/
#ifndef SYCL_CTS_LOCAL_ACCESSOR_LINEAR_COMMON_H
#define SYCL_CTS_LOCAL_ACCESSOR_LINEAR_COMMON_H
#include "accessor_common.h"

namespace local_accessor_linearization {
using namespace accessor_tests_common;

template <typename T, typename DimensionT>
class run_linearization_tests {
  static constexpr int dims = DimensionT::value;
  using AccT = sycl::local_accessor<T, dims>;

 public:
  void operator()(const std::string &type_name) {
    auto queue = util::get_cts_object::queue();
    auto r = util::get_cts_object::range<dims>::get(1, 1, 1);

    SECTION(get_section_name<dims>(type_name, "")) {
      constexpr size_t local_range_size = 4;
      constexpr size_t range_size = 8;

      auto range = util::get_cts_object::range<dims>::get(
          range_size, range_size, range_size);
      auto local_range = util::get_cts_object::range<dims>::get(
          local_range_size, local_range_size, local_range_size);
      sycl::nd_range<dims> nd_range(range, local_range);

      bool res = true;
      {
        sycl::buffer res_buf(&res, sycl::range(1));
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc(local_range, cgh);
              sycl::accessor res_acc(res_buf, cgh);
              cgh.parallel_for(nd_range, [=](sycl::nd_item<dims> item) {
                acc[item.get_local_id()] =
                    value_operations::init<T>(item.get_global_linear_id());
                sycl::group_barrier(item.get_group());
                sycl::id<dims> id{};
                for (auto it = acc.begin(); it < acc.end(); it++) {
                  res_acc[0] &= value_operations::are_equal(*it, acc[id]);
                  add_id_linear(id, local_range_size);
                }
              });
            })
            .wait_and_throw();
      }
      CHECK(res);
    }
  }
};

template <typename T>
class run_local_linearization_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto dimensions = integer_pack<2, 3>::generate_unnamed();
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_linearization_tests, T>(dimensions,
                                                     actual_type_name);
  }
};
}  // namespace local_accessor_linearization
#endif  // SYCL_CTS_LOCAL_ACCESSOR_LINEAR_COMMON_H
