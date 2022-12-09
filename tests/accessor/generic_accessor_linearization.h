/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for generic sycl::accessor correct linearization test
//  For multidimensional accessors the iterator linearizes the data according
//  to Section 3.11.1
//
*******************************************************************************/
#ifndef SYCL_CTS_GENERIC_ACCESSOR_LINEAR_COMMON_H
#define SYCL_CTS_GENERIC_ACCESSOR_LINEAR_COMMON_H
#include "accessor_common.h"

namespace generic_accessor_linearization {
using namespace accessor_tests_common;

template <typename T, typename AccessT, typename DimensionT, typename TargetT>
class run_linearization_tests {
  static constexpr sycl::access_mode AccessMode = AccessT::value;
  static constexpr int dims = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;
  using AccT = sycl::accessor<T, dims, AccessMode, Target>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &access_mode_name,
                  const std::string &target_name) {
    auto queue = util::get_cts_object::queue();
    auto r = util::get_cts_object::range<dims>::get(1, 1, 1);

    SECTION(
        get_section_name<dims>(type_name, access_mode_name, target_name, "")) {
      constexpr size_t range_size = 8;
      constexpr size_t buff_size = (dims == 3) ? 8 * 8 * 8 : 8 * 8;

      auto range = util::get_cts_object::range<dims>::get(
          range_size, range_size, range_size);

      std::remove_const_t<T> data[buff_size];
      std::iota(data, (data + range.size()), 0);
      bool res = true;
      {
        sycl::buffer<T, dims> data_buf(data, range);
        sycl::buffer res_buf(&res, sycl::range(1));
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc(data_buf, cgh);

              if constexpr (Target == sycl::target::host_task) {
                cgh.host_task([=] {
                  sycl::id<dims> id{};
                  for (auto it = acc.begin(); it < acc.end(); it++) {
                    CHECK(value_operations::are_equal(*it, acc[id]));
                    add_id_linear(id, range_size);
                  }
                });
              } else {
                sycl::accessor res_acc(res_buf, cgh);
                cgh.single_task([=]() {
                  sycl::id<dims> id{};
                  for (auto it = acc.begin(); it < acc.end(); it++) {
                    res_acc[0] &= value_operations::are_equal(*it, acc[id]);
                    add_id_linear(id, range_size);
                  }
                });
              }
            })
            .wait_and_throw();
      }
      if constexpr (Target == sycl::target::device) CHECK(res);
    }
  }
};

template <typename T>
class run_generic_linearization_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto access_modes = get_access_modes();
    const auto dimensions = integer_pack<2, 3>::generate_unnamed();
    const auto targets = get_targets();
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_linearization_tests, T>(access_modes, dimensions,
                                                     targets, actual_type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_linearization_tests, const T>(
        read_only_acc_mode, dimensions, targets, actual_type_name);
  }
};
}  // namespace generic_accessor_linearization
#endif  // SYCL_CTS_GENERIC_ACCESSOR_LINEAR_COMMON_H
