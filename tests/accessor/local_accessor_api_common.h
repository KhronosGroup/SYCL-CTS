/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for sycl::local_accessor api tests
//
*******************************************************************************/
#ifndef SYCL_CTS_LOCAL_ACCESSOR_API_COMMON_H
#define SYCL_CTS_LOCAL_ACCESSOR_API_COMMON_H
#include "accessor_common.h"

namespace local_accessor_api_common {
using namespace sycl_cts;
using namespace accessor_tests_common;

template <typename T, typename AccT, sycl::access_mode mode,
          sycl::target target>
void test_local_accessor_types() {
  test_accessor_types_common<T, AccT, mode>();
  STATIC_CHECK(
      std::is_same_v<
          typename AccT::template accessor_ptr<sycl::access::decorated::yes>,
          sycl::multi_ptr<typename AccT::value_type,
                          sycl::access::address_space::local_space,
                          sycl::access::decorated::yes>>);
  STATIC_CHECK(
      std::is_same_v<
          typename AccT::template accessor_ptr<sycl::access::decorated::no>,
          sycl::multi_ptr<typename AccT::value_type,
                          sycl::access::address_space::local_space,
                          sycl::access::decorated::no>>);
}

template <typename T, typename DimensionTypeT>
class run_api_tests {
  static constexpr int dims = DimensionTypeT::value;
  using AccT = sycl::local_accessor<T, dims>;

 public:
  void operator()(const std::string &type_name) {
    auto queue = util::get_cts_object::queue();

    SECTION(
        get_section_name<dims>(type_name, "Check local_accessor alias types")) {
      test_local_accessor_types<T, AccT, AccessModeT, TargetT>();
    }

    SECTION(get_section_name<dims>(type_name,
                                   "Check api for empty local_accessor")) {
      queue
          .submit([&](sycl::handler &cgh) {
            AccT acc;
            test_accessor_methods_common(acc, 0 /* expected_byte_size*/,
                                         0 /*expected_size*/,
                                         util::get_cts_object::range<dims>::get(
                                             0, 0, 0) /*expected_range*/);
          })
          .wait_and_throw();
    }

    SECTION(get_section_name<dims>(type_name, access_mode_name, target_name,
                                   "Check api for local_accessor")) {
      T data(expected_val);
      constexpr size_t global_range_size = 64;
      constexpr size_t local_range_size = 2;
      auto global_range = util::get_cts_object::range<dims>::get(
          global_range_size, global_range_size, global_range_size);
      auto local_range = util::get_cts_object::range<dims>::get(
          local_range_size, local_range_size, local_range_size);
      sycl::nd_range<dim> nd_range(global_range, local_range);
      bool res = false;
      {
        sycl::buffer<T, dims> data_buf(&data, r);
        sycl::buffer res_buf(&res, sycl::range(1));
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc(local_range, cgh);
              AccT acc_other(local_range, cgh);

              test_accessor_methods_common(
                  acc, sizeof(T) * local_range.size() /* expected_byte_size*/,
                  local_range.size() /*expected_size*/,
                  local_range /*expected_range*/);

              sycl::accessor res_acc(res_buf, cgh);
              cgh.parallel_for(nd_range, [=](sycl::nd_item<dim> item) {
                auto ref_1 = acc[0];

                auto id = util::get_cts_object::id<dims>::get(1, 1, 1);
                auto ref_2 = get_subscript_overload<T, AccT, dims>(acc, id);

                res_acc[0] =
                    std::is_same_v<decltype(ref_1), typename AccT::reference>;
                res_acc[0] &=
                    std::is_same_v<decltype(ref_2), typename AccT::reference>;
                if constexpr (!std::is_const_v<T>) {
                  value_helper::change_val(ref_1, expected_val);
                  value_helper::change_val(ref_2, changed_val);

                  res_acc[0] &= value_helper::are_equal(ref_1, expected_val);
                  res_acc[0] &= value_helper::are_equal(ref_2, changed_val);

                  test_accessor_ptr_device(acc, expected_val, res_acc);

                  acc.swap(acc_other);
                  res_acc[0] &=
                      value_helper::are_equal(acc_other[0], expected_val);
                  res_acc[0] &=
                      value_helper::are_equal(acc_other[1], changed_val);
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
class run_local_api_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto dimensions = get_dimensions();

    for_all_combinations<run_api_tests, T>(dimensions, type_name);

    // For covering const types
    const auto const_type_name = std::string("const ") + type_name;
    for_all_combinations<run_api_tests, const T>(dimensions, const_type_name);
  }
};
}  // namespace local_accessor_api_common
#endif  // SYCL_CTS_LOCAL_ACCESSOR_API_COMMON_H
