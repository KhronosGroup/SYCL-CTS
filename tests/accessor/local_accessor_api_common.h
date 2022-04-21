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

template <typename T, typename AccT, typename AccRes>
void test_local_accessor_ptr(AccT &accessor, T expected_data, AccRes &res_acc,
                             size_t item_id) {
  auto acc_multi_ptr_no =
      accessor.template get_multi_ptr<sycl::access::decorated::no>();
  res_acc[4, item_id] = std::is_same_v<
      decltype(acc_multi_ptr_no),
      typename AccT::template accessor_ptr<sycl::access::decorated::no>>;
  res_acc[5, item_id] =
      value_helper::are_equal(*acc_multi_ptr_no.get(), expected_data);

  auto acc_multi_ptr_yes =
      accessor.template get_multi_ptr<sycl::access::decorated::yes>();
  res_acc[6, item_id] = std::is_same_v<
      decltype(acc_multi_ptr_yes),
      typename AccT::template accessor_ptr<sycl::access::decorated::yes>>;
  res_acc[7, item_id] =
      value_helper::are_equal(*acc_multi_ptr_yes.get(), expected_data);

  auto acc_pointer = accessor.get_pointer();
  res_acc[8, item_id] =
      std::is_same_v<decltype(acc_pointer),
                     std::add_pointer_t<typename AccT::value_type>>;
  res_acc[9, item_id] = value_helper::are_equal(*acc_pointer, expected_data);
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
      constexpr size_t global_range_size = 4;
      constexpr size_t local_range_size = 2;
      auto global_range = util::get_cts_object::range<dims>::get(
          global_range_size, global_range_size, global_range_size);
      auto local_range = util::get_cts_object::range<dims>::get(
          local_range_size, local_range_size, local_range_size);
      sycl::nd_range<dim> nd_range(global_range, local_range);
      constexpr size_t checks_count = 11;
      bool res[checks_count, global_range.size()];
      std::fill(res, (res + checks_count * global_range.size()), true);
      {
        sycl::buffer res_buf(res,
                             sycl::range<2>(checks_count, global_range.size()));
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
                size_t item_id = item.get_global_linear_id();
                res_acc[0, item_id] =
                    std::is_same_v<decltype(ref_1), typename AccT::reference>;
                res_acc[1, item_id] =
                    std::is_same_v<decltype(ref_2), typename AccT::reference>;
                if constexpr (!std::is_const_v<T>) {
                  value_helper::change_val(ref_1, expected_val);
                  value_helper::change_val(ref_2, changed_val);

                  res_acc[2, item_id] =
                      value_helper::are_equal(ref_1, expected_val);
                  res_acc[3, item_id] =
                      value_helper::are_equal(ref_2, changed_val);

                  test_local_accessor_ptr(acc, expected_val, res_acc);

                  acc.swap(acc_other);
                  res_acc[10, item_id] =
                      value_helper::are_equal(acc_other[0], expected_val);
                  res_acc[10, item_id] &=
                      value_helper::are_equal(acc_other[1], changed_val);
                }
              });
            })
            .wait_and_throw();
      }
      {
        INFO("return type for operator[](id<Dimensions> index)");
        for (size_t i = 0; i < global_range.size(), i++) CHECK(res[0, i]);
      }
      {
        INFO("return type for operator[]](size_t index)");
        for (size_t i = 0; i < global_range.size(), i++) CHECK(res[1, i]);
      }
      if constexpr (!std::is_const_v<T>) {
        {
          INFO("result for operator[](id<Dimensions> index)");
          for (size_t i = 0; i < global_range.size(), i++) CHECK(res[2, i]);
        }
        {
          INFO("result for operator[]](size_t index)");
          for (size_t i = 0; i < global_range.size(), i++) CHECK(res[3, i]);
        }
        {
          INFO("return type for get_multi_ptr<sycl::access::decorated::no>()");
          for (size_t i = 0; i < global_range.size(), i++) CHECK(res[4, i]);
        }
        {
          INFO("result for get_multi_ptr<sycl::access::decorated::no>()");
          for (size_t i = 0; i < global_range.size(), i++) CHECK(res[5, i]);
        }
        {
          INFO("return type for get_multi_ptr<sycl::access::decorated::yes>()");
          for (size_t i = 0; i < global_range.size(), i++) CHECK(res[6, i]);
        }
        {
          INFO("result for get_multi_ptr<sycl::access::decorated::yes>()");
          for (size_t i = 0; i < global_range.size(), i++) CHECK(res[7, i]);
        }
        {
          INFO("return type for get_pointer()");
          for (size_t i = 0; i < global_range.size(), i++) CHECK(res[8, i]);
        }
        {
          INFO("result for get_pointer()");
          for (size_t i = 0; i < global_range.size(), i++) CHECK(res[9, i]);
        }
        {
          INFO("result for swap()");
          for (size_t i = 0; i < global_range.size(), i++) CHECK(res[10, i]);
        }
      }
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
