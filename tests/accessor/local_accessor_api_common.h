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

enum class check : size_t {
  subscript_id_type = 0U,
  subscript_size_t_type,
  // only for non const
  subscript_id_result,
  subscript_size_t_result,
  get_multi_ptr_no_type,
  get_multi_ptr_no_result,
  get_multi_ptr_yes_type,
  get_multi_ptr_yes_result,
  get_pointer_type,
  get_pointer_result,
  nChecks  // should be the latest one
};

template <typename T, typename AccT>
void test_local_accessor_types() {
  constexpr sycl::access_mode mode = (std::is_const_v<T>)
                                         ? sycl::access_mode::read
                                         : sycl::access_mode::read_write;
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
  res_acc[sycl::id<2>(to_integral(check::get_multi_ptr_no_type), item_id)] =
      std::is_same_v<
          decltype(acc_multi_ptr_no),
          typename AccT::template accessor_ptr<sycl::access::decorated::no>>;
  res_acc[sycl::id<2>(to_integral(check::get_multi_ptr_no_result), item_id)] =
      value_operations::are_equal(*acc_multi_ptr_no.get(), expected_data);

  auto acc_multi_ptr_yes =
      accessor.template get_multi_ptr<sycl::access::decorated::yes>();
  res_acc[sycl::id<2>(to_integral(check::get_multi_ptr_yes_type), item_id)] =
      std::is_same_v<
          decltype(acc_multi_ptr_yes),
          typename AccT::template accessor_ptr<sycl::access::decorated::yes>>;
  res_acc[sycl::id<2>(to_integral(check::get_multi_ptr_yes_result), item_id)] =
      value_operations::are_equal(*acc_multi_ptr_yes.get(), expected_data);

  auto acc_pointer = accessor.get_pointer();
  res_acc[sycl::id<2>(to_integral(check::get_pointer_type), item_id)] =
      std::is_same_v<decltype(acc_pointer),
                     std::add_pointer_t<typename AccT::value_type>>;
  res_acc[sycl::id<2>(to_integral(check::get_pointer_result), item_id)] =
      value_operations::are_equal(*acc_pointer, expected_data);
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
      test_local_accessor_types<T, AccT>();
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

    SECTION(get_section_name<dims>(type_name, "Check api for local_accessor")) {
      constexpr size_t global_range_size = 4;
      constexpr size_t global_range_buffer_size = (dims == 3)   ? 4 * 4 * 4
                                                  : (dims == 2) ? 4 * 4
                                                                : 4;
      constexpr size_t local_range_size = 2;
      auto global_range = util::get_cts_object::range<dims>::get(
          global_range_size, global_range_size, global_range_size);
      auto local_range = util::get_cts_object::range<dims>::get(
          local_range_size, local_range_size, local_range_size);
      sycl::nd_range<dims> nd_range(global_range, local_range);
      std::array<std::array<bool, global_range_buffer_size>,
                 to_integral(check::nChecks)>
          res;
      std::for_each(res.begin(), res.end(),
                    [](std::array<bool, global_range_buffer_size> &arr) {
                      arr.fill(false);
                    });
      {
        sycl::buffer<bool, 2> res_buf(
            res.data()->data(), sycl::range<2>(to_integral(check::nChecks),
                                               global_range_buffer_size));
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc(local_range, cgh);
              AccT acc_other(local_range, cgh);

              test_accessor_methods_common(
                  acc, sizeof(T) * local_range.size() /* expected_byte_size*/,
                  local_range.size() /*expected_size*/,
                  local_range /*expected_range*/);

              sycl::accessor res_acc(res_buf, cgh);
              cgh.parallel_for(nd_range, [=](sycl::nd_item<dims> item) {
                auto &&ref_1 = acc[sycl::id<dims>()];

                auto &&ref_2 = get_subscript_overload<T, AccT, dims>(acc, 1);
                size_t item_id = item.get_global_linear_id();
                res_acc[sycl::id<2>(to_integral(check::subscript_id_type),
                                    item_id)] =
                    std::is_same_v<decltype(ref_1), typename AccT::reference>;
                res_acc[sycl::id<2>(to_integral(check::subscript_size_t_type),
                                    item_id)] =
                    std::is_same_v<decltype(ref_2), typename AccT::reference>;
                if constexpr (!std::is_const_v<T>) {
                  value_operations::assign(ref_1, expected_val);
                  value_operations::assign(ref_2, changed_val);

                  res_acc[sycl::id<2>(to_integral(check::subscript_id_result), item_id)] =
                      value_operations::are_equal(ref_1, expected_val);
                  res_acc[sycl::id<2>(to_integral(check::subscript_size_t_result), item_id)] =
                      value_operations::are_equal(ref_2, changed_val);

                  test_local_accessor_ptr(acc, expected_val, res_acc, item_id);
                }
              });
            })
            .wait_and_throw();
      }

      std::string info_strings[to_integral(check::nChecks)]{
          "return type for operator[](id<Dimensions> index)",
          "return type for operator[]](size_t index)",
          "result for operator[](id<Dimensions> index)",
          "result for operator[]](size_t index)",
          "return type for get_multi_ptr<sycl::access::decorated::no>()",
          "result for get_multi_ptr<sycl::access::decorated::no>()",
          "return type for get_multi_ptr<sycl::access::decorated::yes>()",
          "result for get_multi_ptr<sycl::access::decorated::yes>()",
          "return type for get_pointer()",
          "result for get_pointer()"};

      constexpr size_t N = to_integral(
          (std::is_const_v<T>) ? check::subscript_id_result : check::nChecks);
      for (size_t i = 0; i < N; i++) {
        INFO(info_strings[i]);
        CHECK(std::all_of(res[i].cbegin(), res[i].cend(),
                          [](bool v) { return v; }));
      }
    }

    SECTION(
        get_section_name<dims>(type_name, "Check swap() for local_accessor")) {
      constexpr size_t alloc_size = 2;
      auto local_range = util::get_cts_object::range<dims>::get(
          alloc_size, alloc_size, alloc_size);
      queue.submit([&](sycl::handler &cgh) {
        AccT acc1{local_range, cgh};
        AccT acc2;
        acc2.swap(acc1);
        CHECK(acc1.get_range() ==
              util::get_cts_object::range<dims>::get(0, 0, 0));
        CHECK(acc2.get_range() == local_range);
      });
    }
  }
};

template <typename T>
class run_local_api_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto dimensions = get_dimensions();

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or for_all_device_copyable_std_containers.
    // This will wrap string with type T to string with container<T> if T is
    // an array or other kind of container.
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_api_tests, T>(dimensions, actual_type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    for_all_combinations<run_api_tests, const T>(dimensions, actual_type_name);
  }
};
}  // namespace local_accessor_api_common
#endif  // SYCL_CTS_LOCAL_ACCESSOR_API_COMMON_H
