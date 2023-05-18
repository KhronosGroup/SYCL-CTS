/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

//  Provides common code for sycl::local_accessor api tests

#ifndef SYCL_CTS_LOCAL_ACCESSOR_API_COMMON_H
#define SYCL_CTS_LOCAL_ACCESSOR_API_COMMON_H
#include "accessor_common.h"

namespace local_accessor_api_common {
using namespace sycl_cts;
using namespace accessor_tests_common;

enum class check : size_t {
  iterator_access = 0U,
  get_multi_ptr_no_type,
  get_multi_ptr_yes_type,
  get_pointer_type,
  subscript_id_type,      // only for non-zero dim accessor
  subscript_size_t_type,  // only for non-zero dim accessor
  // only for non const
  subscript_id_result,        // only for non-zero dim accessor
  subscript_size_t_result,    // only for non-zero dim accessor
  operator_assign_lv_result,  // only for zero dim accessor
  operator_assign_rv_result,  // only for zero dim accessor
  get_multi_ptr_no_result,
  get_multi_ptr_yes_result,
  get_pointer_result,
  nChecks  // should be the latest one
};
const std::string info_strings[to_integral(check::nChecks)]{
    "result for iterator_access",
    "return type for get_multi_ptr<sycl::access::decorated::no>()",
    "return type for get_multi_ptr<sycl::access::decorated::yes>()",
    "return type for get_pointer()",
    "return type for operator[](id<Dimensions> index)",
    "return type for operator[]](size_t index)",
    "result for operator[](id<Dimensions> index)",
    "result for operator[]](size_t index)",
    "result for operator=(const T& value)",
    "result for operator=(T&& value)",
    "result for get_multi_ptr<sycl::access::decorated::no>()",
    "result for get_multi_ptr<sycl::access::decorated::yes>()",
    "result for get_pointer()"};

bool is_check_for_dim(check check_value, int dim) {
  if (0 == dim) {
    if (check_value == check::subscript_id_type ||
        check_value == check::subscript_size_t_type ||
        check_value == check::subscript_id_result ||
        check_value == check::subscript_size_t_result) {
      return false;
    }
  } else if (check_value == check::operator_assign_lv_result ||
             check_value == check::operator_assign_rv_result) {
    return false;
  }
  return true;
}

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

  auto acc_multi_ptr_yes =
      accessor.template get_multi_ptr<sycl::access::decorated::yes>();
  res_acc[sycl::id<2>(to_integral(check::get_multi_ptr_yes_type), item_id)] =
      std::is_same_v<
          decltype(acc_multi_ptr_yes),
          typename AccT::template accessor_ptr<sycl::access::decorated::yes>>;

  auto acc_pointer = accessor.get_pointer();
  res_acc[sycl::id<2>(to_integral(check::get_pointer_type), item_id)] =
      std::is_same_v<decltype(acc_pointer),
                     std::add_pointer_t<typename AccT::value_type>>;
  if constexpr (!std::is_const_v<typename AccT::value_type>) {
    res_acc[sycl::id<2>(to_integral(check::get_multi_ptr_no_result), item_id)] =
        value_operations::are_equal(*acc_multi_ptr_no, expected_data);
    res_acc[sycl::id<2>(to_integral(check::get_multi_ptr_yes_result),
                        item_id)] =
        value_operations::are_equal(*(acc_multi_ptr_yes.get_raw()),
                                    expected_data);
    res_acc[sycl::id<2>(to_integral(check::get_pointer_result), item_id)] =
        value_operations::are_equal(*acc_pointer, expected_data);
  }
}

template <typename AccT, int dims, int range_dims>
AccT get_accessor(sycl::handler &cgh, const sycl::range<range_dims> &r) {
  if constexpr (0 == dims) {
    return AccT(cgh);
  } else {
    return AccT(r, cgh);
  }
}

template <typename T, typename DimensionTypeT>
class kernel_api;

template <typename T, typename DimensionTypeT>
class run_api_tests {
  static constexpr int dims = DimensionTypeT::value;
  using AccT = sycl::local_accessor<T, dims>;

 public:
  void operator()(const std::string &type_name) {
    using kname = kernel_api<T, DimensionTypeT>;
    auto queue = once_per_unit::get_queue();
    constexpr int buf_dims = (0 == dims) ? 1 : dims;

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
                                         0 /*expected_size*/);
            if constexpr (0 < dims) {
              test_accessor_get_range_method(
                  acc, util::get_cts_object::range<dims>::get(
                           0, 0, 0) /*expected_range*/);
            }
          })
          .wait_and_throw();
    }
    if constexpr (0 < dims) {
      SECTION(get_section_name<dims>(
          type_name, "Check api for non zero-dimension local_accessor")) {
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
                    local_range.size() /*expected_size*/);
                test_accessor_get_range_method(acc,
                                               local_range /*expected_range*/);

                sycl::accessor res_acc(res_buf, cgh);
                cgh.parallel_for<kname>(nd_range, [=](sycl::nd_item<dims>
                                                          item) {
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

                    res_acc[sycl::id<2>(to_integral(check::subscript_id_result),
                                        item_id)] =
                        value_operations::are_equal(ref_1, expected_val);
                    res_acc[sycl::id<2>(
                        to_integral(check::subscript_size_t_result), item_id)] =
                        value_operations::are_equal(ref_2, changed_val);
                  }
                  test_local_accessor_ptr(acc, expected_val, res_acc, item_id);
                  res_acc[sycl::id<2>(to_integral(check::iterator_access),
                                      item_id)] =
                      test_begin_end_device(acc, expected_val, changed_val,
                                            !std::is_const_v<T>);
                });
              })
              .wait_and_throw();
        }

        constexpr size_t N = to_integral(
            (std::is_const_v<T>) ? check::subscript_id_result : check::nChecks);
        for (size_t i = 0; i < N; i++) {
          if (is_check_for_dim(static_cast<check>(i), dims)) {
            INFO(info_strings[i]);
            CHECK(std::all_of(res[i].cbegin(), res[i].cend(),
                              [](bool v) { return v; }));
          }
        }
      }
    } else {
      SECTION(get_section_name<dims>(
          type_name, "Check api for zero-dimension local_accessor")) {
        // res as array of arrays just for unification of check with non-zero
        // dim accessors
        std::array<std::array<bool, 1>, to_integral(check::nChecks)> res;
        std::for_each(res.begin(), res.end(),
                      [](std::array<bool, 1> &arr) { arr.fill(false); });
        {
          auto r = util::get_cts_object::range<buf_dims>::get(1, 1, 1);
          sycl::buffer<bool, 2> res_buf(
              res.data()->data(),
              sycl::range<2>(to_integral(check::nChecks), 1));
          queue
              .submit([&](sycl::handler &cgh) {
                AccT acc(cgh);

                test_accessor_methods_common(
                    acc, sizeof(T) * r.size() /* expected_byte_size*/,
                    r.size() /*expected_size*/);

                sycl::accessor res_acc(res_buf, cgh);
                cgh.parallel_for<kname>(
                    sycl::nd_range(r, r), [=](sycl::nd_item<1> item) {
                      size_t item_id = item.get_global_linear_id();
                      typename AccT::reference dref = acc;
                      if constexpr (!std::is_const_v<T>) {
                        typename AccT::value_type v_data =
                            value_operations::init<typename AccT::value_type>(
                                expected_val);
                        // check method const AccT::operator=(const T& data)
                        // const
                        acc = v_data;
                        res_acc[sycl::id<2>(
                            to_integral(check::operator_assign_lv_result),
                            item_id)] =
                            value_operations::are_equal(dref, v_data);

                        // check method const AccT::operator=(T&& data) const
                        acc = value_operations::init<typename AccT::value_type>(
                            changed_val);
                        v_data =
                            value_operations::init<typename AccT::value_type>(
                                changed_val);
                        res_acc[sycl::id<2>(
                            to_integral(check::operator_assign_rv_result),
                            item_id)] =
                            value_operations::are_equal(dref, v_data);
                      }
                      test_local_accessor_ptr(acc, changed_val, res_acc,
                                              item_id);
                      res_acc[sycl::id<2>(to_integral(check::iterator_access),
                                          item_id)] =
                          test_begin_end_device(acc, expected_val, changed_val,
                                                !std::is_const_v<T>);
                    });
              })
              .wait_and_throw();
        }
        constexpr size_t N = to_integral(
            (std::is_const_v<T>) ? check::subscript_id_result : check::nChecks);
        for (size_t i = 0; i < N; i++) {
          if (is_check_for_dim(static_cast<check>(i), dims)) {
            INFO(info_strings[i]);
            CHECK(std::all_of(res[i].cbegin(), res[i].cend(),
                              [](bool v) { return v; }));
          }
        }
      }
    }
    SECTION(
        get_section_name<dims>(type_name, "Check swap() for local_accessor")) {
      constexpr size_t alloc_size = 2;
      auto local_range = util::get_cts_object::range<buf_dims>::get(
          alloc_size, alloc_size, alloc_size);
      queue.submit([&](sycl::handler &cgh) {
        AccT acc1 = get_accessor<AccT, dims>(cgh, local_range);
        AccT acc2;
        acc2.swap(acc1);
        if constexpr (0 != dims) {
          CHECK(acc1.get_range() ==
                util::get_cts_object::range<dims>::get(0, 0, 0));
          CHECK(acc2.get_range() == local_range);
        }
        CHECK(acc1.empty());
        CHECK(!acc2.empty());
      });
    }
  }
};

template <typename T>
class run_local_api_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto dimensions = get_all_dimensions();

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
