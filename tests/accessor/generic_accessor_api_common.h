/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for generic sycl::accessor api tests
//
*******************************************************************************/
#ifndef SYCL_CTS_GENERIC_ACCESSOR_API_COMMON_H
#define SYCL_CTS_GENERIC_ACCESSOR_API_COMMON_H
#include "accessor_common.h"

namespace generic_accessor_api_common {
using namespace sycl_cts;
using namespace accessor_tests_common;

template <typename AccT, int dims>
void test_accessor_methods(const AccT &accessor,
                           const size_t expected_byte_size,
                           const size_t expected_size,
                           const bool expected_isPlaceholder,
                           const sycl::range<dims> &expected_range,
                           const sycl::id<dims> &expected_offset) {
  {
    INFO("check is_placeholder() method");
    auto acc_isPlaceholder = accessor.is_placeholder();
    STATIC_CHECK(std::is_same_v<decltype(acc_isPlaceholder), bool>);
    CHECK(acc_isPlaceholder == expected_isPlaceholder);
  }
  {
    INFO("check byte_size() method");
    auto acc_byte_size = accessor.byte_size();
    STATIC_CHECK(std::is_same_v<decltype(acc_byte_size), size_t>);
    CHECK(acc_byte_size == expected_byte_size);
  }
  {
    INFO("check size() method");
    auto acc_size = accessor.size();
    STATIC_CHECK(std::is_same_v<decltype(acc_size), size_t>);
    CHECK(acc_size == expected_size);
  }
  {
    INFO("check max_size() method");
    auto acc_max_size = accessor.max_size();
    STATIC_CHECK(std::is_same_v<decltype(acc_max_size), size_t>);
  }
  {
    INFO("check empty() method");
    auto acc_empty = accessor.empty();
    STATIC_CHECK(std::is_same_v<decltype(acc_empty), bool>);
    CHECK(acc_empty == (expected_size == 0));
  }

#ifdef SYCL_CTS_TEST_DEPRECATED_FEATURES
  {
    INFO("check get_size() method");
    auto acc_get_size = accessor.get_size();
    STATIC_CHECK(std::is_same_v<decltype(acc_get_size), size_t>);
    CHECK(acc_get_size == expected_byte_size);
  }
  {
    INFO("check get_count() method");
    auto acc_get_count = accessor.get_count();
    STATIC_CHECK(std::is_same_v<decltype(acc_get_count), size_t>);
    CHECK(acc_get_count == expected_size);
  }
#endif
  {
    INFO("check get_range() method");
    auto acc_range = accessor.get_range();
    STATIC_CHECK(std::is_same_v<decltype(acc_range), sycl::range<dims>>);
    CHECK(acc_range == expected_range);
  }
  {
    INFO("check get_offset() method");
    auto acc_offset = accessor.get_offset();
    STATIC_CHECK(std::is_same_v<decltype(acc_offset), sycl::id<dims>>);
    CHECK(acc_offset == expected_offset);
  }
}

template <typename T, typename AccT>
void test_accessor_ptr_host(AccT &accessor, T expected_data) {
  {
    INFO("check get_multi_ptr() method");
    auto acc_multi_ptr_no =
        accessor.template get_multi_ptr<sycl::access::decorated::no>();
    STATIC_CHECK(
        std::is_same_v<
            decltype(acc_multi_ptr_no),
            typename AccT::template accessor_ptr<sycl::access::decorated::no>>);
    CHECK(value_helper::compare_vals(*acc_multi_ptr_no.get(), expected_data));

    auto acc_multi_ptr_yes =
        accessor.template get_multi_ptr<sycl::access::decorated::yes>();
    STATIC_CHECK(std::is_same_v<decltype(acc_multi_ptr_yes),
                                typename AccT::template accessor_ptr<
                                    sycl::access::decorated::yes>>);
    CHECK(value_helper::compare_vals(*acc_multi_ptr_yes.get(), expected_data));
  }

  {
    INFO("check get_pointer() method");
    auto acc_pointer = accessor.get_pointer();
    STATIC_CHECK(std::is_same_v<decltype(acc_pointer), std::add_pointer_t<T>>);
    CHECK(value_helper::compare_vals(*acc_pointer, expected_data));
  }
}

template <typename T, typename AccT, typename AccRes>
void test_accessor_ptr_device(AccT &accessor, T expected_data, AccRes &res) {
  auto acc_multi_ptr_no =
      accessor.template get_multi_ptr<sycl::access::decorated::no>();
  res_acc[0] = std::is_same_v<
      decltype(acc_multi_ptr_no),
      typename AccT::template accessor_ptr<sycl::access::decorated::no>>;
  res_acc[0] &=
      value_helper::compare_vals(*acc_multi_ptr_no.get(), expected_data);

  auto acc_multi_ptr_yes =
      accessor.template get_multi_ptr<sycl::access::decorated::yes>();
  res_acc[0] &= std::is_same_v<
      decltype(acc_multi_ptr_yes),
      typename AccT::template accessor_ptr<sycl::access::decorated::yes>>;
  res_acc[0] &=
      value_helper::compare_vals(*acc_multi_ptr_yes.get(), expected_data);

  auto acc_pointer = accessor.get_pointer();
  res_acc[0] &= std::is_same_v<decltype(acc_pointer), std::add_pointer_t<T>>;
  res_acc[0] &= value_helper::compare_vals(*acc_pointer, expected_data);
}

template <typename T, typename AccT, int dims>
T &get_subscript_overload(const AccT &accessor, size_t index) {
  if constexpr (dims == 1) return accessor[index];
  if constexpr (dims == 2) return accessor[index][index];
  if constexpr (dims == 3) return accessor[index][index][index];
}

template <typename T, typename AccessTypeT, typename DimensionTypeT,
          typename TargetTypeT>
class run_api_tests {
  static constexpr sycl::access_mode AccessModeT = AccessTypeT::value;
  static constexpr int dims = DimensionTypeT::value;
  static constexpr sycl::target TargetT = TargetTypeT::value;
  using AccT = sycl::accessor<T, dims, AccessModeT, TargetT>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &access_mode_name,
                  const std::string &target_name) {
    auto queue = util::get_cts_object::queue();
    auto r = util::get_cts_object::range<dims>::get(1, 1, 1);

    SECTION(get_section_name<dims>(type_name, access_mode_name, target_name,
                                   "Check api for empty accessor")) {
      queue
          .submit([&](sycl::handler &cgh) {
            AccT acc;
            test_accessor_methods(acc, 0 /* expected_byte_size*/,
                                  0 /*expected_size*/,
                                  false /*expected_isPlaceholder*/,
                                  util::get_cts_object::range<dims>::get(
                                      0, 0, 0) /*expected_range*/,
                                  sycl::id<dims>() /*&expected_offset)*/);
          })
          .wait_and_throw();
    }

    SECTION(
        get_section_name<dims>(type_name, access_mode_name, target_name,
                               "Check api for buffer placeholder accessor")) {
      T data(expected_val);
      bool res = false;
      {
        sycl::buffer<T, dims> data_buf(&data, r);
        sycl::buffer res_buf(&res, sycl::range(1));
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc(data_buf);

              test_accessor_methods(acc, sizeof(T) /* expected_byte_size*/,
                                    1 /*expected_size*/,
                                    true /*expected_isPlaceholder*/,
                                    util::get_cts_object::range<dims>::get(
                                        1, 1, 1) /*expected_range*/,
                                    sycl::id<dims>() /*&expected_offset)*/);

              test_accessor_ptr(acc, data);
            })
            .wait_and_throw();
      }
    }

    SECTION(get_section_name<dims>(type_name, access_mode_name, target_name,
                                   "Check api for buffer accessor")) {
      T data(expected_val);
      bool res = false;
      {
        sycl::buffer<T, dims> data_buf(&data, r);
        sycl::buffer res_buf(&res, sycl::range(1));
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc(data_buf, cgh);

              test_accessor_methods(acc, sizeof(T) /* expected_byte_size*/,
                                    1 /*expected_size*/,
                                    false /*expected_isPlaceholder*/,
                                    util::get_cts_object::range<dims>::get(
                                        1, 1, 1) /*expected_range*/,
                                    sycl::id<dims>() /*&expected_offset)*/);

              if constexpr (TargetT == sycl::target::host_task) {
                cgh.host_task([=] {
                  test_accessor_ptr_host(acc, expected_val);
                  auto &acc_ref = acc[sycl::id<dims>()];
                  CHECK(value_helper::compare_vals(acc_ref, expected_val));
                  STATIC_CHECK(std::is_same_v<decltype(acc_ref),
                                              typename AccT::reference>);
                  if constexpr (AccessModeT != sycl::access_mode::read)
                    value_helper::change_val(acc_ref, changed_val);
                });
              } else {
                sycl::accessor res_acc(res_buf, cgh);
                cgh.single_task([acc, res_acc]() {
                  test_accessor_ptr_device(acc, expected_val, res_acc);
                  auto &acc_ref = acc[sycl::id<dims>()];
                  res_acc[0] &=
                      value_helper::compare_vals(acc_ref, expected_val);
                  res_acc[0] &= std::is_same_v<decltype(acc_ref),
                                               typename AccT::reference>;
                  if constexpr (AccessModeT != sycl::access_mode::read)
                    value_helper::change_val(acc_ref, changed_val);
                });
              }
            })
            .wait_and_throw();
      }
      if constexpr (TargetT == sycl::target::device) CHECK(res);
      if constexpr (AccessModeT != sycl::access_mode::read)
        CHECK(value_helper::compare_vals(data, changed_val));
    }
    SECTION(
        get_section_name<dims>(type_name, access_mode_name, target_name,
                               "Check api for ranged accessor with offset")) {
      constexpr size_t acc_range_size = 4;
      constexpr size_t buff_range_size = 8;
      constexpr size_t buff_size = (dims == 3)   ? 8 * 8 * 8
                                   : (dims == 2) ? 8 * 8
                                                 : 8;
      constexpr size_t offset = 4;
      constexpr size_t index = 2;
      int linear_index = 0;
      for (size_t i = 0; i < dims; i++) {
        linear_index += (offset + index) * pow(buff_range_size, dims - i - 1);
      }
      auto acc_range = util::get_cts_object::range<dims>::get(
          acc_range_size, acc_range_size, acc_range_size);
      auto buff_range = util::get_cts_object::range<dims>::get(
          buff_range_size, buff_range_size, buff_range_size);
      auto offset_id =
          util::get_cts_object::id<dims>::get(offset, offset, offset);
      T data[buff_size];
      std::iota(data, (data + buff_range.size()), 0);
      bool res = false;
      {
        sycl::buffer<T, dims> data_buf(data, buff_range);
        sycl::buffer res_buf(&res, sycl::range(1));
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc(data_buf, cgh, acc_range, offset_id);
              test_accessor_methods(
                  acc, sizeof(T) * acc_range.size() /* expected_byte_size*/,
                  acc_range.size() /*expected_size*/,
                  false /*expected_isPlaceholder*/,
                  acc_range /*expected_range*/,
                  offset_id /*&expected_offset)*/);

              if constexpr (TargetT == sycl::target::host_task) {
                cgh.host_task([=] {
                  test_accessor_ptr_host(acc, T(0));
                  auto &acc_ref =
                      get_subscript_overload<T, AccT, dims>(acc, index);
                  CHECK(value_helper::compare_vals(acc_ref, linear_index));
                  if constexpr (AccessModeT != sycl::access_mode::read)
                    value_helper::change_val(acc_ref, changed_val);
                });
              } else {
                sycl::accessor res_acc(res_buf, cgh);
                cgh.single_task([=]() {
                  test_accessor_ptr_device(acc, T(0), res_acc);
                  auto &acc_ref =
                      get_subscript_overload<T, AccT, dims>(acc, index);
                  res_acc[0] &=
                      value_helper::compare_vals(acc_ref, linear_index);
                  if constexpr (AccessModeT != sycl::access_mode::read)
                    value_helper::change_val(acc_ref, changed_val);
                });
              }
            })
            .wait_and_throw();
      }
      if constexpr (TargetT == sycl::target::device) CHECK(res);
      if constexpr (AccessModeT != sycl::access_mode::read)
        CHECK(value_helper::compare_vals(data[linear_index], changed_val));
    }
    if constexpr (std::is_const_v<T>) {
      SECTION(get_section_name<dims>(type_name, access_mode_name, target_name,
                                     "Check swap for accessor")) {
        T data1(expected_val);
        T data2(changed_val);
        {
          sycl::buffer<T, dims> data_buf1(&data1, r);
          sycl::buffer<T, dims> data_buf2(&data2, r);
          queue
              .submit([&](sycl::handler &cgh) {
                AccT acc1(data_buf1);
                AccT acc2(data_buf2);
                acc1.swap(acc2);
              })
              .wait_and_throw();
        }
        CHECK(value_helper::compare_vals(data1, changed_val));
        CHECK(value_helper::compare_vals(data2, expected_val));
      }
    }
  }
};

template <typename T>
class run_generic_api_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto access_modes = get_access_modes();
    const auto dimensions = get_dimensions();
    const auto targets = get_targets();
    const auto cur_type =
        named_type_pack<T>::generate(type_name_string<T>::get(type_name));

    for_all_combinations<run_api_tests>(cur_type, access_modes, dimensions,
                                        targets);
  }
};
}  // namespace generic_accessor_api_common
#endif  // SYCL_CTS_GENERIC_ACCESSOR_API_COMMON_H
