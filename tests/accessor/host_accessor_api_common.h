/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for sycl::host_accessor api tests
//
*******************************************************************************/
#ifndef SYCL_CTS_HOST_ACCESSOR_API_COMMON_H
#define SYCL_CTS_HOST_ACCESSOR_API_COMMON_H
#include "accessor_common.h"

namespace host_accessor_api_common {
using namespace sycl_cts;
using namespace accessor_tests_common;

template <typename AccT, int dims>
void test_host_accessor_methods(const AccT &accessor,
                                const size_t expected_byte_size,
                                const size_t expected_size,
                                const sycl::range<dims> &expected_range,
                                const sycl::id<dims> &expected_offset) {
  test_accessor_methods_common<AccT, dims>(accessor, expected_byte_size,
                                           expected_size, expected_range);

  {
    INFO("check get_offset() method");
    auto acc_offset = accessor.get_offset();
    STATIC_CHECK(std::is_same_v<decltype(acc_offset), sycl::id<dims>>);
    CHECK(acc_offset == expected_offset);
  }
}

template <typename T, typename AccT>
void test_accessor_ptr(AccT &accessor, T expected_data) {
  {
    INFO("check get_pointer() method");
    auto acc_pointer = accessor.get_pointer();
    STATIC_CHECK(std::is_same_v<decltype(acc_pointer),
                                std::add_pointer_t<typename AccT::value_type>>);
    CHECK(value_operations::are_equal(*acc_pointer, expected_data));
  }
}

template <typename T, typename AccessT, typename DimensionT>
class run_api_tests {
  static constexpr sycl::access_mode AccessMode = AccessT::value;
  static constexpr int dims = DimensionT::value;
  using AccT = sycl::host_accessor<T, dims, AccessMode>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &access_mode_name) {
    auto r = util::get_cts_object::range<dims>::get(1, 1, 1);

    SECTION(get_section_name<dims>(type_name, access_mode_name,
                                   "Check host_accessor alias types")) {
      test_accessor_types_common<T, AccT, AccessMode>();
    }

    SECTION(get_section_name<dims>(type_name, access_mode_name,
                                   "Check api for empty host_accessor")) {
      AccT acc;
      test_host_accessor_methods(
          acc, 0 /* expected_byte_size*/, 0 /*expected_size*/,
          util::get_cts_object::range<dims>::get(0, 0, 0) /*expected_range*/,
          sycl::id<dims>() /*&expected_offset)*/);
    }

    SECTION(get_section_name<dims>(type_name, access_mode_name,
                                   "Check api for host_accessor")) {
      T data = value_operations::init<T>(expected_val);
      bool res = false;
      {
        sycl::buffer<T, dims> data_buf(&data, r);
        AccT acc{data_buf};

        test_host_accessor_methods(
            acc, sizeof(T) /* expected_byte_size*/, 1 /*expected_size*/,
            util::get_cts_object::range<dims>::get(1, 1, 1) /*expected_range*/,
            sycl::id<dims>() /*&expected_offset)*/);
        test_accessor_ptr(acc, expected_val);
        auto &acc_ref = acc[sycl::id<dims>()];
        CHECK(value_operations::are_equal(acc_ref, expected_val));
        STATIC_CHECK(
            std::is_same_v<decltype(acc_ref), typename AccT::reference>);
        if constexpr (AccessMode != sycl::access_mode::read)
          value_operations::assign(acc_ref, changed_val);
      }
      if constexpr (AccessMode != sycl::access_mode::read)
        CHECK(value_operations::are_equal(data, changed_val));
    }

    SECTION(get_section_name<dims>(
        type_name, access_mode_name, target_name,
        "Check api for ranged host_accessor with offset")) {
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
      std::remove_const_t<T> data[buff_size];
      for (int i = 0; i < buff_size; i++)
        data[i] = value_operations::init<T>(i);
      bool res = false;
      {
        sycl::buffer<T, dims> data_buf(data, buff_range);
        AccT acc(data_buf, acc_range, offset_id);
        test_host_accessor_methods(
            acc, sizeof(T) * acc_range.size() /* expected_byte_size*/,
            acc_range.size() /*expected_size*/, acc_range /*expected_range*/,
            offset_id /*&expected_offset)*/);

        test_accessor_ptr(acc, T());
        auto &acc_ref = get_subscript_overload<T, AccT, dims>(acc, index);
        CHECK(value_operations::are_equal(acc_ref, linear_index));
        if constexpr (AccessMode != sycl::access_mode::read)
          value_operations::assign(acc_ref, changed_val);
      }
      if constexpr (AccessMode != sycl::access_mode::read)
        CHECK(value_operations::are_equal(data[linear_index], changed_val));
    }
    if constexpr (AccessMode != sycl::access_mode::read) {
      SECTION(get_section_name<dims>(type_name, access_mode_name,
                                     "Check swap for host_accessor")) {
        T data1 = value_operations::init<T>(expected_val);
        T data2 = value_operations::init<T>(changed_val);
        {
          sycl::buffer<T, dims> data_buf1(&data1, r);
          sycl::buffer<T, dims> data_buf2(&data2, r);
          AccT acc1(data_buf1);
          AccT acc2(data_buf2);
          acc1.swap(acc2);
        }
        CHECK(value_operations::are_equal(data1, changed_val));
        CHECK(value_operations::are_equal(data2, expected_val));
      }
    }
  }
};

template <typename T>
class run_host_accessor_api_for_type {
 public:
  void operator()(const std::string &type_name) {
    const auto access_modes = get_access_modes();
    const auto dimensions = get_dimensions();

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or for_all_device_copyable_std_containers.
    // This will wrap string with type T to string with container<T> if T is
    // an array or other kind of container.
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_api_tests>(access_modes, dimensions,
                                        actual_type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_api_tests, const T>(read_only_acc_mode, dimensions,
                                                 actual_type_name);
  }
};
}  // namespace host_accessor_api_common
#endif  // SYCL_CTS_HOST_ACCESSOR_API_COMMON_H
