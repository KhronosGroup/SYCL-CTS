/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for sycl::host_accessor api tests
//
*******************************************************************************/
#ifndef SYCL_CTS_HOST_ACCESSOR_API_COMMON_H
#define SYCL_CTS_HOST_ACCESSOR_API_COMMON_H
#include "../accessor_basic/accessor_common.h"
#include <cmath>

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

template <typename T, typename AccessT, typename DimensionT>
class run_api_tests {
  static constexpr sycl::access_mode AccessMode = AccessT::value;
  static constexpr int dims = DimensionT::value;
  using AccT = sycl::host_accessor<T, dims, AccessMode>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &access_mode_name) {
    constexpr int buf_dims = (0 == dims) ? 1 : dims;
    auto r = util::get_cts_object::range<buf_dims>::get(1, 1, 1);

    SECTION(get_section_name<dims>(type_name, access_mode_name,
                                   "Check host_accessor alias types")) {
      test_accessor_types_common<T, AccT, AccessMode>();
    }

    SECTION(get_section_name<dims>(type_name, access_mode_name,
                                   "Check api for empty host_accessor")) {
      AccT acc;
      test_accessor_methods_common(acc, 0 /* expected_byte_size*/,
                                   0 /*expected_size*/);
      if constexpr (0 < dims) {
        test_accessor_range_methods(
            acc,
            util::get_cts_object::range<dims>::get(0, 0, 0) /*expected_range*/,
            sycl::id<dims>() /*&expected_offset)*/);
      }
      test_begin_end_host(acc);
    }

    SECTION(get_section_name<dims>(type_name, access_mode_name,
                                   "Check api for host_accessor")) {
      T data = value_operations::init<T>(expected_val);
      bool res = false;
      {
        sycl::buffer<T, buf_dims> data_buf(&data, r);
        AccT acc{data_buf};

        test_accessor_ptr(acc, expected_val);

        test_accessor_methods_common(acc, sizeof(T) /* expected_byte_size*/,
                                     1 /*expected_size*/);
        test_begin_end_host(acc, expected_val, expected_val, false);
        if constexpr (0 < dims) {
          test_accessor_range_methods(acc,
                                      util::get_cts_object::range<dims>::get(
                                          1, 1, 1) /*expected_range*/,
                                      sycl::id<dims>() /*&expected_offset)*/);
          auto &acc_ref1 = acc[sycl::id<dims>()];
          auto &acc_ref2 = get_subscript_overload<T, AccT, dims>(acc, 0);
          CHECK(value_operations::are_equal(acc_ref1, expected_val));
          CHECK(value_operations::are_equal(acc_ref2, expected_val));
          STATIC_CHECK(
              std::is_same_v<decltype(acc_ref1), typename AccT::reference>);
          STATIC_CHECK(
              std::is_same_v<decltype(acc_ref2), typename AccT::reference>);
          if constexpr (AccessMode != sycl::access_mode::read) {
            value_operations::assign(acc_ref1, changed_val);
            CHECK(value_operations::are_equal(acc_ref2, changed_val));
          }
        } else {
          T some_data = value_operations::init<T>(expected_val);
          typename AccT::reference dref = acc;
          CHECK(value_operations::are_equal(some_data, dref));
          if constexpr (AccessMode != sycl::access_mode::read) {
            typename AccT::value_type v_data =
                value_operations::init<typename AccT::value_type>(changed_val);
            // check method const AccT::operator=(const T& data) const
            acc = v_data;
            CHECK(value_operations::are_equal(dref, v_data));

            // check method const AccT::operator=(T&& data) const
            acc =
                value_operations::init<typename AccT::value_type>(changed_val);
            CHECK(value_operations::are_equal(dref, v_data));
          }
        }
      }
      if constexpr (AccessMode != sycl::access_mode::read)
        CHECK(value_operations::are_equal(data, changed_val));
    }
    if constexpr (0 < dims) {
      SECTION(get_section_name<dims>(
          type_name, access_mode_name,
          "Check api for ranged host_accessor with offset")) {
        // Partially duplicates tests/accessor/generic_accessor_api_common.h
        // The maximum value of the linear_index variable should not be more
        // than CHAR_MAX (usually 127 for schar). Otherwise the test fails here
        // with the char type:
        // CHECK(value_operations::are_equal(acc_ref1, linear_index));
        // As acc_ref1 contains corrupted by the overflow value.
        constexpr size_t acc_range_size = 2;
        constexpr size_t buff_range_size = 4;
        constexpr size_t buff_size = (dims == 3)   ? 4 * 4 * 4
                                     : (dims == 2) ? 4 * 4
                                                   : 4;
        constexpr size_t offset = 2;
        constexpr size_t index = 1;
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
        auto offset_start = linearize(buff_range, offset_id);
        std::remove_const_t<T> data[buff_size];
        for (size_t i = 0; i < buff_size; i++) {
          data[i] = value_operations::init<T>(i);
        }
        auto acc_first_value = value_operations::init<T>(offset_start);
        bool res = false;
        {
          sycl::buffer<T, dims> data_buf(data, buff_range);
          AccT acc(data_buf, acc_range, offset_id);
          test_accessor_methods_common(
              acc, sizeof(T) * acc_range.size() /* expected_byte_size*/,
              acc_range.size() /*expected_size*/);
          test_accessor_range_methods(acc, acc_range /*expected_range*/,
                                      offset_id /*&expected_offset)*/);

          test_accessor_ptr(acc, T());
          test_begin_end_host(acc, acc_first_value,
                              value_operations::init<T>(buff_size - 1), false);
          auto &acc_ref1 = get_subscript_overload<T, AccT, dims>(acc, index);
          auto &acc_ref2 = acc[sycl::id<dims>()];
          CHECK(value_operations::are_equal(acc_ref1, linear_index));
          CHECK(value_operations::are_equal(acc_ref2, acc_first_value));
          if constexpr (AccessMode != sycl::access_mode::read) {
            value_operations::assign(acc_ref1, changed_val);
            value_operations::assign(acc_ref2, expected_val);
          }
        }
        if constexpr (AccessMode != sycl::access_mode::read) {
          CHECK(value_operations::are_equal(data[linear_index], changed_val));
          CHECK(value_operations::are_equal(data[offset_start], expected_val));
        }
      }
    }
    SECTION(get_section_name<dims>(type_name, access_mode_name,
                                   "Check swap for host_accessor")) {
      T data1 = value_operations::init<T>(expected_val);
      T data2 = value_operations::init<T>(changed_val);
      {
        sycl::buffer<T, buf_dims> data_buf1(&data1, r);
        sycl::buffer<T, buf_dims> data_buf2(&data2, r);
        AccT acc1(data_buf1);
        AccT acc2(data_buf2);
        acc1.swap(acc2);
        typename AccT::reference acc_ref1 = get_accessor_reference<dims>(acc1);
        typename AccT::reference acc_ref2 = get_accessor_reference<dims>(acc2);
        CHECK(value_operations::are_equal(acc_ref1, changed_val));
        CHECK(value_operations::are_equal(acc_ref2, expected_val));
        if constexpr (AccessMode != sycl::access_mode::read) {
          value_operations::assign(acc_ref1, expected_val);
          value_operations::assign(acc_ref2, changed_val);
        }
      }
      if constexpr (AccessMode != sycl::access_mode::read) {
        CHECK(value_operations::are_equal(data1, changed_val));
        CHECK(value_operations::are_equal(data2, expected_val));
      } else {
        CHECK(value_operations::are_equal(data1, expected_val));
        CHECK(value_operations::are_equal(data2, changed_val));
      }
    }
  }
};

using test_combinations =
    typename get_combinations<access_modes_pack, all_dimensions_pack>::type;

template <typename T, typename ArgCombination>
class run_host_accessor_api_for_type {
 public:
  void operator()(const std::string &type_name) {
    // Get the packs from the test combination type.
    using AccessModePack = std::tuple_element_t<0, ArgCombination>;
    using DimensionsPack = std::tuple_element_t<1, ArgCombination>;

    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = AccessModePack::generate_named();
    const auto dimensions = DimensionsPack::generate_unnamed();

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or
    // for_all_device_copyable_std_containers. This will wrap string with type T
    // to string with container<T> if T is an array or other kind of container.
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_api_tests, T>(access_modes, dimensions,
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
