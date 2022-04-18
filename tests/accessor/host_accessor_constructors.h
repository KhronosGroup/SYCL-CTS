/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::host_accessor constructors
//
*******************************************************************************/
#ifndef SYCL_CTS_HOST_ACCESSOR_CONSTRUCTORS_H
#define SYCL_CTS_HOST_ACCESSOR_CONSTRUCTORS_H
#include "catch2/catch_test_macros.hpp"

#include "accessor_common.h"

namespace host_accessor_constructors {
using namespace sycl_cts;
using namespace accessor_tests_common;

constexpr accessor_type AccType = accessor_type::host_accessor;

template <typename DataT, int Dimension, sycl::access_mode AccessMode>
void test_default_constructor(const std::string& access_mode_name,
                              const std::string& type_name) {
  const auto section_name = get_section_name<Dimension>(
      type_name, access_mode_name, "Default constructor");

  SECTION(section_name) {
    auto get_acc_functor = []() {
      return sycl::host_accessor<DataT, Dimension, AccessMode>();
    };
    check_def_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }
}

template <typename DataT, sycl::access_mode AccessMode>
void test_zero_dimension_buffer_constructor(const std::string& access_mode_name,
                                            const std::string& type_name) {
  const auto section_name = get_section_name<0>(type_name, access_mode_name,
                                                "Zero dimension constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, 1> data_buf) {
      return sycl::host_accessor<DataT, 0, AccessMode>(data_buf);
    };
    check_zero_dim_constructor<AccType, DataT, AccessMode>(get_acc_functor);
  }
}

template <typename DataT, int Dimension, sycl::access_mode AccessMode>
void test_common_buffer_constructors(const std::string& access_mode_name,
                                     const std::string& type_name) {
  auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  auto offset = sycl::id<Dimension>();

  auto section_name = get_section_name<Dimension>(type_name, access_mode_name,
                                                  "From buffer constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name, "From buffer and range constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf, r);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name, "From buffer,range and offset constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r,
                            offset](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf, r,
                                                               offset);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }
}

template <typename DataT, int Dimension, sycl::access_mode AccessMode>
void test_common_buffer_constructors_tag_t_deduction(
    const std::string& access_mode_name, const std::string& type_name) {
  auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  auto offset = sycl::id<Dimension>();
  auto tagT = get_tag<AccessMode>();

  auto section_name = get_section_name<Dimension>(
      type_name, access_mode_name, "TagT deduction from buffer constructor");

  SECTION(section_name) {
    auto get_acc_functor = [tagT](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor<DataT, Dimension, AccessMode>(data_buf, tagT);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name = get_section_name<Dimension>(
      type_name, access_mode_name,
      "TagT deduction from buffer and range constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r, tagT](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor(data_buf, r, tagT);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }

  section_name =
      get_section_name<Dimension>(type_name, access_mode_name,
                                  "TagT deduction from buffer,range and "
                                  "offset constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r, offset,
                            tagT](sycl::buffer<DataT, Dimension> data_buf) {
      return sycl::host_accessor(data_buf, r, offset, tagT);
    };
    check_common_constructor<AccType, DataT, Dimension, AccessMode>(
        get_acc_functor);
  }
}

template <typename T, typename AccessModeT, typename DimensionT>
class run_tests_constructors {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name) {
    test_zero_dimension_buffer_constructor<T, AccessMode>(access_mode_name,
                                                          type_name);
    test_default_constructor<T, Dimension, AccessMode>(access_mode_name,
                                                       type_name);
    test_common_buffer_constructors<T, Dimension, AccessMode>(access_mode_name,
                                                              type_name);
    test_common_buffer_constructors_tag_t_deduction<T, Dimension, AccessMode>(
        access_mode_name, type_name);
  }
};

template <typename T>
class run_host_constructors_test {
 public:
  void operator()(const std::string& type_name) {
    // Type packs instances have to be const, otherwise for_all_combination will
    // not compile
    const auto access_modes = get_access_modes();
    const auto dimensions = get_dimensions();

    for_all_combinations<run_tests_constructors, T>(access_modes, dimensions,
                                                    type_name);

    // For covering const types
    const auto const_type_name = std::string("const ") + type_name;
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_tests_constructors, T>(
        read_only_acc_mode, dimensions, const_type_name);
  }
};
}  // namespace host_accessor_constructors

#endif  // SYCL_CTS_HOST_ACCESSOR_CONSTRUCTORS_H
