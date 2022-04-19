/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::local_accessor constructors
//
*******************************************************************************/
#ifndef SYCL_CTS_LOCAL_ACCESSOR_H
#define SYCL_CTS_LOCAL_ACCESSOR_H
#include "catch2/catch_test_macros.hpp"

#include "accessor_common.h"

namespace local_accessor_constructors {
using namespace sycl_cts;
using namespace accessor_tests_common;

constexpr accessor_type AccType = accessor_type::local_accessor;

template <typename DataT, int Dimension>
void test_default_constructor(const std::string& type_name) {
  const auto section_name =
      get_section_name<Dimension>(type_name, "Default constructor");

  SECTION(section_name) {
    auto get_acc_functor = [] {
      return sycl::local_accessor<DataT, Dimension>();
    };
    if constexpr (std::is_const_v<DataT>) {
      check_def_constructor<AccType, DataT, Dimension, sycl::access_mode::read,
                            sycl::target::device>(get_acc_functor);
    } else {
      check_def_constructor<AccType, DataT, Dimension,
                            sycl::access_mode::read_write,
                            sycl::target::device>(get_acc_functor);
    }
  }
}

template <typename DataT>
void test_zero_dimension_buffer_constructor(const std::string& type_name) {
  const auto section_name =
      get_section_name<0>(type_name, "Zero dimension constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, 1>& data_buf,
                              sycl::handler& cgh) {
      return sycl::local_accessor<DataT, 0>(cgh);
    };
    if constexpr (std::is_const_v<DataT>) {
      check_zero_dim_constructor<AccType, DataT, sycl::access_mode::read,
                                 sycl::target::device>(get_acc_functor);
    } else {
      check_zero_dim_constructor<AccType, DataT, sycl::access_mode::read_write,
                                 sycl::target::device>(get_acc_functor);
    }
  }
}

template <typename DataT, int Dimension>
void test_common_constructors(const std::string& type_name) {
  auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
  auto offset = sycl::id<Dimension>();

  auto section_name =
      get_section_name<Dimension>(type_name, "From sycl::range constructor");

  SECTION(section_name) {
    auto get_acc_functor = [r](sycl::buffer<DataT, Dimension>& data_buf,
                               sycl::handler& cgh) {
      return sycl::local_accessor<DataT, Dimension>(r, cgh);
    };
    if constexpr (std::is_const_v<DataT>) {
      check_common_constructor<AccType, DataT, Dimension,
                               sycl::access_mode::read, sycl::target::device>(
          get_acc_functor);
    } else {
      check_common_constructor<AccType, DataT, Dimension,
                               sycl::access_mode::read_write,
                               sycl::target::device>(get_acc_functor);
    }
  }
}

template <typename T, typename DimensionT>
class run_tests_constructors {
  static constexpr int Dimension = DimensionT::value;

 public:
  void operator()(const std::string& type_name) {
    test_zero_dimension_buffer_constructor<T>(type_name);
    test_default_constructor<T, Dimension>(type_name);
    test_common_constructors<T, Dimension>(type_name);
  }
};

template <typename T>
class run_local_constructors_test {
 public:
  void operator()(const std::string& type_name) {
    // Type packs instances have to be const, otherwise for_all_combination will
    // not compile
    const auto dimensions = get_dimensions();

    for_all_combinations<run_tests_constructors, T>(dimensions, type_name);

    // For covering const types
    const auto const_type_name = std::string("const ") + type_name;
    for_all_combinations<run_tests_constructors, const T>(dimensions,
                                                          const_type_name);
  }
};
}  // namespace local_accessor_constructors
#endif  // SYCL_CTS_LOCAL_ACCESSOR_H
