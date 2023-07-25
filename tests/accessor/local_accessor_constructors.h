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

/**
 * @brief Functor to assign value to local accessor
 * @tparam AccT accessor type
 * @tparam Dimension AccT dimension
 * @param src_acc Instance of a source accessor type
 */
template <typename AccT, int Dimension>
struct assign_value_to_accessor {
  auto operator()(const AccT& acc) const {
    AccT result(acc);
    if constexpr (0 != Dimension) {
      value_operations::assign(result[sycl::id<Dimension>()], expected_val);
    } else {
      typename AccT::value_type& ref = result;
      value_operations::assign(ref, expected_val);
    }
    return result;
  }
};

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
  using Acc = sycl::local_accessor<DataT, 0>;

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, 1>& data_buf,
                              sycl::handler& cgh) { return Acc(cgh); };
    if constexpr (std::is_const_v<DataT>) {
      check_zero_dim_constructor<AccType, DataT, sycl::access_mode::read,
                                 sycl::target::device>(get_acc_functor);
    } else {
      const assign_value_to_accessor<Acc, 0> modify_acc_functor;
      check_zero_dim_constructor<AccType, DataT, sycl::access_mode::read_write,
                                 sycl::target::device>(get_acc_functor,
                                                       modify_acc_functor);
    }
  }
}

template <typename DataT, int Dimension>
void test_common_constructors(const std::string& type_name) {
  constexpr int buf_dims = (0 == Dimension) ? 1 : Dimension;
  using Acc = sycl::local_accessor<DataT, Dimension>;

  auto section_name =
      get_section_name<Dimension>(type_name, "From sycl::range constructor");

  SECTION(section_name) {
    auto get_acc_functor = [](sycl::buffer<DataT, buf_dims>& data_buf,
                              sycl::handler& cgh) {
      if constexpr (0 != Dimension) {
        const auto r = util::get_cts_object::range<Dimension>::get(1, 1, 1);
        return sycl::local_accessor<DataT, Dimension>(r, cgh);
      } else {
        return sycl::local_accessor<DataT, Dimension>(cgh);
      }
    };
    if constexpr (std::is_const_v<DataT>) {
      check_common_constructor<AccType, DataT, Dimension,
                               sycl::access_mode::read, sycl::target::device>(
          get_acc_functor);
    } else {
      const assign_value_to_accessor<Acc, Dimension> modify_acc_functor;
      check_common_constructor<AccType, DataT, Dimension,
                               sycl::access_mode::read_write,
                               sycl::target::device>(get_acc_functor,
                                                     modify_acc_functor);
    }
  }
}

template <typename DataT, int Dimension>
void test_constructor_with_empty_property_list(const std::string& type_name) {
  constexpr int buf_dims = (0 == Dimension) ? 1 : Dimension;
  const auto r = util::get_cts_object::range<buf_dims>::get(1, 1, 1);
  const sycl::property_list prop_list;
  using Acc = sycl::local_accessor<DataT, Dimension>;

  auto section_name = get_section_name<Dimension>(
      type_name, "From constructor with empty property list");

  SECTION(section_name) {
    auto get_acc_functor = [&](sycl::buffer<DataT, buf_dims>& data_buf,
                               sycl::handler& cgh) {
      if constexpr (0 != Dimension) {
        auto acc = Acc(r, cgh, prop_list);
        return acc;
      } else {
        auto acc = Acc(cgh, prop_list);
        return acc;
      }
    };
    if constexpr (std::is_const_v<DataT>) {
      check_common_constructor<AccType, DataT, Dimension,
                               sycl::access_mode::read, sycl::target::device>(
          get_acc_functor);
    } else {
      const assign_value_to_accessor<Acc, Dimension> modify_acc_functor;
      check_common_constructor<AccType, DataT, Dimension,
                               sycl::access_mode::read_write,
                               sycl::target::device>(get_acc_functor,
                                                     modify_acc_functor);
    }
  }
}

template <typename T, typename DimensionT>
class run_tests_constructors {
  static constexpr int Dimension = DimensionT::value;

 public:
  void operator()(const std::string& type_name) {
    if constexpr (0 == Dimension) {
      test_zero_dimension_buffer_constructor<T>(type_name);
    }
    test_default_constructor<T, Dimension>(type_name);
    test_common_constructors<T, Dimension>(type_name);
    test_constructor_with_empty_property_list<T, Dimension>(type_name);
  }
};

using test_combinations = typename get_combinations<dimensions_pack>::type;

template <typename T, typename ArgCombination>
class run_local_constructors_test {
 public:
  void operator()(const std::string& type_name) {
    // Get the packs from the test combination type.
    using DimensionsPack = std::tuple_element_t<0, ArgCombination>;

    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto dimensions = DimensionsPack::generate_unnamed();

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or for_all_device_copyable_std_containers.
    // This will wrap string with type T to string with container<T> if T is
    // an array or other kind of container.
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_tests_constructors, T>(dimensions,
                                                    actual_type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    for_all_combinations<run_tests_constructors, const T>(dimensions,
                                                          actual_type_name);
  }
};
}  // namespace local_accessor_constructors
#endif  // SYCL_CTS_LOCAL_ACCESSOR_H
