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
//  Provides tests for generic sycl::accessor placeholder zero-length buffer
//  constructor
//
*******************************************************************************/
#ifndef SYCL_CTS_GENERIC_ACCESSOR_PLACEHOLDER_ZERO_LENGTH_BUFFER_CONSTRUCTOR_H
#define SYCL_CTS_GENERIC_ACCESSOR_PLACEHOLDER_ZERO_LENGTH_BUFFER_CONSTRUCTOR_H
#include "../accessor_basic/accessor_common.h"

#include "catch2/catch_test_macros.hpp"

namespace generic_accessor_placeholder_zero_length_buffer_constructor {
using namespace sycl_cts;
using namespace accessor_tests_common;

constexpr accessor_type AccType = accessor_type::generic_accessor;

template <typename DataT, int Dimension, sycl::access_mode AccessMode,
          sycl::target Target>
void test_placeholder_zero_length_buffer_constructor(
    const std::string& type_name, const std::string& access_mode_name,
    const std::string& target_name) {
  auto section_name = get_section_name<Dimension>(
      type_name, access_mode_name, target_name,
      "From zero-length buffer placeholder constructor");

  SECTION(section_name) {
    constexpr int dim_buf = (0 == Dimension) ? 1 : Dimension;
    auto get_acc_functor = [](sycl::buffer<DataT, dim_buf>& data_buf) {
      return sycl::accessor<DataT, Dimension, AccessMode, Target>(data_buf);
    };
    check_zero_length_buffer_placeholder_constructor<AccType, DataT, Dimension,
                                                     AccessMode, Target>(
        get_acc_functor);
  }
}

template <typename T, typename AccessT, typename TargetT, typename DimensionT>
class run_tests_placeholder_zero_length_buffer_constructor {
  static constexpr sycl::access_mode AccessMode = AccessT::value;
  static constexpr int Dimension = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    test_placeholder_zero_length_buffer_constructor<T, Dimension, AccessMode,
                                                    Target>(
        type_name, access_mode_name, target_name);
  }
};

using test_combinations =
    typename get_combinations<access_modes_pack, all_dimensions_pack,
                              targets_pack>::type;

template <typename T, typename ArgCombination>
class run_generic_placeholder_zero_length_buffer_constructor {
 public:
  void operator()(const std::string& type_name) {
    // Get the packs from the test combination type.
    using AccessModePack = std::tuple_element_t<0, ArgCombination>;
    using DimensionsPack = std::tuple_element_t<1, ArgCombination>;
    using TargetsPack = std::tuple_element_t<2, ArgCombination>;

    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = AccessModePack::generate_named();
    const auto dimensions = DimensionsPack::generate_unnamed();
    const auto targets = TargetsPack::generate_named();

    // To handle cases when class was called from functions
    // like for_all_types_vectors_marray or
    // for_all_device_copyable_std_containers. This will wrap string with type T
    // to string with container<T> if T is an array or other kind of container.
    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_tests_placeholder_zero_length_buffer_constructor,
                         T>(access_modes, targets, dimensions, type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_tests_placeholder_zero_length_buffer_constructor,
                         const T>(read_only_acc_mode, targets, dimensions,
                                  actual_type_name);
  }
};
}  // namespace generic_accessor_placeholder_zero_length_buffer_constructor
#endif  // SYCL_CTS_GENERIC_ACCESSOR_PLACEHOLDER_ZERO_LENGTH_BUFFER_CONSTRUCTOR_H
