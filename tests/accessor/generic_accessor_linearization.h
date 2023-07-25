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
//  Provides common code for generic sycl::accessor correct linearization test
//  For multidimensional accessors the iterator linearizes the data according
//  to Section 3.11.1
//
*******************************************************************************/
#ifndef SYCL_CTS_GENERIC_ACCESSOR_LINEAR_COMMON_H
#define SYCL_CTS_GENERIC_ACCESSOR_LINEAR_COMMON_H
#include "accessor_common.h"

namespace generic_accessor_linearization {
using namespace accessor_tests_common;

template <typename T, typename AccessT, typename DimensionT, typename TargetT>
class run_linearization_tests {
  static constexpr sycl::access_mode AccessMode = AccessT::value;
  static constexpr int dims = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;
  using AccT = sycl::accessor<T, dims, AccessMode, Target>;

 public:
  void operator()(const std::string &type_name,
                  const std::string &access_mode_name,
                  const std::string &target_name) {
    auto queue = once_per_unit::get_queue();
    auto r = util::get_cts_object::range<dims>::get(1, 1, 1);

    SECTION(
        get_section_name<dims>(type_name, access_mode_name, target_name, "")) {
      check_linearization<accessor_type::generic_accessor, T, dims, AccessMode,
                          Target>();
    }
  }
};

using test_combinations =
    typename get_combinations<access_modes_pack, integer_pack<2, 3>,
                              targets_pack>::type;

template <typename T, typename ArgCombination>
class run_generic_linearization_for_type {
 public:
  void operator()(const std::string &type_name) {
    // Get the packs from the test combination type.
    using AccessModePack = std::tuple_element_t<0, ArgCombination>;
    using DimensionsPack = std::tuple_element_t<1, ArgCombination>;
    using TargetsPack = std::tuple_element_t<2, ArgCombination>;

    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = AccessModePack::generate_named();
    const auto dimensions = DimensionsPack::generate_unnamed();
    const auto targets = TargetsPack::generate_named();

    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_linearization_tests, T>(access_modes, dimensions,
                                                     targets, actual_type_name);

    // For covering const types
    actual_type_name = std::string("const ") + actual_type_name;
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<run_linearization_tests, const T>(
        read_only_acc_mode, dimensions, targets, actual_type_name);
  }
};
}  // namespace generic_accessor_linearization
#endif  // SYCL_CTS_GENERIC_ACCESSOR_LINEAR_COMMON_H
