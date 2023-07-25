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

//  Provides tests that the sycl::local_accessor can access the memory shared
//  among work-items.

#ifndef SYCL_CTS_LOCAL_ACCESSOR_ACCESS_AMONG_WORK_ITEMS_H
#define SYCL_CTS_LOCAL_ACCESSOR_ACCESS_AMONG_WORK_ITEMS_H
#include "accessor_common.h"

namespace local_accessor_access_among_work_items {
using namespace sycl_cts;
using namespace accessor_tests_common;

template <typename T, typename DimensionTypeT>
class kernel_local_accessor;

/**
 * @brief Provides a functor that provides verification that local_accessor can
 *        access the memory shared among work-items
 * @tparam T Current data type
 * @tparam DimensionTypeT Current current dimension size
 */
template <typename T, typename DimensionTypeT>
class run_test {
  static constexpr int Dimension = DimensionTypeT::value;

 public:
  /**
   * @brief Functor that provides verification that local_accessor can access
   *        the memory shared among work-items
   * @param type_name Current data type string representation
   */
  void operator()(const std::string& type_name) {
    auto queue = once_per_unit::get_queue();

    auto section_name = get_section_name<Dimension>(
        type_name,
        "Verify possibility to access the memory shared among work-items. "
        "[local_accessor]");
    SECTION(section_name) {
      T values_arr[2] = {T(1), T(2)};
      constexpr size_t range_size = 2;
      auto range = util::get_cts_object::range<Dimension>::get(
          range_size, range_size, range_size);
      const T valid_value = value_operations::init<T>(5);
      const T invalid_value = value_operations::init<T>(6);

      bool is_acc_val_equal_to_expected = false;
      {
        sycl::buffer<bool> val_is_equal_to_expected_buffer(
            &is_acc_val_equal_to_expected, sycl::range(1));
        queue.submit([&](sycl::handler& cgh) {
          auto val_is_equal_to_expected_acc =
              val_is_equal_to_expected_buffer
                  .template get_access<sycl::access_mode::write>(cgh);
          sycl::local_accessor<T, Dimension> acc(range, cgh);
          cgh.parallel_for<kernel_local_accessor<T, DimensionTypeT>>(
              sycl::nd_range(range, range), [=](sycl::nd_item<Dimension> item) {
                auto lid = item.get_local_id();
                auto zid = sycl::id<Dimension>();
                // Initialize local memory with invalid value
                acc[lid] = invalid_value;
                // Wait for work-items to finish initialization
                sycl::group_barrier(item.get_group());

                // Work-items with index greater than 0 writes to valid data to
                // the first element of the local data
                if (lid != zid) {
                  acc[sycl::id<Dimension>()] = valid_value;
                }

                // Wait for data store to finish
                sycl::group_barrier(item.get_group());
                // 0th work-item reports the result
                if (lid == zid) {
                  val_is_equal_to_expected_acc[0] =
                      value_operations::are_equal(acc[zid], valid_value);
                }
              });
        });
      }
      CHECK(is_acc_val_equal_to_expected);
    }
  }
};

using test_combinations = typename get_combinations<dimensions_pack>::type;

template <typename T, typename ArgCombination>
class run_local_accessor_access_among_work_items_tests {
 public:
  void operator()(const std::string& type_name) {
    // Get the packs from the test combination type.
    using DimensionsPack = std::tuple_element_t<0, ArgCombination>;

    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto dimensions = DimensionsPack::generate_unnamed();

    for_all_combinations<run_test, T>(dimensions, type_name);
  }
};
}  // namespace local_accessor_access_among_work_items

#endif  // SYCL_CTS_LOCAL_ACCESSOR_ACCESS_AMONG_WORK_ITEMS_H
