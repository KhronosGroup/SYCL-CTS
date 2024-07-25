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
//  Provides common code for sycl::local_accessor correct linearization test
//  For multidimensional accessors the iterator linearizes the data according
//  to Section 3.11.1
//
*******************************************************************************/
#ifndef SYCL_CTS_LOCAL_ACCESSOR_LINEAR_COMMON_H
#define SYCL_CTS_LOCAL_ACCESSOR_LINEAR_COMMON_H
#include "../accessor_basic/accessor_common.h"

namespace local_accessor_linearization {
using namespace accessor_tests_common;

template <typename T, typename DimensionT>
class kernel_linearization;

template <typename T, typename DimensionT>
class run_linearization_tests {
  static constexpr int dims = DimensionT::value;
  using AccT = sycl::local_accessor<T, dims>;

 public:
  void operator()(const std::string &type_name) {
    auto queue = once_per_unit::get_queue();
    auto r = util::get_cts_object::range<dims>::get(1, 1, 1);

    SECTION(get_section_name<dims>(type_name, "")) {
      constexpr size_t local_range_size = 2;
      constexpr size_t range_size = 8;

      auto range = util::get_cts_object::range<dims>::get(
          range_size, range_size, range_size);
      auto local_range = util::get_cts_object::range<dims>::get(
          local_range_size, local_range_size, local_range_size);
      sycl::nd_range<dims> nd_range(range, local_range);

      bool res = true;
      {
        sycl::buffer res_buf(&res, sycl::range(1));
        queue
            .submit([&](sycl::handler &cgh) {
              AccT acc(local_range, cgh);
              sycl::accessor res_acc(res_buf, cgh);
              cgh.parallel_for<kernel_linearization<T, DimensionT>>(
                  nd_range, [=](sycl::nd_item<dims> item) {
                    acc[item.get_local_id()] =
                        value_operations::init<T>(item.get_global_linear_id());
                    sycl::group_barrier(item.get_group());
                    sycl::id<dims> id{};
                    for (auto &elem : acc) {
                      res_acc[0] &= value_operations::are_equal(elem, acc[id]);
                      id = next_id_linearly(id, range_size);
                    }
                  });
            })
            .wait_and_throw();
      }
      CHECK(res);
    }
  }
};

using test_combinations = typename get_combinations<integer_pack<2, 3>>::type;

template <typename T, typename ArgCombination>
class run_local_linearization_for_type {
 public:
  void operator()(const std::string &type_name) {
    // Get the packs from the test combination type.
    using DimensionsPack = std::tuple_element_t<0, ArgCombination>;

    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto dimensions = DimensionsPack::generate_unnamed();

    auto actual_type_name = type_name_string<T>::get(type_name);

    for_all_combinations<run_linearization_tests, T>(dimensions,
                                                     actual_type_name);
  }
};
}  // namespace local_accessor_linearization
#endif  // SYCL_CTS_LOCAL_ACCESSOR_LINEAR_COMMON_H
