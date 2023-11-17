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

#ifndef SYCLCTS_TESTS_MARRAY_MARRAY_CONSTRUCTOR_H
#define SYCLCTS_TESTS_MARRAY_MARRAY_CONSTRUCTOR_H

#include "../common/common.h"
#include "../common/section_name_builder.h"
#include "marray_common.h"

namespace marray_constructor {

template <typename DataT, typename NumElementsT>
class run_marray_constructor_test {
  static constexpr std::size_t NumElements = NumElementsT::value;

  static constexpr size_t num_test_cases = 7;

  static constexpr const char* check_names[num_test_cases] = {
      "default constructor",
      "scalar constructor",
      "variadic constructor (NumElements DataT instances)",
      "variadic constructor (one DataT instance, one marray instance)",
      "variadic constructor (one marray instance, one DataT instance)",
      "copy constructor",
      "copy constructor rval reference"};

  using marray_t = sycl::marray<DataT, NumElements>;

  template <typename IteratorT>
  static void run_checks(IteratorT results) {
    // default constructor
    {
      marray_t ma;
      *(results++) = value_operations::are_equal(ma, DataT{});
    }

    // scalar constructor
    {
      constexpr DataT value{1};
      constexpr marray_t ma{value};
      *(results++) = value_operations::are_equal(ma, value);
    }

    // variadic constructor
    {
      // NumElements DataT instances
      constexpr auto a = marray_common::iota_marray<DataT, NumElements, 1>();
      marray_t ma_inc;
      marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
      *(results++) = value_operations::are_equal(ma_inc, a);

      // only compiled when NumElements != 1
      if constexpr (NumElements != 1) {
        //  one DataT instance, one marray instance
        {
          constexpr sycl::marray<DataT, NumElements - 1> ma_const =
              marray_common::iota_marray<DataT, NumElements - 1, 2>();
          constexpr marray_t ma{1, ma_const};
          marray_t ma_inc;
          marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
          *(results++) = value_operations::are_equal(ma_inc, ma);
        }

        // one marray instance, one DataT instance
        {
          constexpr sycl::marray<DataT, NumElements - 1> ma_const =
              marray_common::iota_marray<DataT, NumElements - 1, 1>();
          constexpr marray_t ma{ma_const, DataT(NumElements)};
          marray_t ma_inc;
          marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
          *(results++) = value_operations::are_equal(ma_inc, ma);
        }
      } else {
        // Two checks were skipped.
        *(results++) = true;
        *(results++) = true;
      }
    }

    // copy constructor
    {
      constexpr DataT value{1};
      constexpr marray_t rhs{value};
      constexpr marray_t ma{rhs};
      *(results++) = value_operations::are_equal(ma, value);
    }

    // copy constructor rval reference
    {
      constexpr marray_t ma{
          marray_common::iota_marray<DataT, NumElements, 1>()};
      marray_t ma_inc;
      marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
      *(results++) = value_operations::are_equal(ma_inc, ma);
    }
  }

 public:
  void operator()(const std::string&) {
    INFO("for number of elements \"" << NumElements << "\": ");

    {
      INFO("validation on host");

      bool check_results[num_test_cases] = {false};
      run_checks(check_results);
      for (size_t i = 0; i < num_test_cases; ++i) {
        INFO(check_names[i]);
        CHECK(check_results[i]);
      }
    }

    {
      INFO("validation on device");

      auto queue = sycl_cts::util::get_cts_object::queue();
      bool check_results[num_test_cases] = {false};
      {
        sycl::buffer<bool, 1> check_results_buff{
            check_results, sycl::range<1>{num_test_cases}};

        queue
            .submit([&](sycl::handler& cgh) {
              sycl::accessor check_results_acc{check_results_buff, cgh,
                                               sycl::read_write};
              cgh.single_task([=]() { run_checks(check_results_acc.begin()); });
            })
            .wait_and_throw();
      }
      run_checks(check_results);
      for (size_t i = 0; i < num_test_cases; ++i) {
        INFO(check_names[i]);
        CHECK(check_results[i]);
      }
    }
  }
};

template <typename DataT>
class check_marray_constructor_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();
    for_all_combinations<run_marray_constructor_test, DataT>(num_elements,
                                                             type_name);
  }
};

}  // namespace marray_constructor

#endif  // SYCLCTS_TESTS_MARRAY_MARRAY_CONSTRUCTOR_H
