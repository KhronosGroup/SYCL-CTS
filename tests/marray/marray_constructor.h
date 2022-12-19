/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
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

  using marray_t = sycl::marray<DataT, NumElements>;

 private:
  template <size_t num_elements = NumElements,
            std::enable_if_t<num_elements == 1, bool> = true>
  void check_constexpr_single_element() {}

  template <size_t num_elements = NumElements,
            std::enable_if_t<num_elements != 1, bool> = true>
  void check_constexpr_single_element() {
    // cannot construct a constexpr instance using another constexpr instance
#if !(defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP) || \
      defined(SYCL_CTS_COMPILING_WITH_DPCPP))
    //  one DataT instance, one marray instance
    {
      constexpr sycl::marray<DataT, num_elements - 1> ma_const =
          marray_common::ctor<DataT, num_elements - 1, 2>::value;
      constexpr marray_t ma{1, ma_const};
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);
      CHECK(value_operations::are_equal(ma_inc, ma));
    }

    // one marray instance, one DataT instance
    {
      constexpr sycl::marray<DataT, num_elements - 1> ma_const =
          marray_common::ctor<DataT, num_elements - 1, 1>::value;
      constexpr marray_t ma{ma_const, DataT(num_elements)};
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);
      CHECK(value_operations::are_equal(ma_inc, ma));
    }
#else
    WARN(
        "ComputeCPP and DPCPP do not support constexpr constructors that use"
        "other constexpr instances. Skipping the test case.");
#endif
  }

 public:
  void operator()(const std::string&) {
    INFO("for number of elements \"" << NumElements << "\": ");

    // default constructor
    {
      marray_t ma;
      CHECK(value_operations::are_equal(ma, DataT{}));
    }

    // scalar constructor
    {
      constexpr DataT value{1};
      constexpr marray_t ma{value};
      CHECK(value_operations::are_equal(ma, value));
    }

    // variadic constructor
    {
      // NumElements DataT instances
      constexpr marray_t a = marray_common::ctor<DataT, NumElements, 1>::value;
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);
      CHECK(value_operations::are_equal(ma_inc, a));

      // only compiled when NumElements != 1
      check_constexpr_single_element();
    }

    // copy constructor
    {
      constexpr DataT value{1};
      constexpr marray_t rhs{value};
      constexpr marray_t ma{rhs};
      CHECK(value_operations::are_equal(ma, value));
    }

    // copy constructor rval reference
    {
      constexpr marray_t ma{marray_common::ctor<DataT, NumElements, 1>::value};
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);
      CHECK(value_operations::are_equal(ma_inc, ma));
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
