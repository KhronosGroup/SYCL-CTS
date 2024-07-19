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

#ifndef SYCLCTS_TESTS_MARRAY_MARRAY_MEMBERS_H
#define SYCLCTS_TESTS_MARRAY_MARRAY_MEMBERS_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "marray_common.h"

namespace marray_members {

template <typename DataT, typename NumElementsT>
class run_marray_members_test {
  static constexpr std::size_t NumElements = NumElementsT::value;

  static constexpr size_t num_test_cases = 19;

  static constexpr const char* check_names[num_test_cases] = {
      "implicit conversion",
      "size() (noexcept)",
      "size() (result)",
      "operator[] (before write)",
      "operator[] (after write)",
      "const operator[]",
      "operator=(marray)",
      "operator=(T)",
      "iterator (increment)",
      "iterator (decrement)",
      "iterator (equivalence)",
      "const iterator (increment)",
      "const iterator (decrement)",
      "const iterator (equivalence)",
      "value_type",
      "reference",
      "const_reference",
      "iterator",
      "const_iterator",
  };

  using marray_t = sycl::marray<DataT, NumElements>;

  template <typename IteratorT>
  static void run_checks(IteratorT results) {
    // implicit conversion
    {
      // only compiled when NumElements == 1
      if constexpr (NumElements == 1) {
        marray_t ma_inc;
        marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
        DataT t{ma_inc};
        *(results++) = (t == DataT{1});
      } else {
        // One check were skipped.
        *(results++) = true;
      }
    }

    // size()
    {
      marray_t ma_inc;
      marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
      *(results++) = noexcept(ma_inc.size());
      *(results++) = (ma_inc.size() == NumElements);
    }

    // operator[]
    {
      marray_t ma_inc;
      marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
      *(results++) = (ma_inc[0] == DataT{1});
      ma_inc[0] = DataT{0};
      *(results++) = (ma_inc[0] == DataT{0});
    }

    // const operator[]
    {
      marray_t ma_inc;
      marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
      const marray_t ma_const{ma_inc};
      *(results++) = (ma_const[0] == DataT{1});
    }

    // operator=(marray)
    {
      marray_t ma_inc;
      marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
      const marray_t ma_const{ma_inc};

      marray_t ma_tmp{DataT{0}};
      ma_tmp = ma_const;
      *(results++) = value_operations::are_equal(ma_tmp, ma_const);
    }

    // operator=(T)
    {
      marray_t ma_tmp{DataT{0}};
      ma_tmp = DataT{1};
      *(results++) = value_operations::are_equal(ma_tmp, marray_t(DataT(1)));
    }

    // iterator
    {
      marray_t ma_inc;
      marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);

      auto it_ma = ma_inc.begin();
      auto it_ma_tmp = it_ma;
      it_ma++;
      if (NumElements > 1) {
        *(results++) = (*it_ma == DataT(2));
      } else {
        *(results++) = true;
      }
      it_ma--;
      *(results++) = (*it_ma == DataT(1));
      *(results++) = (it_ma == it_ma_tmp);
    }

    // const iterator
    {
      marray_t ma_inc;
      marray_common::iota(ma_inc.begin(), ma_inc.end(), 1);
      const marray_t ma_const = ma_inc;
      auto it_ma = ma_const.begin();
      auto it_ma_tmp = it_ma;
      it_ma++;
      if (NumElements > 1) {
        *(results++) = (*it_ma == DataT(2));
      } else {
        *(results++) = true;
      }
      it_ma--;
      *(results++) = (*it_ma == DataT(1));
      *(results++) = (it_ma == it_ma_tmp);
    }

    // member types
    {
      *(results++) = std::is_same_v<typename marray_t::value_type, DataT>;
      *(results++) = std::is_same_v<typename marray_t::reference, DataT&>;
      *(results++) =
          std::is_same_v<typename marray_t::const_reference, const DataT&>;
      *(results++) = std::is_same_v<typename marray_t::iterator, DataT*>;
      *(results++) =
          std::is_same_v<typename marray_t::const_iterator, const DataT*>;
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
class check_marray_members_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("for type \"" << type_name << "\": ");

    const auto num_elements = marray_common::get_num_elements();
    for_all_combinations<run_marray_members_test, DataT>(num_elements,
                                                         type_name);
  }
};

}  // namespace marray_members

#endif  // SYCLCTS_TESTS_MARRAY_MARRAY_MEMBERS_H
