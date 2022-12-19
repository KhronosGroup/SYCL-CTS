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

#ifndef SYCLCTS_TESTS_MARRAY_MARRAY_MEMBERS_H
#define SYCLCTS_TESTS_MARRAY_MARRAY_MEMBERS_H

#include "../common/common.h"

#include "../common/section_name_builder.h"
#include "marray_common.h"

namespace marray_members {

template <typename DataT, typename NumElementsT>
class run_marray_members_test {
  static constexpr std::size_t NumElements = NumElementsT::value;

  using marray_t = sycl::marray<DataT, NumElements>;

 private:
  template <size_t num_elements,
            std::enable_if_t<num_elements != 1, bool> = true>
  void check_conversion() {}

  template <size_t num_elements,
            std::enable_if_t<num_elements == 1, bool> = true>
  void check_conversion() {
    marray_t ma_inc;
    std::iota(ma_inc.begin(), ma_inc.end(), 1);
    DataT t{ma_inc};
    CHECK((t == DataT{1}));
  }

 public:
  void operator()(const std::string&) {
    INFO("for number of elements \"" << NumElements << "\": ");

    // implicit conversion
    {
      // only compiled when NumElements == 1
      check_conversion<NumElements>();
    }

    // size()
    {
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);
      CHECK(noexcept(ma_inc.size()));
      CHECK((ma_inc.size() == NumElements));
    }

    // operator[]
    {
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);
      CHECK((ma_inc[0] == DataT{1}));
      ma_inc[0] = DataT{0};
      CHECK((ma_inc[0] == DataT{0}));
    }

    // const operator[]
    {
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);
      const marray_t ma_const{ma_inc};
      CHECK((ma_const[0] == DataT{1}));
    }

    // operator=(marray)
    {
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);
      const marray_t ma_const{ma_inc};

      marray_t ma_tmp{DataT{0}};
      ma_tmp = ma_const;
      CHECK(value_operations::are_equal(ma_tmp, ma_const));
    }

    // operator=(T)
    {
      marray_t ma_tmp{DataT{0}};
      ma_tmp = DataT{1};
      CHECK(value_operations::are_equal(ma_tmp, marray_t(DataT(1))));
    }

    // iterator
    {
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);

      auto it_ma = ma_inc.begin();
      auto it_ma_tmp = it_ma;
      it_ma++;
      if (NumElements > 1) {
        CHECK((*it_ma == DataT(2)));
      }
      it_ma--;
      CHECK((*it_ma == DataT(1)));
      CHECK((it_ma == it_ma_tmp));
    }

    // const iterator
    {
      marray_t ma_inc;
      std::iota(ma_inc.begin(), ma_inc.end(), 1);
      const marray_t ma_const = ma_inc;
      auto it_ma = ma_const.begin();
      auto it_ma_tmp = it_ma;
      it_ma++;
      if (NumElements > 1) {
        CHECK((*it_ma == DataT(2)));
      }
      it_ma--;
      CHECK((*it_ma == DataT(1)));
      CHECK((it_ma == it_ma_tmp));
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
