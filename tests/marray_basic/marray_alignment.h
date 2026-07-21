/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#ifndef SYCLCTS_TESTS_MARRAY_MARRAY_ALIGNMENT_H
#define SYCLCTS_TESTS_MARRAY_MARRAY_ALIGNMENT_H

#include "../common/common.h"
#include "../common/section_name_builder.h"
#include "marray_common.h"

namespace marray_alignment {

template <typename DataT, typename NumElementsT>
class run_marray_alignment_test {
  static constexpr std::size_t NumElements = NumElementsT::value;

  using marray_t = sycl::marray<DataT, NumElements>;
  using sarray_t = std::array<DataT, NumElements>;

 public:
  void operator()(const std::string&) {
    INFO("for number of elements \"" << NumElements << "\": ");

    // alignof
    CHECK((alignof(marray_t) == alignof(sarray_t)));

    // sizeof
    REQUIRE((sizeof(marray_t) == sizeof(sarray_t)));

    // memcmp
    marray_t ma;
    std::iota(ma.begin(), ma.end(), 1);
    sarray_t sa;
    std::iota(sa.begin(), sa.end(), 1);
    CHECK((std::memcmp(&ma, &sa, sizeof(marray_t)) == 0));
  }
};

template <typename DataT>
class check_marray_alignment_for_type {
 public:
  void operator()(const std::string& type_name) {
    INFO("for type \"" << type_name << "\": ");
    const auto num_elements = marray_common::get_num_elements();
    for_all_combinations<run_marray_alignment_test, DataT>(num_elements,
                                                           type_name);
  }
};

}  // namespace marray_alignment

#endif  // SYCLCTS_TESTS_MARRAY_MARRAY_ALIGNMENT_H
