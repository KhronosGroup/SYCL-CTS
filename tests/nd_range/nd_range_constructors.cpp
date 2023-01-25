/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#include "../common/common.h"

#define TEST_NAME nd_range_constructors

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/**
 * @brief Constructs a default nd_range
 * @tparam dim Number of dimensions of the nd_range
 * @return nd_range<dim> object with default values
 */
template <int dim>
inline sycl::nd_range<dim> get_default_nd_range() {
  const auto range = util::get_cts_object::range<dim>::get(1, 1, 1);
  return sycl::nd_range<dim>(range, range);
}

template <int dim>
void test_nd_range_constructors(util::logger &log, sycl::range<dim> gs,
                                sycl::range<dim> ls, sycl::id<dim> offset) {
  sycl::nd_range<dim> no_offset(gs, ls);
  sycl::nd_range<dim> with_offset(gs, ls, offset);

  {  // Copy assignment, no offset
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = no_offset;

    for (int i = 0; i < dim; i++) {
      CHECK_VALUE(log, defaultRange.get_global_range()[i], gs[i], i);
      CHECK_VALUE(log, defaultRange.get_local_range()[i], ls[i], i);
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
      CHECK_VALUE(log, defaultRange.get_offset()[i], (size_t)0, i);
#endif
      CHECK_VALUE(log, defaultRange.get_group_range()[i], gs[i] / ls[i], i);
    }
  }
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  {  // Copy assignment, with offset
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = with_offset;
    for (int i = 0; i < dim; i++) {
      CHECK_VALUE(log, defaultRange.get_global_range()[i], gs[i], i);
      CHECK_VALUE(log, defaultRange.get_local_range()[i], ls[i], i);
      CHECK_VALUE(log, defaultRange.get_offset()[i], offset[i], i);
      CHECK_VALUE(log, defaultRange.get_group_range()[i], gs[i] / ls[i], i);
    }
  }
#endif
  {  // Move assignment, no offset
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = std::move(no_offset);
    for (int i = 0; i < dim; i++) {
      CHECK_VALUE(log, defaultRange.get_global_range()[i], gs[i], i);
      CHECK_VALUE(log, defaultRange.get_local_range()[i], ls[i], i);
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
      CHECK_VALUE(log, defaultRange.get_offset()[i], (size_t)0, i);
#endif
      CHECK_VALUE(log, defaultRange.get_group_range()[i], gs[i] / ls[i], i);
    }
  }
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  {  // Move assignment, with offset
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = std::move(with_offset);
    for (int i = 0; i < dim; i++) {
      CHECK_VALUE(log, with_offset.get_global_range()[i], gs[i], i);
      CHECK_VALUE(log, with_offset.get_local_range()[i], ls[i], i);
      CHECK_VALUE(log, with_offset.get_offset()[i], offset[i], i);
      CHECK_VALUE(log, with_offset.get_group_range()[i], gs[i] / ls[i], i);
    }
  }
#endif
}

/** test sycl::nd_range initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    constexpr size_t sizes[] = {16, 32, 64};

    {
      // global size to be set to the size
      sycl::range<1> gs_1d(sizes[0]);
      // local size to be set to 1/4 of the sizes
      sycl::range<1> ls_1d(sizes[0] / 4u);
      // offset to be set to 1/8 of the sizes
      sycl::id<1> offset_1d(sizes[0] / 8u);
      test_nd_range_constructors(log, gs_1d, ls_1d, offset_1d);

      // global size to be set to the size
      sycl::range<2> gs_2d(sizes[0], sizes[1]);
      // local size to be set to 1/4 of the sizes
      sycl::range<2> ls_2d(sizes[0] / 4u, sizes[1] / 4u);
      // offset to be set to 1/8 of the sizes
      sycl::range<2> range_2d(sizes[0] / 8u, sizes[1] / 8u);
      sycl::id<2> offset_2d(range_2d);
      test_nd_range_constructors(log, gs_2d, ls_2d, offset_2d);

      // global size to be set to the size
      sycl::range<3> gs_3d(sizes[0], sizes[1], sizes[2]);
      // local size to be set to 1/4 of the sizes
      sycl::range<3> ls_3d(sizes[0] / 4, sizes[1] / 4, sizes[2] / 4);
      // offset to be set to 1/8 of the sizes
      sycl::range<3> range_3d(sizes[0] / 8u, sizes[1] / 8u, sizes[2] / 8u);
      sycl::id<3> offset_3d(range_3d);
      test_nd_range_constructors(log, gs_3d, ls_3d, offset_3d);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
