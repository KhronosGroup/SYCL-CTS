/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
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

#include "../common/common.h"

#define TEST_NAME nd_range_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

static const size_t sizes[] = {16, 32, 64};

template <int dim>
void test_nd_range(util::logger &log, sycl::range<dim> gs,
                   sycl::range<dim> ls, sycl::id<dim> offset) {
  for (int i = 0; i < dim; i++) {
    sycl::nd_range<dim> no_offset(gs, ls);
    CHECK_TYPE(log, no_offset.get_global_range()[i], sizes[i]);
    CHECK_VALUE(log, no_offset.get_global_range()[i], sizes[i], i);
    CHECK_TYPE(log, no_offset.get_local_range()[i], sizes[i] / 4);
    CHECK_VALUE(log, no_offset.get_local_range()[i], sizes[i] / 4, i);

    // TODO: mark this check as testing deprecated functionality
    CHECK_TYPE(log, no_offset.get_offset()[i], (size_t)0);
    CHECK_VALUE(log, no_offset.get_offset()[i], (size_t)0, i);

    CHECK_TYPE(log, no_offset.get_group_range()[i], sizes[i] / (sizes[i] / 4));
    CHECK_VALUE(log, no_offset.get_group_range()[i], sizes[i] / (sizes[i] / 4), i);

    sycl::nd_range<dim> deep_copy(no_offset);

    CHECK_TYPE(log, deep_copy.get_global_range()[i], sizes[i]);
    CHECK_VALUE(log, deep_copy.get_global_range()[i], sizes[i], i);
    CHECK_TYPE(log, deep_copy.get_local_range()[i], sizes[i] / 4);
    CHECK_VALUE(log, deep_copy.get_local_range()[i], sizes[i] / 4, i);

    // TODO: mark this check as testing deprecated functionality
    CHECK_TYPE(log, deep_copy.get_offset()[i], (size_t)0);
    CHECK_VALUE(log, deep_copy.get_offset()[i], (size_t)0, i);

    CHECK_TYPE(log, deep_copy.get_group_range()[i], sizes[i] / (sizes[i] / 4));
    CHECK_VALUE(log, deep_copy.get_group_range()[i], sizes[i] / (sizes[i] / 4), i);

    // TODO: mark this check as testing deprecated functionality
    sycl::nd_range<dim> with_offset(gs, ls, offset);
    CHECK_TYPE(log, with_offset.get_global_range()[i], sizes[i]);
    CHECK_VALUE(log, with_offset.get_global_range()[i], sizes[i], i);
    CHECK_TYPE(log, with_offset.get_local_range()[i], sizes[i] / 4);
    CHECK_VALUE(log, with_offset.get_local_range()[i], sizes[i] / 4, i);
    CHECK_TYPE(log, with_offset.get_offset()[i], sizes[i] / 8);
    CHECK_VALUE(log, with_offset.get_offset()[i], sizes[i] / 8, i);
    CHECK_TYPE(log, with_offset.get_group_range()[i], sizes[i] / (sizes[i] / 4));
    CHECK_VALUE(log, with_offset.get_group_range()[i], sizes[i] / (sizes[i] / 4), i);

    // TODO: mark this check as testing deprecated functionality
    sycl::nd_range<dim> deep_copy_offset(with_offset);
    CHECK_TYPE(log, deep_copy_offset.get_global_range()[i], sizes[i]);
    CHECK_VALUE(log, deep_copy_offset.get_global_range()[i], sizes[i], i);
    CHECK_TYPE(log, deep_copy_offset.get_local_range()[i], sizes[i] / 4);
    CHECK_VALUE(log, deep_copy_offset.get_local_range()[i], sizes[i] / 4, i);
    CHECK_TYPE(log, deep_copy_offset.get_offset()[i], sizes[i] / 8);
    CHECK_VALUE(log, deep_copy_offset.get_offset()[i], sizes[i] / 8, i);
    CHECK_TYPE(log, deep_copy_offset.get_group_range()[i], sizes[i] / (sizes[i] / 4));
    CHECK_VALUE(log, deep_copy_offset.get_group_range()[i], sizes[i] / (sizes[i] / 4),
                i);
  }
}

/** test sycl::range::get(int index) return size_t
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   *  @param info, test_base::info structure as output
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   *  @param log, test transcript logging class
   */
  void run(util::logger &log) override {
    {
      // global size to be set to the size
      sycl::range<1> gs_1d(sizes[0]);
      // local size to be set to 1/4 of the sizes
      sycl::range<1> ls_1d(sizes[0] / 4);
      // offset to be set to 1/8 of the sizes
      sycl::id<1> offset_1d(sizes[0] / 8);
      test_nd_range(log, gs_1d, ls_1d, offset_1d);

      // global size to be set to the size
      sycl::range<2> gs_2d(sizes[0], sizes[1]);
      // local size to be set to 1/4 of the sizes
      sycl::range<2> ls_2d(sizes[0] / 4, sizes[1] / 4);
      // offset to be set to 1/8 of the sizes
      sycl::range<2> range_2d(sizes[0] / 8u, sizes[1] / 8u);
      sycl::id<2> offset_2d(range_2d);
      test_nd_range(log, gs_2d, ls_2d, offset_2d);

      // global size to be set to the size
      sycl::range<3> gs_3d(sizes[0], sizes[1], sizes[2]);
      // local size to be set to 1/4 of the sizes
      sycl::range<3> ls_3d(sizes[0] / 4, sizes[1] / 4, sizes[2] / 4);
      // offset to be set to 1/8 of the sizes
      sycl::range<3> range_3d(sizes[0] / 8u, sizes[1] / 8u, sizes[2] / 8u);
      sycl::id<3> offset_3d(range_3d);
      test_nd_range(log, gs_3d, ls_3d, offset_3d);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_range_api__ */
