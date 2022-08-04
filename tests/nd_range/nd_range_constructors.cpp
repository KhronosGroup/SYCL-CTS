/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
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

#define TEST_NAME nd_range_constructors

namespace nd_range_constructors__ {
using namespace sycl_cts;

/**
 * @brief Constructs a default nd_range
 * @tparam dim Number of dimensions of the nd_range
 * @return nd_range<dim> object with default values
 */
template <int dim>
inline cl::sycl::nd_range<dim> get_default_nd_range() {
  return cl::sycl::nd_range<dim>(getRange<dim>(1), getRange<dim>(1));
}

template <int dim>
void test_nd_range_constructors(util::logger &log, cl::sycl::range<dim> gs,
                                cl::sycl::range<dim> ls,
                                cl::sycl::id<dim> offset) {
  cl::sycl::nd_range<dim> no_offset(gs, ls);
  cl::sycl::nd_range<dim> deep_copy(no_offset);
  cl::sycl::nd_range<dim> with_offset(gs, ls, offset);
  cl::sycl::nd_range<dim> deep_copy_offset(with_offset);

  {  // Copy assignment, no offset
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = no_offset;
  }
  {  // Copy assignment, with offset
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = with_offset;
  }
  {  // Move assignment, no offset
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = std::move(no_offset);
  }
  {  // Move assignment, with offset
    auto defaultRange = get_default_nd_range<dim>();
    defaultRange = std::move(with_offset);
  }
}

/** test cl::sycl::nd_range initialization
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
    const size_t sizes[] = {16, 32, 64};

    try {
      // global size to be set to the size
      cl::sycl::range<1> gs_1d(sizes[0]);
      // local size to be set to 1/4 of the sizes
      cl::sycl::range<1> ls_1d(sizes[0] / 4u);
      // offset to be set to 1/8 of the sizes
      cl::sycl::id<1> offset_1d(sizes[0] / 8u);
      test_nd_range_constructors(log, gs_1d, ls_1d, offset_1d);

      // global size to be set to the size
      cl::sycl::range<2> gs_2d(sizes[0], sizes[1]);
      // local size to be set to 1/4 of the sizes
      cl::sycl::range<2> ls_2d(sizes[0] / 4u, sizes[1] / 4u);
      // offset to be set to 1/8 of the sizes
      cl::sycl::range<2> range_2d(sizes[0] / 8u, sizes[1] / 8u);
      cl::sycl::id<2> offset_2d(range_2d);
      test_nd_range_constructors(log, gs_2d, ls_2d, offset_2d);

      // global size to be set to the size
      cl::sycl::range<3> gs_3d(sizes[0], sizes[1], sizes[2]);
      // local size to be set to 1/4 of the sizes
      cl::sycl::range<3> ls_3d(sizes[0] / 4, sizes[1] / 4, sizes[2] / 4);
      // offset to be set to 1/8 of the sizes
      cl::sycl::range<3> range_3d(sizes[0] / 8u, sizes[1] / 8u, sizes[2] / 8u);
      cl::sycl::id<3> offset_3d(range_3d);
      test_nd_range_constructors(log, gs_3d, ls_3d, offset_3d);
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_range_constructors__ */
