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

#define TEST_NAME hierarchical_implicit_conditional

namespace TEST_NAMESPACE {

template <int dim> class kernel;

static const int globalItems1d = 8;
static const int globalItems2d = 4;
static const int globalItems3d = 2;
static const int localItems1d = 4;
static const int localItems2d = 2;
static const int localItems3d = 1;
static const int groupRange1d = (globalItems1d / localItems1d);
static const int groupRange2d = (globalItems2d / localItems2d);
static const int groupItemsTotal =
    (globalItems1d * globalItems2d * globalItems3d);
static const int localItemsTotal = (localItems1d * localItems2d * localItems3d);
static const int groupRangeTotal = (groupItemsTotal / localItemsTotal);

using namespace sycl_cts;

template <int dim> void check_dim(util::logger &log) {
  {
    int outputData[groupItemsTotal];
    for (int i = 0; i < groupItemsTotal; i++) {
      outputData[i] = 0;
    }

    {
      sycl::buffer<int, 1> outputBuffer(
          outputData, sycl::range<1>(groupItemsTotal));

      sycl::queue myQueue(util::get_cts_object::queue());

      myQueue.submit([&](sycl::handler &cgh) {

        auto groupRange =
            sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
                groupRangeTotal>(groupRange1d, groupRange2d);
        auto localRange =
            sycl_cts::util::get_cts_object::range<dim>::template get_fixed_size<
                localItemsTotal>(localItems1d, localItems2d);

        auto outputPtr =
            outputBuffer.get_access<sycl::access_mode::read_write>(cgh);

        cgh.parallel_for_work_group<kernel<dim>>(
            groupRange, localRange, [=](sycl::group<dim> group) {
              // Create a local variable to store the work item id.
              int work_item_id;

              group.parallel_for_work_item([&](sycl::h_item<dim> item) {
                // Assign the work item id to a local variable.
                work_item_id =
                    group.get_linear_id() * item.get_local_range().size() +
                    item.get_local().get_linear_id();
              });

              // Assign a value for the work item stored. Although this is
              // not recommened behaviour for the hierarchical API as there
              // is a data race on the itemIds accessor and there is no
              // guarantee which work item id will be taken, this test makes
              // sure that the assigment is only being done once.
              outputPtr[work_item_id] += 2;
            });
      });
    }

    for (int j = 0; j < groupRangeTotal; j++) {
      int sum = 0;
      for (int i = 0; i < localItemsTotal; i++) {
        sum += outputData[j * groupRangeTotal + i];
      }
      // Exactly one thread should have written the memory
      // for the current work group
      if (sum != 2) {
        FAIL(log, "Result not as expected.");
      }
    }
  }
}

/** test sycl::range::get(int index) return size_t
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
    check_dim<1>(log);
    check_dim<2>(log);
    check_dim<3>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace hierarchical_implicit_conditional__ */
