/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_implicit_conditional

namespace TEST_NAMESPACE {

static const int globalItems1d = 8;
static const int globalItems2d = 4;
static const int globalItems3d = 2;
static const int localItems1d = 4;
static const int localItems2d = 2;
static const int localItems3d = 1;
static const int groupRange1d = (globalItems1d / localItems1d);
static const int groupRange2d = (globalItems2d / localItems2d);
static const int groupRange3d = (globalItems3d / localItems3d);
static const int groupItemsTotal =
    (globalItems1d * globalItems2d * globalItems3d);
static const int localItemsTotal = (localItems1d * localItems2d * localItems3d);
static const int groupRangeTotal = (groupItemsTotal / localItemsTotal);

using namespace sycl_cts;

/** test cl::sycl::range::get(int index) return size_t
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
    try {
      int outputData[groupItemsTotal];
      for (int i = 0; i < groupItemsTotal; i++) {
        outputData[i] = 0;
      }

      {
        cl::sycl::buffer<int, 1> outputBuffer(
            outputData, cl::sycl::range<1>(groupItemsTotal));

        cl::sycl::queue myQueue(util::get_cts_object::queue());

        myQueue.submit([&](cl::sycl::handler &cgh) {

          auto groupRange =
              cl::sycl::range<3>(groupRange1d, groupRange2d, groupRange3d);
          auto localRange =
              cl::sycl::range<3>(localItems1d, localItems2d, localItems3d);

          auto outputPtr =
              outputBuffer.get_access<cl::sycl::access::mode::read_write>(cgh);

          cgh.parallel_for_work_group<class hierarchical_implicit_conditional>(
              groupRange, localRange, [=](cl::sycl::group<3> group) {
                // Create a local variable to store the work item id.
                int work_item_id;

                group.parallel_for_work_item([&](cl::sycl::h_item<3> item) {
                  // Assign the work item id to a local variable.
                  work_item_id =
                      group.get_linear() * item.get_local_range().size() +
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

        myQueue.wait_and_throw();
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

} /* namespace hierarchical_implicit_conditional__ */
