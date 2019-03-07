/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME hierarchical_implicit_barriers

namespace TEST_NAMESPACE {

static const unsigned int globalItems1d = 8;
static const unsigned int globalItems2d = 4;
static const unsigned int globalItems3d = 2;
static const unsigned int localItems1d = 4;
static const unsigned int localItems2d = 2;
static const unsigned int localItems3d = 1;
static const unsigned int groupRange1d = (globalItems1d / localItems1d);
static const unsigned int groupRange2d = (globalItems2d / localItems2d);
static const unsigned int groupRange3d = (globalItems3d / localItems3d);
static const int groupItemsTotal =
    (globalItems1d * globalItems2d * globalItems3d);
static const unsigned int localItemsTotal =
    (localItems1d * localItems2d * localItems3d);
static const unsigned int groupRangeTotal = (groupItemsTotal / localItemsTotal);

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
      int inputData[groupItemsTotal];

      auto testQueue = util::get_cts_object::queue();

      for (size_t i = 0; i < groupItemsTotal; i++) {
        inputData[i] = i;
      }
      {
        cl::sycl::buffer<int, 1> input_buffer(
            inputData, cl::sycl::range<1>(groupItemsTotal));

        testQueue.submit([&](cl::sycl::handler &cgh) {
          auto globalRange =
              cl::sycl::range<3>(groupRange1d, groupRange2d, groupRange3d);
          auto localRange =
              cl::sycl::range<3>(localItems1d, localItems2d, localItems3d);

          auto inputPtr =
              input_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);

          cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local>
              localPtr(cl::sycl::range<1>(localItemsTotal), cgh);

          cgh.parallel_for_work_group<class hierarchical_implicit_barriers>(
              globalRange, localRange, [=](cl::sycl::group<3> group) {
                group.parallel_for_work_item([&](cl::sycl::h_item<3> item) {
                  auto globalId = item.get_global().get_linear_id();
                  auto localId = item.get_local().get_linear_id();

                  int globalSize = group.get_global_range().size();
                  int invertedVal = (globalSize - 1) - inputPtr[globalId];

                  localPtr[localId] = invertedVal;
                });

                group.parallel_for_work_item([&](cl::sycl::h_item<3> item) {
                  auto globalId = item.get_global().get_linear_id();
                  auto localId = item.get_local().get_linear_id();

                  inputPtr[globalId] = localPtr[localId];
                });

              });
        });
      }
      testQueue.wait_and_throw();

      for (int i = 0; i < groupItemsTotal; i++) {
        if (inputData[(groupItemsTotal - 1) - i] != i) {
          std::cout << i << " : " << inputData[(groupItemsTotal - 1) - i]
                    << "\n";
          FAIL(log, "Values not equal.");
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

} /* namespace id_api__ */
