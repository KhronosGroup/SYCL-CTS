/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME group_constructors

namespace TEST_NAMESPACE {
using namespace sycl_cts;

static const size_t GROUP_RANGE_1D = 16;
static const size_t GROUP_RANGE_2D = 2;
static const size_t GROUP_RANGE_3D = 4;

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
      // dim 1
      {
        cts_selector selector;
        cl::sycl::queue q(selector);

        q.submit([&](cl::sycl::handler &cgh) {
          cgh.parallel_for_work_group<class group_kernel_1>(
              cl::sycl::range<1>(GROUP_RANGE_1D),
              [=](cl::sycl::group<1> my_group) {
                // Check copy constructor
                cl::sycl::group<1> group_copy(my_group);

                // Check copy assignment operator
                my_group = group_copy;
              });
        });

        q.wait_and_throw();
      }

      // dim 2
      {
        cts_selector selector;
        cl::sycl::queue q(selector);

        q.submit([&](cl::sycl::handler &cgh) {
          cgh.parallel_for_work_group<class group_kernel_2>(
              cl::sycl::range<2>(GROUP_RANGE_1D, GROUP_RANGE_2D),
              [=](cl::sycl::group<2> my_group) {
                // Check copy constructor
                cl::sycl::group<2> group_copy(my_group);

                // Check copy assignment operator
                my_group = group_copy;
              });
        });

        q.wait_and_throw();
      }

      // dim 3
      {
        cts_selector selector;
        cl::sycl::queue q(selector);

        q.submit([&](cl::sycl::handler &cgh) {

          cgh.parallel_for_work_group<class group_kernel_3>(
              cl::sycl::range<3>(GROUP_RANGE_1D, GROUP_RANGE_2D,
                                 GROUP_RANGE_3D),
              [=](cl::sycl::group<3> my_group) {
                // Check copy constructor
                cl::sycl::group<3> group_copy(my_group);

                // Check copy assignment operator
                my_group = group_copy;
              });
        });

        q.wait_and_throw();
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

}  // namespace TEST_NAMESPACE
