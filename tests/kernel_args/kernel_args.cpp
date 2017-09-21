/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_args

namespace kernel_args__ {
using namespace sycl_cts;

struct nestedStruct {
  long long a;
};

struct outerStruct {
  int a;
  float b;
  nestedStruct innerStruct;
};

/** Test kernel args conform to the rules for parameter passing to kernels
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
      // Test values
      int testScalar = 1;
      auto testVec = cl::sycl::vec<int, 4>(1, 2, 3, 4);
      outerStruct testStruct{1, 2.0f, nestedStruct{3}};
      cl::sycl::sampler sampler(
          cl::sycl::coordinate_normalization_mode::unnormalized,
          cl::sycl::addressing_mode::clamp, cl::sycl::filtering_mode::nearest);

      auto my_queue = util::get_cts_object::queue();
      {
        my_queue.submit([&](cl::sycl::handler &cgh) {

          cgh.single_task<TEST_NAME>([=]() {

            // Test that the values outside kernel have been captured
            int kernelScalar = testScalar;

            cl::sycl::vec<int, 4> kernelVec = cl::sycl::vec<int, 4>(
                testVec.x(), testVec.y(), testVec.z(), testVec.w());

            int a = testStruct.a;
            float b = testStruct.b;
            long long c = testStruct.innerStruct.a;

            cl::sycl::sampler kernel_sampler = sampler;

          });
        });
      }

      my_queue.wait_and_throw();

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_args__ */
