/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME nd_item_mem_fence

namespace nd_item_mem_fence__ {
using namespace sycl_cts;

class mem_fence_kernel;
void test_mem_fence(util::logger &log, cl::sycl::queue &queue) {
  /* set workspace size */
  const int globalSize = 64;
  const int localSize = 2;

  /* allocate and assign host data */

  /* init ranges*/
  cl::sycl::range<1> globalRange(globalSize);
  cl::sycl::range<1> localRange(localSize);
  cl::sycl::nd_range<1> NDRange(globalRange, localRange);

  /* run kernel to check mem_fence interface is available*/
  {
    queue.submit([&](cl::sycl::handler &cgh) {

      cgh.parallel_for<class mem_fence_kernel>(
          NDRange, [=](cl::sycl::nd_item<1> item) {

            item.mem_fence(cl::sycl::access::fence_space::local_space);
            item.mem_fence(cl::sycl::access::fence_space::global_space);
            item.mem_fence(cl::sycl::access::fence_space::global_and_local);
            item.mem_fence();

          });
    });
  }
}

/** test cl::sycl::nd_item mem_fence
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
    try {
      auto cmdQueue = util::get_cts_object::queue();

      test_mem_fence(log, cmdQueue);

      cmdQueue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_item_mem_fence__ */
