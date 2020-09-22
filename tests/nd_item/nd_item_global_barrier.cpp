/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../nd_item/nd_item_barrier_common.h"

#define TEST_NAME nd_item_global_barrier

namespace nd_item_global_barrier__ {
using namespace sycl_cts;

class global_barrier_kernel_fence;

/** test cl::sycl::nd_item global barrier
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

      const auto barrierCall = [](cl::sycl::nd_item<1> item) {
          item.barrier(cl::sycl::access::fence_space::global_space);
        };

      // Verify global barrier works as fence for global address space
      {
        const bool passed =
            test_barrier_global_space<global_barrier_kernel_fence>(
                log, cmdQueue, barrierCall);

        if (!passed) {
          FAIL(log, "global barrier failed for global address space");
        }
      }

      cmdQueue.wait_and_throw();
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

} /* namespace nd_item_global_barrier__ */
