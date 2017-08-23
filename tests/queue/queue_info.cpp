/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_info

namespace queue_info__ {
using namespace sycl_cts;

/** tests the info for cl::sycl::queue
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  virtual void run(util::logger &log) override {
    try {
      auto queue = util::get_cts_object::queue();

      /** check types
      */
      using queueInfo = cl::sycl::info::queue;

      /** initialize return values
      */
      cl_uint refCount;
      cl::sycl::context contextInfo;
      cl::sycl::device deviceInfo;

      /** check device info parameters
      */
      refCount = queue.get_info<cl::sycl::info::queue::reference_count>();
      contextInfo = queue.get_info<cl::sycl::info::queue::context>();
      deviceInfo = queue.get_info<cl::sycl::info::queue::device>();
      auto test = queue.get_info<cl::sycl::info::queue::queue_profiling>();

      TEST_TYPE_TRAIT(queue, reference_count, queue);
      TEST_TYPE_TRAIT(queue, context, queue);
      TEST_TYPE_TRAIT(queue, device, queue);
      TEST_TYPE_TRAIT(queue, queue_profiling, queue);
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace queue_info__ */
