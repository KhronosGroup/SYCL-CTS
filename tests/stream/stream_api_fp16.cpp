/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "stream_api_common.h"

#define TEST_NAME stream_api_fp16

namespace TEST_NAMESPACE {

using namespace sycl_cts;

class test_kernel;

/** test sycl::stream interface
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
      // Check stream operator for sycl::half
      {
        auto testQueue = util::get_cts_object::queue();

        if (!testQueue.get_device().has(sycl::aspect::fp16)) {
          log.note(
            "Device does not support half precision floating point operations");
        return;
      }

        testQueue.submit([&](sycl::handler &cgh) {
          sycl::stream os(2048, 80, cgh);

          cgh.single_task<class test_kernel>([=]() {
            check_all_vec_dims(os, sycl::half(0.2f));
            check_all_vec_dims(os, sycl::cl_half(0.3f));
          });
        });

        testQueue.wait_and_throw();
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      sycl::string_class errorMsg =
          "a SYCL exception was caught: " + sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection.
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
