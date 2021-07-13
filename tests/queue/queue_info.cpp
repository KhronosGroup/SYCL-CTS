/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_info

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test the info for sycl::queue
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
  */
  void run(util::logger &log) override {
    try {
      /** check sycl::info::queue
      */
      check_enum_class_value(sycl::info::queue::reference_count);
      check_enum_class_value(sycl::info::queue::context);
      check_enum_class_value(sycl::info::queue::device);

      /** check get_info parameters
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);
        check_get_info_param<sycl::info::queue, sycl::cl_uint,
                             sycl::info::queue::reference_count>(log,
                                                                     queue);
        check_get_info_param<sycl::info::queue, sycl::context,
                             sycl::info::queue::context>(log, queue);
        check_get_info_param<sycl::info::queue, sycl::device,
                             sycl::info::queue::device>(log, queue);
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
