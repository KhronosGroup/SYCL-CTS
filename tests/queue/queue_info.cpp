/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_info

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test the info for cl::sycl::queue
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

      /** check cl::sycl::info::queue
      */
      check_enum_class_value(cl::sycl::info::queue::reference_count);
      check_enum_class_value(cl::sycl::info::queue::context);
      check_enum_class_value(cl::sycl::info::queue::device);

      /** check get_info parameters
      */
      {
        auto queue = util::get_cts_object::queue();
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);
        check_get_info_param<cl::sycl::info::queue,
          cl::sycl::cl_uint, cl::sycl::info::queue::reference_count>(log,
          queue);
        check_get_info_param<cl::sycl::info::queue, cl::sycl::context,
          cl::sycl::info::queue::context>(log, queue);
        check_get_info_param<cl::sycl::info::queue, cl::sycl::device,
          cl::sycl::info::queue::device>(log, queue);
      }
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

} /* namespace TEST_NAMESPACE */
