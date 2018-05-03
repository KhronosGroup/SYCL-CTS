/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_properties

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** tests the properties for cl::sycl::queue
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

      /** check property::queue::enable_profiling
      */
      {
        cl::sycl::queue queue(cl::sycl::property_list{
            cl::sycl::property::queue::enable_profiling()});

        if (!queue
                 .has_property<cl::sycl::property::queue::enable_profiling>()) {
          FAIL(log,
               "queue with enable_profiling property was not constructed "
               "correctly");
        }

        auto prop =
            queue.get_property<cl::sycl::property::queue::enable_profiling>();
        check_return_type<cl::sycl::property::queue::enable_profiling>(
            log, prop,
            "cl::sycl::queue::has_property<cl::sycl::property::queue::"
            "enable_profiling>()");
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
