/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_selector_custom

namespace device_selector_custom__ {
using namespace sycl_cts;

/** tests custom selectors for cl::sycl::device_selector
 */
class call_selector : public cl::sycl::device_selector {
 public:
  mutable bool called;

  call_selector() : called(false) {}

  virtual int operator()(const cl::sycl::device &device) const {
    called = true;
    return 1;
  }
};

/** create a selector for testing negative scores
 */
class negative_selector : public cl::sycl::device_selector {
 public:
  negative_selector() {}

  virtual int operator()(const cl::sycl::device &device) const {
    if (!device.is_host()) {
      return -1;
    } else {
      return 1;
    }
  }
};

/** check that we can use a custom device selector
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      /** check a custom selector for a device
      */
      {
        call_selector selector;
        auto device = util::get_cts_object::device(selector);

        /* check our device selector was used */
        if (!selector.called) {
          FAIL(log, "custom selector was never used for creating a device");
        }
      }

      /** check a custom selector for a platform
      */
      {
        call_selector selector;
        auto platform = util::get_cts_object::platform(selector);

        /* check our device selector was used */
        if (!selector.called) {
          FAIL(log, "custom selector was never used for creating a platform");
        }
      }

      /** check a custom selector for a context
      */
      {
        call_selector selector;
        auto context = util::get_cts_object::context(selector);

        /* check our device selector was used */
        if (!selector.called) {
          FAIL(log, "custom selector was never used for creating a context");
        }
      }

      /** check a custom selector for a queue
      */
      {
        call_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        /* check our device selector was used */
        if (!selector.called) {
          FAIL(log, "custom selector was never used for creating a queue");
        }
      }

      /** check the ability to set negative scores
      */
      {
        negative_selector selector;
        auto device = util::get_cts_object::device(selector);

        /* check our device selector was used */
        if (!device.is_host()) {
          FAIL(log, "custom selector selected a device with a negative score");
        }
      }
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

} /* namespace device_selector_custom__ */
