/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_constructors

namespace event_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::event
 */
class TEST_NAME : public util::test_base {
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      /** check copy constructor
      */
      {
        cts_selector selector;
        cl::sycl::queue queue(selector);
        cl::sycl::event eventA =
            queue.submit([&](cl::sycl::handler &handler) {}).get_complete();
        cl::sycl::event eventB(eventA);

        if (eventA.get() != eventB.get()) {
          FAIL(log, "event was not copied correctly.");
        }
      }

      /** check assignment operator
      */
      {
        cts_selector selector;
        cl::sycl::queue queue(selector);
        cl::sycl::event eventA =
            queue.submit([&](cl::sycl::handler &handler) {}).get_complete();
        cl::sycl::event eventB = eventB;

        if (eventA.get() != eventB.get()) {
          FAIL(log, "event was not assigned correctly.");
        }
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "a sycl exception was caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace event_constructors__ */
