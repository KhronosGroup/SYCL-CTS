/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME context_constructors

namespace context_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::context
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
      cl::sycl::function_class<void(cl::sycl::exception_list)> asyncHandler =
          [&](cl::sycl::exception_list l) {};

      /** check default constructor and destructor
      */
      {
        cl::sycl::context context;

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (functor_class) constructor
      */
      {
        cl::sycl::context context(asyncHandler);

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (host_selector) constructor
      */
      {
        cl::sycl::host_selector selector;
        cl::sycl::context context(selector);

        if (!context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() != nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (device_selector) constructor
      */
      {
        cts_selector selector;
        cl::sycl::context context(selector);

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (device_selector, functor_class) constructor
      */
      {
        cts_selector selector;
        cl::sycl::context context(selector, asyncHandler);

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (device) constructor
      */
      {
        cts_selector selector;
        cl::sycl::device device(selector);
        cl::sycl::context context(device);

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (device, functor_class) constructor
      */
      {
        cts_selector selector;
        cl::sycl::device device(selector);
        cl::sycl::context context(device, asyncHandler);

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (vector_class<device>) constructor
      */
      {
        cts_selector selector;
        cl::sycl::platform platform(selector);
        auto deviceList = platform.get_devices();
        cl::sycl::context context(deviceList);

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (vector_class<device>, functor_class) constructor
      */
      {
        cts_selector selector;
        cl::sycl::platform platform(selector);
        auto deviceList = platform.get_devices();
        cl::sycl::context context(deviceList, asyncHandler);

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (platform) constructor
      */
      {
        cts_selector selector;
        cl::sycl::platform platform(selector);
        cl::sycl::context context(platform);

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check (platform, functor_class) constructor
      */
      {
        cts_selector selector;
        cl::sycl::platform platform(selector);
        cl::sycl::context context(platform, asyncHandler);

        if (context.is_host()) {
          FAIL(log, "context was not constructed correctly");
        }

        if (context.get() == nullptr) {
          FAIL(log, "context was not constructed correctly");
        }
      }

      /** check copy constructor
      */
      {
        cts_selector selector;
        cl::sycl::context contextA(selector);
        cl::sycl::context contextB(contextA);

        if (contextA.get() != contextB.get()) {
          FAIL(log, "context was not copied correctly");
        }
      }

      /** check assignment operator
      */
      {
        cts_selector selector;
        cl::sycl::context contextA(selector);
        cl::sycl::context contextB = contextA;

        if (contextA.get() != contextB.get()) {
          FAIL(log, "context was not assigned correctly");
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

} /* namespace context_constructors__ */