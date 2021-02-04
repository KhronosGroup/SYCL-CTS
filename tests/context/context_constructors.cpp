/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME context_constructors

namespace context_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::context
 */
class TEST_NAME : public util::test_base {
  /** return information about this test
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    cts_async_handler asyncHandler;

    try {
      /** check default constructor and destructor
      */
      {
        cl::sycl::context context;
      }

      /** check (async_handler) constructor
      */
      {
        cl::sycl::context context(asyncHandler);
      }

      /** check (device) constructor
      */
      {
        cts_selector selector;
        auto device = util::get_cts_object::device(selector);
        cl::sycl::context context(device);

        if (context.is_host() != selector.is_host()) {
          FAIL(log, "context was not constructed correctly (is_host)");
        }
      }

      /** check (device, async_handler) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        auto device = util::get_cts_object::device(selector);
        cl::sycl::context context(device, asyncHandler);

        if (context.is_host() != selector.is_host()) {
          FAIL(log, "context was not constructed correctly (is_host)");
        }
      }

      /** check (vector_class<device>) constructor
      */
      {
        cts_selector selector;
        auto platform = util::get_cts_object::platform(selector);
        auto deviceList = platform.get_devices();
        cl::sycl::context context(deviceList);

        if (context.is_host() != selector.is_host()) {
          FAIL(log, "context was not constructed correctly (is_host)");
        }
      }

      /** check (vector_class<device>, async_handler) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        auto platform = util::get_cts_object::platform(selector);
        auto deviceList = platform.get_devices();
        cl::sycl::context context(deviceList, asyncHandler);

        if (context.is_host() != selector.is_host()) {
          FAIL(log, "context was not constructed correctly (is_host)");
        }
      }

      /** check (platform) constructor
      */
      {
        cts_selector selector;
        auto platform = util::get_cts_object::platform(selector);
        cl::sycl::context context(platform);

        if (context.is_host() != selector.is_host()) {
          FAIL(log, "context was not constructed correctly (is_host)");
        }
      }

      /** check (platform, async_handler) constructor
      */
      {
        cts_selector selector;
        cts_async_handler asyncHandler;
        auto platform = util::get_cts_object::platform(selector);
        cl::sycl::context context(platform, asyncHandler);

        if (context.is_host() != selector.is_host()) {
          FAIL(log, "context was not constructed correctly (is_host)");
        }
      }

      /** check copy constructor
      */
      {
        cts_selector selector;
        auto contextA = util::get_cts_object::context(selector);
        cl::sycl::context contextB(contextA);

        if (contextA.is_host() != contextB.is_host()) {
          FAIL(log, "context was not copied correctly (is_host)");
        }

#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
        if (!selector.is_host() && (contextA.get() != contextB.get())) {
          FAIL(log, "context was not copied correctly (get)");
        }
#endif
      }

      /** check assignment operator
      */
      {
        cts_selector selector;
        auto contextA = util::get_cts_object::context(selector);
        cl::sycl::context contextB = contextA;

        if (contextA.is_host() != contextB.is_host()) {
          FAIL(log, "context was not assigned correctly (is_host)");
        }

#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
        if (!selector.is_host() && (contextA.get() != contextB.get())) {
          FAIL(log, "context was not assigned correctly (get)");
        }
#endif
      }

      /** check move constructor
      */
      {
        cts_selector selector;
        auto contextA = util::get_cts_object::context(selector);
        cl::sycl::context contextB(std::move(contextA));

        if (selector.is_host() != contextB.is_host()) {
          FAIL(log, "context was not move constructed correctly (is_host)");
        }
      }

      /** check move assignment operator
      */
      {
        cts_selector selector;
        auto contextA = util::get_cts_object::context(selector);
        cl::sycl::context contextB = std::move(contextA);

        if (selector.is_host() != contextB.is_host()) {
          FAIL(log, "context was not move assigned correctly (is_host)");
        }
      }

      /* check equality operator
      */
      {
        cts_selector selector;
        cl::sycl::context contextA = util::get_cts_object::context(selector);
        cl::sycl::context contextB{contextA};
        cl::sycl::context contextC = util::get_cts_object::context(selector);
        contextC = contextA;
        cl::sycl::context contextD = util::get_cts_object::context(selector);

        if (!(contextA == contextB)) {
          FAIL(log,
               "device equality does not work correctly (equality of equal "
               "failed)");
        }
        if (!(contextA == contextC)) {
          check_equality(log, contextA, contextC, !selector.is_host());
          FAIL(log,
               "device equality does not work correctly (equality of equal "
               "failed)");
        }
        if (contextA != contextB) {
          FAIL(log,
               "context non-equality does not work correctly"
               "(copy constructed)");
        }
        if (contextA != contextC) {
          FAIL(log,
               "context non-equality does not work correctly"
               "(copy assigned)");
        }
        if (contextC == contextD) {
          FAIL(log,
               "context equality does not work correctly"
               "(comparing same)");
        }
        if (!(contextC != contextD)) {
          FAIL(log,
               "context non-equality does not work correctly"
               "(comparing same)");
        }
      }

      /** check hash
      */
      {
        auto contextA = util::get_cts_object::context();
        cl::sycl::context contextB(contextA);
        cl::sycl::hash_class<cl::sycl::context> hasher;

        if (hasher(contextA) != hasher(contextB)) {
          FAIL(log,
               "context hash_class does not work correctly. (hashing of equals "
               "failed)");
        }
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

} /* namespace context_constructors__ */
