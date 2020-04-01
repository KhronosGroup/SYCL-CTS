/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME platform_constructors

namespace platform_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::platform
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
      /** check default constructor and destructor
      */
      {
        cl::sycl::platform platform;

        if (!platform.is_host()) {
          FAIL(log, "platform was not constructed correctly (is_host)");
        }
      }

      /** check (const device_selector) constructor
      */
      {
        const cts_selector selector;
        cl::sycl::platform platform(selector);

        if (platform.is_host() != selector.is_host()) {
          FAIL(log, "platform was not constructed correctly (is_host)");
        }
      }

      /** check copy constructor
      */
      {
        cts_selector selector;
        cl::sycl::platform platformA(selector);
        cl::sycl::platform platformB(platformA);

        if (platformA.is_host() != platformB.is_host()) {
          FAIL(log, "platform was not copy constructed correctly (is_host)");
        }

#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
        if (!selector.is_host() && platformA.get() != platformB.get()) {
          FAIL(log, "platform was not copy constructed correctly (get)");
        }
#endif
      }

      /** check assignment operator
      */
      {
        cts_selector selector;
        cl::sycl::platform platformA(selector);
        cl::sycl::platform platformB = platformA;

        if (platformA.is_host() != platformB.is_host()) {
          FAIL(log, "platform was not copy assigned correctly (is_host)");
        }

#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
        if (!selector.is_host() && platformA.get() != platformB.get()) {
          FAIL(log, "platform was not copy assigned correctly (get)");
        }
#endif
      }

      /** check move constructor
       */
      {
        cts_selector selector;
        cl::sycl::platform platformA(selector);
        cl::sycl::platform platformB(std::move(platformA));

        if (selector.is_host() != platformB.is_host()) {
          FAIL(log, "platform was not move constructed correctly (is_host)");
        }
      }

      /** check move assignment operator
       */
      {
        cts_selector selector;
        cl::sycl::platform platformA(selector);
        cl::sycl::platform platformB = std::move(platformA);

        if (selector.is_host() != platformB.is_host()) {
          FAIL(log, "platform was not move assigned correctly (is_host)");
        }
      }

      /* check equality operator
       */
      {
        cts_selector selector;
        cl::sycl::platform platformA(selector);
        cl::sycl::platform platformB = platformA;
        cl::sycl::platform platformC(selector);
        platformC = platformA;
        cl::sycl::platform platformD(selector);

        if (!(platformA == platformB)) {
          FAIL(log,
               "platform equality does not work correctly (copy constructed)");
        }
        if (!(platformA == platformC)) {
          FAIL(log,
               "platform equality does not work correctly (copy assigned)");
        }
        if (platformA != platformB) {
          FAIL(log,
               "platform non-equality does not work correctly"
               "(copy constructed)");
        }
        if (platformA != platformC) {
          FAIL(log,
               "platform non-equality does not work correctly"
               "(copy assigned)");
        }
        if (platformC == platformD) {
          FAIL(log,
               "platform equality does not work correctly"
               "(comparing same)");
        }
        if (!(platformC != platformD)) {
          FAIL(log,
               "platform non-equality does not work correctly"
               "(comparing same)");
        }
      }

      /* check hash
       */
      {
        cts_selector selector;
        cl::sycl::platform platformA(selector);
        cl::sycl::platform platformB = platformA;
        cl::sycl::platform platformC(platformA);

        cl::sycl::hash_class<cl::sycl::platform> hasher;

        if (hasher(platformA) != hasher(platformB)) {
          FAIL(
              log,
              "platform hash_class does not work correctly (copy constructed)");
        }
        if (hasher(platformA) != hasher(platformC)) {
          FAIL(log,
               "platform hash_class does not work correctly (copy assigned)");
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

} /* namespace platform_constructors__ */
