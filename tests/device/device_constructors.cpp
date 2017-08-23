/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_constructors

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::device
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
      /** check default constructor and destructor
       */
      {
        cl::sycl::device device;

        if (!device.is_host()) {
          FAIL(log, "device was not constructed correctly (is_host)");
        }
      }

      /** check (device_selector) constructor
       */
      {
        cts_selector selector;
        cl::sycl::device device(selector);

        if (device.is_host() != selector.is_host()) {
          FAIL(log, "device was not constructed correctly (is_host)");
        }

        if (!selector.is_host() && device.get() == nullptr) {
          FAIL(log, "device was not constructed correctly (get)");
        }
      }

      /** check copy constructor
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB(deviceA);

        if (deviceA.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not copied correctly (is_host)");
        }

        if (!selector.is_host() && deviceA.get() != deviceB.get()) {
          FAIL(log, "device was not assigned correctly (get)");
        }
      }

      /** check assignment operator
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB = deviceA;

        if (deviceA.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not assigned correctly (is_host)");
        }
        if (!selector.is_host() && deviceA.get() != deviceB.get()) {
          FAIL(log, "device was not assigned correctly (get)");
        }
      }

      /** check move constructor
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB(std::move(deviceA));

        if (selector.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not move constructed correctly (is_host)");
        }

        if (!selector.is_host() && deviceB.get() == nullptr) {
          FAIL(log, "device was not move constructed correctly (get)");
        }
      }

      /** check move assignment operator
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB = std::move(deviceA);

        if (selector.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not move assigned correctly (is_host)");
        }

        if (!selector.is_host() && deviceB.get() == nullptr) {
          FAIL(log, "device was not move assigned correctly (get)");
        }
      }

      /* check equality operator
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB(deviceA);
        cl::sycl::device deviceC = deviceA;

        if (!(deviceA == deviceB)) {
          FAIL(log,
               "device equality does not work correctly (copy constructed)");
        }
        if (!(deviceA == deviceC)) {
          FAIL(log, "device equality does not work correctly (copy assigned)");
        }
      }

      /* check hash
       */
      {
        cts_selector selector;
        cl::sycl::device deviceA(selector);
        cl::sycl::device deviceB(deviceA);
        cl::sycl::device deviceC = deviceA;

        cl::sycl::hash_class<cl::sycl::device> hasher;

        if (hasher(deviceA) != hasher(deviceB)) {
          FAIL(log,
               "device hash_class does not work correctly (copy constructed)");
        }
        if (hasher(deviceA) != hasher(deviceC)) {
          FAIL(log,
               "device hash_class does not work correctly (copy assigned)");
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

} /* namespace device_constructors__ */
