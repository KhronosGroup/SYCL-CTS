/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_constructors

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** tests the constructors for sycl::device
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
    {
      /** check default constructor and destructor
       */
      {
        sycl::device device;

        if (!device.is_host()) {
          FAIL(log, "device was not constructed correctly (is_host)");
        }
      }

      /** check (device_selector) constructor
       */
      {
        cts_selector selector;
        sycl::device device(selector);

        if (device.is_host() != selector.is_host()) {
          FAIL(log, "device was not constructed correctly (is_host)");
        }
      }

      /** check copy constructor
       */
      {
        cts_selector selector;
        sycl::device deviceA(selector);
        sycl::device deviceB(deviceA);

        if (deviceA.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not copied correctly (is_host)");
        }

#ifdef SYCL_BACKEND_OPENCL
        auto queue = util::get_cts_object::queue();
        if (queue.get_backend() == sycl::backend::opencl) {
          if (sycl::get_native<sycl::backend::opencl>(deviceA) !=
              sycl::get_native<sycl::backend::opencl>(deviceB)) {
            FAIL(log, "device was not assigned correctly (get)");
          }
        }
#endif
      }

      /** check assignment operator
       */
      {
        cts_selector selector;
        sycl::device deviceA(selector);
        sycl::device deviceB = deviceA;

        if (deviceA.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not assigned correctly (is_host)");
        }
#ifdef SYCL_BACKEND_OPENCL
        auto queue = util::get_cts_object::queue();
        if (queue.get_backend() == sycl::backend::opencl) {
          if (sycl::get_native<sycl::backend::opencl>(deviceA) !=
              sycl::get_native<sycl::backend::opencl>(deviceB)) {
            FAIL(log, "device was not assigned correctly (get)");
          }
        }
#endif
      }

      /** check move constructor
       */
      {
        cts_selector selector;
        sycl::device deviceA(selector);
        sycl::device deviceB(std::move(deviceA));

        if (selector.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not move constructed correctly (is_host)");
        }
      }

      /** check move assignment operator
       */
      {
        cts_selector selector;
        sycl::device deviceA(selector);
        sycl::device deviceB = std::move(deviceA);

        if (selector.is_host() != deviceB.is_host()) {
          FAIL(log, "device was not move assigned correctly (is_host)");
        }
      }

      /* check equality operator
       */
      {
        cts_selector selector;
        sycl::device deviceA(selector);
        sycl::device deviceB(deviceA);
        sycl::device deviceC(selector);
        deviceC = deviceA;

        if (!(deviceA == deviceB)) {
          FAIL(log,
               "device equality does not work correctly (copy constructed)");
        }
        if (!(deviceA == deviceC)) {
          FAIL(log, "device equality does not work correctly (copy assigned)");
        }
        if (deviceA != deviceB) {
          FAIL(log,
               "device non-equality does not work correctly"
               "(copy constructed)");
        }
        if (deviceA != deviceC) {
          FAIL(log,
               "device non-equality does not work correctly"
               "(copy assigned)");
        }
      }

      /* check hash
       */
      {
        cts_selector selector;
        sycl::device deviceA(selector);
        sycl::device deviceB(deviceA);
        sycl::device deviceC = deviceA;

        std::hash<sycl::device> hasher;

        if (hasher(deviceA) != hasher(deviceB)) {
          FAIL(log,
               "device hash_class does not work correctly (copy constructed)");
        }
        if (hasher(deviceA) != hasher(deviceC)) {
          FAIL(log,
               "device hash_class does not work correctly (copy assigned)");
        }
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace device_constructors__ */
