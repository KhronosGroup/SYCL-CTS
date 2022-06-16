/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME platform_constructors

namespace platform_constructors__ {
using namespace sycl_cts;

/** tests the constructors for sycl::platform
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
        sycl::platform platform;
        sycl::device device;
        // Check if the devices platform contain the device returned by the
        // `default_selector_v`
        // Assume that `device` default constructor use `default_selector_v` as
        // by the spec
        auto platform_devices = platform.get_devices();
        if (std::find(platform_devices.begin(), platform_devices.end(),
                      device) == platform_devices.end()) {
          FAIL(log,
               "platform was not constructed correctly (doesn't contain "
               "default)");
        }
      }

      /** check (const device_selector) constructor
       */
      {
        const cts_selector selector;
        sycl::platform platform(selector);
        sycl::device device(selector);
        const auto platform_devices = platform.get_devices();
        if (std::find(platform_devices.begin(), platform_devices.end(),
                      device) == platform_devices.end()) {
          FAIL(log,
               "platform was not constructed correctly (doesn't contain asked "
               "device)");
        }
      }

      /** check copy constructor
       */
      {
        cts_selector selector;
        sycl::platform platformA(selector);
        sycl::platform platformB(platformA);

        if (platformA != platformB) {
          FAIL(log, "platform was not copy constructed correctly");
        }

#ifdef SYCL_BACKEND_OPENCL
        auto queue = util::get_cts_object::queue();
        if (queue.get_backend() == sycl::backend::opencl) {
          if (sycl::get_native<sycl::backend::opencl>(platformA) !=
              sycl::get_native<sycl::backend::opencl>(platformB)) {
            FAIL(log, "platform was not copy constructed correctly");
          }
        }
#endif
      }

      /** check assignment operator
       */
      {
        cts_selector selector;
        sycl::platform platformA(selector);
        sycl::platform platformB = platformA;

        // Assume `==` work
        if (platformA != platformB) {
          FAIL(log, "platform was not copy assigned correctly");
        }

#ifdef SYCL_BACKEND_OPENCL
        auto queue = util::get_cts_object::queue();
        if (queue.get_backend() == sycl::backend::opencl) {
          if (sycl::get_native<sycl::backend::opencl>(platformA) !=
              sycl::get_native<sycl::backend::opencl>(platformB)) {
            FAIL(log, "platform was not copy assigned correctly");
          }
        }
#endif
      }

      /** check move constructor
       */
      {
        cts_selector selector;
        sycl::platform platformA(selector);
        sycl::platform platformB(platformA);
        sycl::platform platformC(std::move(platformA));

        if (platformB != platformC) {
          FAIL(log, "platform was not move constructed correctly");
        }
      }

      /** check move assignment operator
       */
      {
        cts_selector selector;
        sycl::platform platformA(selector);
        sycl::platform platformB(platformA);
        sycl::platform platformC = std::move(platformA);

        if (platformB != platformC) {
          FAIL(log, "platform was not move assigned correctly");
        }
      }

      /* check equality operator
       */
      {
        cts_selector selector;
        sycl::platform platformA(selector);
        sycl::platform platformB = platformA;
        sycl::platform platformC(selector);
        platformC = platformA;

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
      }

      /* check hash
       */
      {
        cts_selector selector;
        sycl::platform platformA(selector);
        sycl::platform platformB = platformA;
        sycl::platform platformC(platformA);

        std::hash<sycl::platform> hasher;

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
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace platform_constructors__ */
