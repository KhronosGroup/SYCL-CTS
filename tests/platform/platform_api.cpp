/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME platform_api

namespace platform_api__ {
using namespace sycl_cts;

/** tests the api for sycl::platform
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
      /** check get_devices() member function
      */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto devs = plt.get_devices();
        check_return_type<std::vector<sycl::device>>(
            log, devs, "platform::get_devices()");
      }

      /** check get_devices(info::device_type::all) member function
      */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto devs = plt.get_devices(sycl::info::device_type::all);
        if (devs.size() != 0) {
          check_return_type<std::vector<sycl::device>>(
              log, devs, "platform::get_devices(info::device_type::all)");
        }
      }

      /** check has() member function
      */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto extensionSupported = plt.has(sycl::aspect::cpu);
        check_return_type<bool>(log, extensionSupported,
                                "platform::has(sycl::aspect)");
      }

      /** check has_extensions() member function
      */
      // TODO: mark this check as testing deprecated functionality
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto extensionSupported =
            plt.has_extension(std::string("cl_khr_icd"));
        check_return_type<bool>(log, extensionSupported,
                                "platform::has_extension(string_class)");
      }

      /** check get_info() member function
      */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto platformName = plt.get_info<sycl::info::platform::name>();
        check_return_type<std::string>(log, platformName,
                                                  "platform::get_info()");
      }

      /** check is_host() member function
      */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto isHost = plt.is_host();
        check_return_type<bool>(log, isHost, "platform::is_host()");
      }

      /** check get_platforms() static method
      */
      {
        auto plt = sycl::platform::get_platforms();
        check_return_type<std::vector<sycl::platform>>(
            log, plt, "platform::get_platform()");
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace platform_api__ */
