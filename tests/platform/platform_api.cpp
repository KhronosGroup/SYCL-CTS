/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME platform_api

namespace platform_api__ {
using namespace sycl_cts;

/** tests the api for cl::sycl::platform
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
      /** check get_devices() member function
      */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto devs = plt.get_devices();
        check_return_type<cl::sycl::vector_class<cl::sycl::device>>(
            log, devs, "platform::get_devices()");
      }

      /** check get_devices(info::device_type::all) member function
      */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto devs = plt.get_devices(cl::sycl::info::device_type::all);
        if (devs.size() != 0) {
          check_return_type<cl::sycl::vector_class<cl::sycl::device>>(
              log, devs, "platform::get_devices(info::device_type::all)");
        }
      }

      /** check has_extensions() member function
      */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto extensionSupported =
            plt.has_extension(cl::sycl::string_class("cl_khr_icd"));
        check_return_type<bool>(log, extensionSupported,
                                "platform::has_extension(string_class)");
      }

      /** check get_info() member function
      */
      {
        cts_selector selector;
        auto plt = util::get_cts_object::platform(selector);
        auto platformName = plt.get_info<cl::sycl::info::platform::name>();
        check_return_type<cl::sycl::string_class>(log, platformName,
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
        auto plt = cl::sycl::platform::get_platforms();
        check_return_type<cl::sycl::vector_class<cl::sycl::platform>>(
            log, plt, "platform::get_platform()");
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

} /* namespace platform_api__ */
