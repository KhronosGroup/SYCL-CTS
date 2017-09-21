/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME device_selector_predefined

namespace device_selector_predefined__ {
using namespace sycl_cts;

/** tests predefined selectors for cl::sycl::device_selector
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      bool gpuAvailable = false;
      bool cpuAvailable = false;
      auto devices = cl::sycl::device::get_devices();
      for (auto device : devices) {
        if (device.is_gpu()) {
          gpuAvailable = true;
        }
        if (device.is_cpu()) {
          cpuAvailable = true;
        }
      }

      /** check default_selector
      */
      if (!cpuAvailable && !gpuAvailable) {
        cl::sycl::default_selector defaultSelector;
        try {
          auto defaultDevice = util::get_cts_object::device(defaultSelector);
          if (!defaultDevice.is_host()) {
            cl::sycl::string_class errorMsg =
                "a SYCL exception occured when default_selector tried to "
                "select a device when no opencl devices available";
            FAIL(log, errorMsg.c_str());
          }
        } catch (const cl::sycl::exception &e) {
          log_exception(log, e);
          FAIL(log,
               "default_selector failed to select a host device when no opencl "
               "devices available");
        }
      }

      /** check host_selector
      */
      cl::sycl::host_selector hostSelector;
      auto hostDevice = util::get_cts_object::device(hostSelector);

      if (!hostDevice.is_host()) {
        FAIL(log, "host_selector failed to select an appropriate device");
      }

      /** check cpu_selector
      */
      if (cpuAvailable) {
        cl::sycl::cpu_selector cpuSelector;
        auto cpuDevice = util::get_cts_object::device(cpuSelector);
        if (!(cpuDevice.is_cpu())) {
          FAIL(log, "cpu_selector failed to select an appropriate device");
        }
      }

      /** check gpu_selector
      */
      if (gpuAvailable) {
        cl::sycl::gpu_selector gpuSelector;
        auto gpuDevice = util::get_cts_object::device(gpuSelector);
        if (!(gpuDevice.is_gpu())) {
          FAIL(log, "gpu_selector failed to select an appropriate device");
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

} /* namespace device_selector_predefined__ */
