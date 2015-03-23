/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

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
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      bool gpuAvailable = false;
      bool cpuAvailable = false;
      auto devices = cl::sycl::device::get_devices();
      for (auto device : devices) {
        if (device.is_cpu()) {
          gpuAvailable = true;
        }
        if (device.is_gpu()) {
          cpuAvailable = true;
        }
      }

      /** check default_selector
      */
      cl::sycl::default_selector defaultSelector;
      cl::sycl::device defaultDevice(defaultSelector);

      /** check host_selector
      */
      cl::sycl::host_selector hostSelector;
      cl::sycl::device hostDevice(hostSelector);

      if (!hostDevice.is_host()) {
        FAIL(log, "host_selector failed to select an appropriate device");
      }

      /** check cpu_selector
      */
      if (cpuAvailable) {
        cl::sycl::cpu_selector cpuSelector;
        cl::sycl::device cpuDevice(cpuSelector);
        if (!(cpuDevice.is_cpu())) {
          FAIL(log, "cpu_selector failed to select an appropriate device");
        }
      }

      /** check gpu_selector
      */
      if (gpuAvailable) {
        cl::sycl::gpu_selector gpuSelector;
        cl::sycl::device gpuDevice(gpuSelector);
        if (!(gpuDevice.is_gpu())) {
          FAIL(log, "gpu_selector failed to select an appropriate device");
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

} /* namespace device_selector_predefined__ */
