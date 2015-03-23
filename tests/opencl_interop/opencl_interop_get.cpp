/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_get

namespace opencl_interop_get__ {
using namespace sycl_cts;

/** tests the get() methods for OpenCL inter-op
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl {
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
      cts_selector ctsSelector;

      /** check platform get() method
      */
      cl::sycl::platform platform(ctsSelector);
      auto interopPlatformID = platform.get();
      if (typeid(interopPlatformID) != typeid(cl_platform_id)) {
        FAIL(log, "get() does not return cl_platform_id.");
      }
      if (platform.is_host()) {
        if (interopPlatformID != 0) {
          FAIL(log,
               "platform is in host mode but get() did not return a nullptr");
        }
      } else {
        if (interopPlatformID == 0) {
          FAIL(log, "get() did not return a valid cl_platform_id");
        }
      }

      /** check device get() method
      */
      cl::sycl::device device(ctsSelector);
      auto interopDeviceID = device.get();
      if (typeid(interopDeviceID) != typeid(cl_device_id)) {
        FAIL(log, "get() does not return cl_device_id");
      }
      if (device.is_host()) {
        if (interopDeviceID != 0) {
          FAIL(log,
               "device is in host mode but get() did not return a nullptr");
        }
      } else {
        if (interopDeviceID == 0) {
          FAIL(log, "get() did not return a valid cl_device_id");
        }
        if (!CHECK_CL_SUCCESS(log, clReleaseDevice(interopDeviceID))) {
          FAIL(log, "failed to release the OpenCL device ID");
        }
      }

      /** check context get() method
      */
      cl::sycl::context context(ctsSelector);
      auto interopContext = context.get();
      if (typeid(interopContext) != typeid(cl_context)) {
        FAIL(log, "get() does not return cl_context");
      }
      if (context.is_host()) {
        if (interopContext != nullptr) {
          FAIL(log,
               "context is in host mode but get() did not return a nullptr");
        }
      } else {
        if (interopContext == nullptr) {
          FAIL(log, "get() did not return a valid cl_context");
        }
        if (!CHECK_CL_SUCCESS(log, clReleaseContext(interopContext))) {
          FAIL(log, "failed to release the cl_context");
        }
      }

      /** check queue get() method
      */
      cl::sycl::queue queue(ctsSelector);
      auto interopQueue = queue.get();
      if (typeid(interopQueue) != typeid(cl_command_queue)) {
        FAIL(log, "get() does not return cl_command_queue");
      }
      if (queue.is_host()) {
        if (interopQueue != nullptr) {
          FAIL(log, "queue is in host mode but get() did not return a nullptr");
        }
      } else {
        if (interopQueue == nullptr) {
          FAIL(log, "get() did not return a valid cl_command_queue");
        }
        if (!CHECK_CL_SUCCESS(log, clReleaseCommandQueue(interopQueue))) {
          FAIL(log, "failed to release the cl_command_queue");
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

} /* namespace opencl_interop_get__ */
