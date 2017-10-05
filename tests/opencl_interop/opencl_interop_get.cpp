/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_get

namespace opencl_interop_get__ {
using namespace sycl_cts;

class kernel_get_test;

/** tests the get() methods for OpenCL inter-op
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl {
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
      cts_selector ctsSelector;

      /** check platform get() method
      */
      {
        auto platform = util::get_cts_object::platform(ctsSelector);
        if (!platform.is_host()) {
          auto interopPlatformID = platform.get();
          check_return_type<cl_platform_id>(log, interopPlatformID,
                                            "cl::sycl::platform::get()");

          if (interopPlatformID == 0) {
            FAIL(log,
                 "cl::sycl::platform::get() did not return a valid "
                 "cl_platform_id");
          }
        }
      }

      /** check device get() method
      */
      {
        auto device = util::get_cts_object::device(ctsSelector);
        if (!device.is_host()) {
          auto interopDeviceID = device.get();
          check_return_type<cl_device_id>(log, interopDeviceID,
                                          "cl::sycl::device::get()");

          if (interopDeviceID == 0) {
            FAIL(log,
                 "cl::sycl::device::get() did not return a valid cl_device_id");
          }
        }
      }

      /** check context get() method
      */
      {
        auto context = util::get_cts_object::context(ctsSelector);
        if (!context.is_host()) {
          auto interopContext = context.get();
          check_return_type<cl_context>(log, interopContext,
                                        "cl::sycl::context::get()");

          if (interopContext == nullptr) {
            FAIL(log,
                 "cl::sycl::context::get() did not return a valid cl_context");
          }
        }
      }

      /** check queue get() method
      */
      {
        auto queue = util::get_cts_object::queue(ctsSelector);
        if (!queue.is_host()) {
          auto interopQueue = queue.get();
          check_return_type<cl_command_queue>(log, interopQueue,
                                              "cl::sycl::queue::get()");

          if (interopQueue == nullptr) {
            FAIL(log,
                 "cl::sycl::queue::get() did not return a valid "
                 "cl_command_queue");
          }
        }
      }

      /** check program get() method
      */
      {
        auto context = util::get_cts_object::context(ctsSelector);
        auto program = cl::sycl::program(context);
        if (!program.is_host()) {
          auto interopProgram = program.get();
          check_return_type<cl_program>(log, interopProgram,
                                        "cl::sycl::program::get()");

          if (interopProgram == nullptr) {
            FAIL(log,
                 "cl::sycl::program::get() did not return a valid cl_program");
          }
        }
      }

      /** check kernel get() method
      */
      {
        auto queue = util::get_cts_object::queue(ctsSelector);
        auto kernel =
            util::get_cts_object::kernel::prebuilt<kernel_get_test>(queue);
        if (!kernel.is_host()) {
          auto interopKernel = kernel.get();
          check_return_type<cl_kernel>(log, interopKernel,
                                       "cl::sycl::kernel::get()");

          if (interopKernel == nullptr) {
            FAIL(log,
                 "cl::sycl::kernel::get() did not return a valid cl_kernel");
          }
        }
      }

      /** check event get() method
      */
      {
        auto event = cl::sycl::event();
        if (!event.is_host()) {
          auto interopEvent = event.get();
          check_return_type<cl_event>(log, interopEvent,
                                      "cl::sycl::event::get()");

          if (interopEvent == nullptr) {
            FAIL(log, "cl::sycl::event::get() did not return a valid cl_event");
          }
        }
      }

      /** check sampler get() method
      */
      {
        auto sampler = cl::sycl::sampler(
            cl::sycl::coordinate_normalization_mode::normalized,
            cl::sycl::addressing_mode::mirrored_repeat,
            cl::sycl::filtering_mode::nearest);
        if (!sampler.is_host()) {
          auto interopSampler = sampler.get();
          check_return_type<cl_sampler>(log, interopSampler,
                                        "cl::sycl::sampler::get()");

          if (interopSampler == nullptr) {
            FAIL(log,
                 "cl::sycl::sampler::get() did not return a valid cl_sampler");
          }
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

} /* namespace opencl_interop_get__ */
