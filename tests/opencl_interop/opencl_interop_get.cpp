/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_get

struct program_get_kernel {
  void operator()() const {}
};

namespace opencl_interop_get__ {
using namespace sycl_cts;

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
      const auto ctsContext = util::get_cts_object::context(ctsSelector);
      const auto ctsDevice = ctsContext.get_devices()[0];

      if (ctsContext.is_host()) {
        log.note("OpenCL interop doesn't work on host");
        return;
      }

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
        auto program =
            util::get_cts_object::program::compiled<program_get_kernel>(
                ctsContext);
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
        if (!ctsContext.is_host()) {
          cl::sycl::string_class kernelSource = R"(
          __kernel void test_kernel() {}
          )";

          cl_program clProgram = nullptr;
          if (!create_built_program(kernelSource, ctsContext.get(),
                                    ctsDevice.get(), clProgram, log)) {
            FAIL(log, "create_built_program failed");
          }

          cl_kernel clKernel = nullptr;
          if (!create_kernel(clProgram, "test_kernel", clKernel, log)) {
            FAIL(log, "create_kernel failed");
          }

          cl::sycl::kernel kernel(clKernel, ctsContext);

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
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);

        cl::sycl::event event = ctsQueue.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task<class event_kernel>([]() {});
        });

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
        if (!ctsContext.is_host()) {
          cl_sampler clSampler;
          create_sampler(clSampler, log);
          cl::sycl::sampler sampler(clSampler, ctsContext);

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
