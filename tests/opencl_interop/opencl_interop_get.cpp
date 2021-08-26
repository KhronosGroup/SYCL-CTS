/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../../util/opencl_helper.h"
#include "../../util/test_base_opencl.h"
#include "../common/common.h"

#define TEST_NAME opencl_interop_get

struct program_get_kernel {
  void operator()() const {}
};

namespace opencl_interop_get__ {
using namespace sycl_cts;

class event_kernel;

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
                                          "sycl::platform::get()");

        if (interopPlatformID == 0) {
          FAIL(log,
               "sycl::platform::get() did not return a valid "
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
                                        "sycl::device::get()");

        if (interopDeviceID == 0) {
          FAIL(log, "sycl::device::get() did not return a valid cl_device_id");
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
                                      "sycl::context::get()");

        if (interopContext == nullptr) {
          FAIL(log, "sycl::context::get() did not return a valid cl_context");
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
                                            "sycl::queue::get()");

        if (interopQueue == nullptr) {
          FAIL(log,
               "sycl::queue::get() did not return a valid "
               "cl_command_queue");
        }
      }
    }

    /** check program get() method
     */
    {
      if (!util::get_cts_object::queue(ctsSelector)
               .get_device()
               .get_info<sycl::info::device::is_compiler_available>()) {
        log.note("online compiler not available -- skipping check");
      } else {
        auto program =
            util::get_cts_object::program::compiled<program_get_kernel>(
                ctsContext);
        if (!program.is_host()) {
          auto interopProgram = program.get();
          check_return_type<cl_program>(log, interopProgram,
                                        "sycl::program::get()");

          if (interopProgram == nullptr) {
            FAIL(log, "sycl::program::get() did not return a valid cl_program");
          }
        }
      }
    }

    /** check kernel get() method
     */
    {
      if (!ctsContext.is_host()) {
        cl_program clProgram{};
        if (online_compiler_supported(ctsDevice.get(), log)) {
          std::string kernelSource = R"(
            __kernel void opencl_interop_get_kernel() {}
            )";

          if (!create_built_program(kernelSource, ctsContext.get(),
                                    ctsDevice.get(), clProgram, log)) {
            FAIL(log, "create_built_program failed");
          }
        } else {
          std::string programBinaryFile = "opencl_interop_get.bin";

          if (!create_program_with_binary(programBinaryFile, ctsContext.get(),
                                          ctsDevice.get(), clProgram, log)) {
            std::string errorMsg = "create_program_with_binary failed.";
            errorMsg +=
                " Since online compile is not supported, expecting to find " +
                programBinaryFile + " in same path as the executable binary";
            FAIL(log, errorMsg.c_str());
          }
        }

        cl_kernel clKernel{};
        if (!create_kernel(clProgram, "opencl_interop_get_kernel", clKernel,
                           log)) {
          FAIL(log, "create_kernel failed");
        }

        sycl::kernel kernel(clKernel, ctsContext);

        auto interopKernel = kernel.get();
        check_return_type<cl_kernel>(log, interopKernel, "sycl::kernel::get()");

        if (interopKernel == nullptr) {
          FAIL(log, "sycl::kernel::get() did not return a valid cl_kernel");
        }
      }
    }

    /** check event get() method
     */
    {
      auto ctsQueue = util::get_cts_object::queue(ctsSelector);

      sycl::event event = ctsQueue.submit([&](sycl::handler &cgh) {
        cgh.single_task<class event_kernel>([]() {});
      });

      if (!event.is_host()) {
        auto interopEvent = event.get();
        check_return_type<cl_event>(log, interopEvent, "sycl::event::get()");

        if (interopEvent == nullptr) {
          FAIL(log, "sycl::event::get() did not return a valid cl_event");
        }
      }

      ctsQueue.wait_and_throw();
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace opencl_interop_get__ */
