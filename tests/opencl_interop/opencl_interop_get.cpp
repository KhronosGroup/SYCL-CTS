/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifdef SYCL_BACKEND_OPENCL
#include "../../util/opencl_helper.h"
#include "../../util/test_base_opencl.h"
#endif
#include "../common/common.h"

#define TEST_NAME opencl_interop_get

namespace opencl_interop_get__ {
using namespace sycl_cts;

class event_kernel;

/** tests the get_native() methods for OpenCL inter-op
 */
class TEST_NAME :
#ifdef SYCL_BACKEND_OPENCL
    public sycl_cts::util::test_base_opencl
#else
    public util::test_base
#endif
{
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
#ifdef SYCL_BACKEND_OPENCL
    auto queue = util::get_cts_object::queue();
    if (queue.get_backend() != sycl::backend::opencl) {
      log.note("Interop part is not supported on non-OpenCL backend types");
      return;
    }
    cts_selector ctsSelector;
    const auto ctsContext = util::get_cts_object::context(ctsSelector);
    const auto ctsDevice = ctsContext.get_devices()[0];

    /** check get_native() for platform
     */
    {
      auto platform = util::get_cts_object::platform(ctsSelector);
      auto interopPlatformID =
          sycl::get_native<sycl::backend::opencl>(platform);
      check_return_type<cl_platform_id>(log, interopPlatformID,
                                        "get_native(platform)");

      if (interopPlatformID == 0) {
        FAIL(log,
             "get_native(platform) did not return a valid "
             "cl_platform_id");
      }
    }

    /** check get_native() for device
     */
    {
      auto device = util::get_cts_object::device(ctsSelector);
      auto interopDeviceID = sycl::get_native<sycl::backend::opencl>(device);
      check_return_type<cl_device_id>(log, interopDeviceID,
                                      "get_native(device)");

      if (interopDeviceID == 0) {
        FAIL(log, "get_native(device) did not return a valid cl_device_id");
      }
    }

    /** check get_native() for context
     */
    {
      auto context = util::get_cts_object::context(ctsSelector);
      auto interopContext = sycl::get_native<sycl::backend::opencl>(context);
      check_return_type<cl_context>(log, interopContext, "get_native(context)");

      if (interopContext == nullptr) {
        FAIL(log, "get_native(context) did not return a valid cl_context");
      }
    }

    /** check get_native() for queue
     */
    {
      auto queue = util::get_cts_object::queue(ctsSelector);
      auto interopQueue = sycl::get_native<sycl::backend::opencl>(queue);
      check_return_type<cl_command_queue>(log, interopQueue,
                                          "get_native(queue)");

      if (interopQueue == nullptr) {
        FAIL(log,
             "get_native(queue) did not return a valid "
             "cl_command_queue");
      }
    }

    /** check get_native() for kernel_bundle
     */
    {
      if (!util::get_cts_object::queue(ctsSelector)
               .get_device()
               .get_info<sycl::info::device::is_compiler_available>()) {
        log.note("online compiler not available -- skipping check");
      } else {
        auto bundle =
            sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctsContext);
        if (!program.is_host()) {
          auto interopProgram = sycl::get_native<sycl::backend::opencl>(bundle);
          check_return_type<cl_program>(log, interopProgram,
                                        "get_native(kernel_bundle)");

          if (interopProgram == nullptr) {
            FAIL(log,
                 "get_native(kernel_bundle) did not return a valid "
                 "cl_program");
          }
        }
      }
    }

    /** check get_native() for kernel
     */
    {
      if (!ctsContext.is_host()) {
        cl_program clProgram{};
        if (online_compiler_supported(
                sycl::get_native<sycl::backend::opencl>(ctsDevice), log)) {
          std::string kernelSource = R"(
            __kernel void opencl_interop_get_kernel() {}
            )";

          if (!create_built_program(
                  kernelSource,
                  sycl::get_native<sycl::backend::opencl>(ctsContext),
                  sycl::get_native<sycl::backend::opencl>(ctsDevice), clProgram,
                  log)) {
            FAIL(log, "create_built_program failed");
          }
        } else {
          std::string programBinaryFile = "opencl_interop_get.bin";

          if (!create_program_with_binary(
                  programBinaryFile,
                  sycl::get_native<sycl::backend::opencl>(ctsContext),
                  sycl::get_native<sycl::backend::opencl>(ctsDevice), clProgram,
                  log)) {
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

        sycl::kernel kernel = sycl::make_kernel(clKernel, ctsContext);

        auto interopKernel = sycl::get_native<sycl::backend::opencl>(kernel);
        check_return_type<cl_kernel>(log, interopKernel, "get_native(kernel)");

        if (interopKernel == nullptr) {
          FAIL(log, "get_native(kernel) did not return a valid cl_kernel");
        }
      }
    }

    /** check get_native() for event
     */
    {
      auto ctsQueue = util::get_cts_object::queue(ctsSelector);

      sycl::event event = ctsQueue.submit([&](sycl::handler &cgh) {
        cgh.single_task<class event_kernel>([](){});
      });

      if (!event.is_host()) {
        auto interopEvent = sycl::get_native<sycl::backend::opencl>(event);
        check_return_type<cl_event>(log, interopEvent, "get_native(event)");

        if (interopEvent == nullptr) {
          FAIL(log, "get_native(event) did not return a valid cl_event");
        }
      }

      ctsQueue.wait_and_throw();
    }

#else
    log.note("The test is skipped because OpenCL back-end is not supported");
#endif  // SYCL_BACKEND_OPENCL
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace opencl_interop_get__ */
