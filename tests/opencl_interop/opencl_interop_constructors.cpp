/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_BACKEND_OPENCL
#include "../../util/opencl_helper.h"
#include "../../util/test_base_opencl.h"
#endif

#define TEST_NAME opencl_interop_constructors

namespace opencl_interop_constructors__ {
using namespace sycl_cts;

class buffer_interop_constructor_kernel;

/** tests the constructors for OpenCL inter-op
 */
class TEST_NAME :
#ifdef SYCL_BACKEND_OPENCL
    public util::test_base_opencl
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
    {
      auto queue = util::get_cts_object::queue();
      if (queue.get_backend() != sycl::backend::opencl) {
        log.skip(
            "OpenCL interoperability part is not supported on non-OpenCL "
            "backend types");
        return;
      }

      cts_selector ctsSelector;
      const auto ctsContext = util::get_cts_object::context(ctsSelector);
      const auto ctsDevice = ctsContext.get_devices()[0];


      std::string kernelSource = R"(
            __kernel void opencl_interop_constructors_kernel(__global float *input)
            {
              input[get_global_id(0)] = get_global_id(0);
            }
            )";
      std::string programBinaryFile =
          "opencl_interop_constructors.bin";
      /** check make_platform (cl_platform_id)
       */
      {
        sycl::platform platform =
            sycl::make_platform<sycl::backend::opencl>(m_cl_platform_id);

        cl_platform_id interopPlatformID =
            sycl::get_native<sycl::backend::opencl>(platform);
        if (interopPlatformID != m_cl_platform_id) {
          FAIL(log, "platform was not constructed correctly");
        }
      }

      /** check make_device (cl_device_id)
       */
      {
        sycl::device device =
            sycl::make_device<sycl::backend::opencl>(m_cl_device_id);

        cl_device_id interopDeviceID =
            sycl::get_native<sycl::backend::opencl>(device);
        if (interopDeviceID != m_cl_device_id) {
          FAIL(log, "device was not constructed correctly");
        }
        if (!CHECK_CL_SUCCESS(log, clReleaseDevice(interopDeviceID))) {
          FAIL(log, "failed to release OpenCL device");
        }
      }

      /** check make_context (cl_context)
       */
      {
        sycl::context context =
            sycl::make_context<sycl::backend::opencl>(m_cl_context);

        cl_context interopContext =
            sycl::get_native<sycl::backend::opencl>(context);
        if (interopContext != m_cl_context) {
          FAIL(log, "context was not constructed correctly");
        }
        if (!CHECK_CL_SUCCESS(log, clReleaseContext(interopContext))) {
          FAIL(log, "failed to release OpenCL context");
        }
      }

      /** check make_context (cl_context, async_handler)
       */
      {
        cts_async_handler asyncHandler;
        sycl::context context = sycl::make_context<sycl::backend::opencl>(
            m_cl_context, asyncHandler);

        cl_context interopContext =
            sycl::get_native<sycl::backend::opencl>(context);
        if (interopContext != m_cl_context) {
          FAIL(log, "context was not constructed correctly");
        }
        if (!CHECK_CL_SUCCESS(log, clReleaseContext(interopContext))) {
          FAIL(log, "failed to release OpenCL context");
        }
      }

      /** check make_queue (cl_command_queue, const context&)
       */
      {
        sycl::queue queue = sycl::make_queue<sycl::backend::opencl>(
            m_cl_command_queue, ctsContext);

        cl_command_queue interopQueue =
            sycl::get_native<sycl::backend::opencl>(queue);
        if (interopQueue != m_cl_command_queue) {
          FAIL(log, "queue was not constructed correctly");
        }

        /** check that queue copy constructor preserve the same OpenCL queue
         */
        sycl::queue queueCopy(queue);
        auto clQueueCopy = sycl::get_native<sycl::backend::opencl>(queueCopy);
        if (interopQueue != clQueueCopy) {
          FAIL(log, "queue destination was not copy constructed correctly");
        }

        if (!CHECK_CL_SUCCESS(log, clReleaseCommandQueue(clQueueCopy))) {
          FAIL(log, "failed to release OpenCL command queue");
        }

        if (!CHECK_CL_SUCCESS(log, clReleaseCommandQueue(interopQueue))) {
          FAIL(log, "failed to release OpenCL command queue");
        }
      }

      /** check make_queue (cl_command_queue, const context&, async_handler)
       */
      {
        cts_async_handler asyncHandler;
        sycl::queue queue = sycl::make_queue<sycl::backend::opencl>(
            m_cl_command_queue, ctsContext, asyncHandler);

        cl_command_queue interopQueue =
            sycl::get_native<sycl::backend::opencl>(queue);
        if (interopQueue != m_cl_command_queue) {
          FAIL(log, "queue was not constructed correctly");
        }
        if (!CHECK_CL_SUCCESS(log, clReleaseCommandQueue(interopQueue))) {
          FAIL(log, "failed to release OpenCL command queue");
        }
      }

      /** check make_kernel_bundle (cl_program, context)
       */
      {
        cl_program clProgram{};
        if (online_compiler_supported(
                sycl::get_native<sycl::backend::opencl>(ctsDevice), log)) {
          if (!create_built_program(
                  kernelSource,
                  sycl::get_native<sycl::backend::opencl>(ctsContext),
                  sycl::get_native<sycl::backend::opencl>(ctsDevice), clProgram,
                  log)) {
            FAIL(log, "create_built_program failed");
          }
        } else {
          if (!create_program_with_binary(
                  programBinaryFile,
                  sycl::get_native<sycl::backend::opencl>(ctsContext),
                  sycl::get_native<sycl::backend::opencl>(ctsDevice), clProgram,
                  log)) {
            std::string errorMsg =
                "create_program_with_binary failed.";
            errorMsg +=
                " Since online compile is not supported, expecting to find " +
                programBinaryFile + " in same path as the executable binary";
            FAIL(log, errorMsg.c_str());
          }
        }

        auto kernel_bundle =
            sycl::make_kernel_bundle<sycl::backend::opencl,
                                     sycl::bundle_state::executable>(
                clProgram, ctsContext);

        std::vector<cl_program> interopProgramVec =
            sycl::get_native<sycl::backend::opencl>(kernel_bundle);
        if (interopProgramVec[0] != clProgram) {
          FAIL(log, "program was not constructed correctly");
        }
        for (int i = 0; i < interopProgramVec.size(); i++) {
          if (!CHECK_CL_SUCCESS(log, clReleaseProgram(interopProgramVec[i]))) {
            FAIL(log, "failed to release OpenCL program");
          }
        }
      }

      /** check make_kernel (cl_kernel, const context&)
       */
      {
        cl_program clProgram{};
        if (online_compiler_supported(
                sycl::get_native<sycl::backend::opencl>(ctsDevice), log)) {
          if (!create_built_program(
                  kernelSource,
                  sycl::get_native<sycl::backend::opencl>(ctsContext),
                  sycl::get_native<sycl::backend::opencl>(ctsDevice), clProgram,
                  log)) {
            FAIL(log, "create_built_program failed");
          }
        } else {
          if (!create_program_with_binary(
                  programBinaryFile,
                  sycl::get_native<sycl::backend::opencl>(ctsContext),
                  sycl::get_native<sycl::backend::opencl>(ctsDevice), clProgram,
                  log)) {
            std::string errorMsg =
                "create_program_with_binary failed.";
            errorMsg +=
                " Since online compile is not supported, expecting to find " +
                programBinaryFile + " in same path as the executable binary";
            FAIL(log, errorMsg.c_str());
          }
        }

        cl_kernel clKernel{};
        if (!create_kernel(clProgram, "opencl_interop_constructors_kernel",
                           clKernel, log)) {
          FAIL(log, "create_kernel failed");
        }

        sycl::kernel kernel =
            sycl::make_kernel<sycl::backend::opencl>(clKernel, ctsContext);

        cl_kernel interopKernel =
            sycl::get_native<sycl::backend::opencl>(kernel);
        if (interopKernel != clKernel) {
          FAIL(log, "kernel was not constructed correctly");
        }
        if (!CHECK_CL_SUCCESS(log, clReleaseKernel(interopKernel))) {
          FAIL(log, "failed to release OpenCL kernel");
        }
      }

      /** check make_buffer (cl_mem, contex)
       */
      {
        const size_t size = 32;
        int data[size] = {0};
        cl_int error = CL_SUCCESS;

        auto queue = util::get_cts_object::queue(ctsSelector);

        cl_mem clBuffer = clCreateBuffer(
            sycl::get_native<sycl::backend::opencl>(queue.get_context()),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size * sizeof(int), data,
            &error);
        if (!CHECK_CL_SUCCESS(log, error)) {
          FAIL(log, "create buffer failed");
        }

        sycl::buffer<int, 1> buffer =
            sycl::make_buffer<sycl::backend::opencl, int>(clBuffer,
                                                          queue.get_context());

        // calculate element count, size and range for the interop buffer
        sycl::range<1> interopRange{size};
        size_t interopSize = size * sizeof(int);


        // check the buffer
        if (buffer.is_sub_buffer()) {
          FAIL(log,
               "opencl buffer was not interop constructed properly. "
               "(is_sub_buffer) ");
        }
        if (buffer.byte_size() != interopSize) {
          FAIL(log,
               "opencl buffer was not interop constructed properly. "
               "(byte_size) ");
        }
        if (buffer.get_range() != interopRange) {
          FAIL(
              log,
              "opencl buffer was not interop constructed properly. (get_range) ");
        }
        if (buffer.size() != size) {
          FAIL(log,
               "opencl buffer was not interop constructed properly. (size) ");
        }

        queue.submit([&](sycl::handler &handler) {
          auto accessor =
              buffer.get_access<sycl::access_mode::read_write,
                                sycl::target::device>(
                  handler);
          handler.single_task<class buffer_interop_constructor_kernel_no_event>([]() {});
        });

        error = clReleaseMemObject(clBuffer);
        if (!CHECK_CL_SUCCESS(log, error)) {
          FAIL(log, "failed to release OpenCL buffer");
        }

        queue.wait_and_throw();
      }

      /** check make_buffer (cl_mem, context, event)
       */
      {
        const size_t size = 32;
        int data[size] = {0};
        cl_int error = CL_SUCCESS;

        auto queue = util::get_cts_object::queue(ctsSelector);

        // create an event to wait for
        sycl::event event = queue.submit([](sycl::handler &cgh) {
          cgh.single_task<class buffer_interop_event>(
              []() {});  // do not do anything here, we only need the event
        });

        cl_mem clBuffer = clCreateBuffer(
            sycl::get_native<sycl::backend::opencl>(queue.get_context()),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size * sizeof(int), data,
            &error);
        if (!CHECK_CL_SUCCESS(log, error)) {
          FAIL(log, "create buffer failed");
        }

        sycl::buffer<int, 1> buffer =
            sycl::make_buffer<sycl::backend::opencl, int>(
                clBuffer, queue.get_context(), event);

        // calculate element count, size and range for the interop buffer
        sycl::range<1> interopRange{size};
        size_t interopSize = size * sizeof(int);


        // check the buffer
        if (buffer.is_sub_buffer()) {
          FAIL(log,
               "opencl buffer was not interop constructed properly. "
               "(is_sub_buffer) ");
        }
        if (buffer.byte_size() != interopSize) {
          FAIL(log,
               "opencl buffer was not interop constructed properly. "
               "(byte_size) ");
        }
        if (buffer.get_range() != interopRange) {
          FAIL(
              log,
              "opencl buffer was not interop constructed properly. (get_range) ");
        }
        if (buffer.size() != size) {
          FAIL(log,
               "opencl buffer was not interop constructed properly. (size) ");
        }

        queue.submit([&](sycl::handler &handler) {
          auto accessor =
              buffer.get_access<sycl::access_mode::read_write,
                                sycl::target::device>(
                  handler);
          handler.single_task<class buffer_interop_constructor_kernel_with_event>([]() {});
        });

        error = clReleaseMemObject(clBuffer);
        if (!CHECK_CL_SUCCESS(log, error)) {
          FAIL(log, "failed to release OpenCL buffer");
        }

        queue.wait_and_throw();
      }
      // TODO: add checks for make_sampled_image and make_unsampled_image

      /** check make_event (cl_event, const context&)
       */
      {
        cl_event clEvent = clCreateUserEvent(
            sycl::get_native<sycl::backend::opencl>(ctsContext), nullptr);
        sycl::event event =
            sycl::make_event<sycl::backend::opencl>(clEvent, ctsContext);

        std::vector<cl_event> interopEventVec =
            sycl::get_native<sycl::backend::opencl>(event);

        if (interopEventVec[0] != clEvent) {
          FAIL(log, "event was not constructed correctly");
        }
      }
    }
#else
    log.note("The test is skipped because OpenCL back-end is not supported");
#endif  // SYCL_BACKEND_OPENCL
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace opencl_interop_constructors__ */
