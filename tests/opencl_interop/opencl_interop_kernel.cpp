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

#define TEST_NAME opencl_interop_kernel

namespace opencl_interop_kernel__ {
using namespace sycl_cts;

/**
 * @brief Trivially-copyable standard layout custom type
 */
struct simple_struct {
  int a;
  float b;
};

/** tests the kernel execution for OpenCL inter-op
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

    {
      const size_t bufferSize = 32;
      int bufferData[bufferSize] = {0};

      auto queue = util::get_cts_object::queue(ctsSelector);
      auto context = queue.get_context();
      auto device = queue.get_device();

      sycl::buffer<int, 1> buffer(bufferData, sycl::range<1>(bufferSize));

      cl_program clProgram{};
      if (online_compiler_supported(
              sycl::get_native<sycl::backend::opencl>(device), log)) {
        std::string kernelSource = R"(
            struct simple_struct {
              int a;
              float b;
            };
            __kernel void opencl_interop_kernel_kernel(__global int* arg0_buffer,
                                                   float arg1,
                                                   int arg2,
                                                   struct simple_struct arg3)
            {}
            )";

        if (!create_built_program(
                kernelSource, sycl::get_native<sycl::backend::opencl>(context),
                sycl::get_native<sycl::backend::opencl>(device), clProgram,
                log)) {
          FAIL(log, "create_built_program failed");
        }
      } else {
        std::string programBinaryFile = "opencl_interop_kernel.bin";

        if (!create_program_with_binary(
                programBinaryFile,
                sycl::get_native<sycl::backend::opencl>(context),
                sycl::get_native<sycl::backend::opencl>(device), clProgram,
                log)) {
          std::string errorMsg = "create_program_with_binary failed.";
          errorMsg +=
              " Since online compile is not supported, expecting to find " +
              programBinaryFile + " in same path as the executable binary";
          FAIL(log, errorMsg.c_str());
        }
      }

      cl_kernel clKernel{};
      if (!create_kernel(clProgram, "opencl_interop_kernel_kernel", clKernel,
                         log)) {
        FAIL(log, "create_kernel failed");
      }

      sycl::kernel kernel = sycl::make_kernel(clKernel, context);

      /** test single_task(kernel)
       */
      queue.submit([&](sycl::handler &handler) {
        auto bufferAccessor = buffer.get_access<sycl::access_mode::read_write,
                                                sycl::target::device>(handler);

        simple_struct simpleStruct{19, 13.37f};

        /** check the set_arg() methods
         */

        // set_args(int, buffer)
        handler.set_arg(0, bufferAccessor);
        // set_args(int, float)
        handler.set_arg(1, 15.0f);
        // set_args(int, int)
        handler.set_arg(2, 17);
        // set_args(int, simple_struct)
        handler.set_arg(3, simpleStruct);

        handler.single_task(kernel);
      });

      /** test parallel_for(const nd range<dimensions>&, kernel)
       */
      queue.submit([&](sycl::handler &handler) {
        auto bufferAccessor = buffer.get_access<sycl::access_mode::read_write,
                                                sycl::target::device>(handler);

        simple_struct simpleStruct{19, 13.37f};

        /** check the set_args() method
         */
        handler.set_args(bufferAccessor, 15.0f, 17, simpleStruct);

        sycl::range<1> myRange(1024);
        handler.parallel_for(myRange, kernel);
      });

      queue.wait_and_throw();
    }

    // TODO: add checks to sampled_image_accessor, unsampled_image_accessor

#else
    log.note("The test is skipped because OpenCL back-end is not supported");
#endif  // SYCL_BACKEND_OPENCL
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace opencl_interop_kernel__ */
