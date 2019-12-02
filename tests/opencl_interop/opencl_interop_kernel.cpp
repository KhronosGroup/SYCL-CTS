/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME opencl_interop_kernel

namespace opencl_interop_kernel__ {
using namespace sycl_cts;

/** check inter-op types
 */
template <typename T>
using globalPtrType = typename cl::sycl::global_ptr<T>::pointer;
template <typename T>
using constantPtrType = typename cl::sycl::constant_ptr<T>::pointer;
template <typename T>
using localPtrType = typename cl::sycl::local_ptr<T>::pointer;
template <typename T>
using privatePtrType = typename cl::sycl::private_ptr<T>::pointer;
template <typename T>
using globalMultiPtrType = typename cl::sycl::multi_ptr<
    T, cl::sycl::access::address_space::global_space>::pointer;
template <typename T>
using constantMultiPtrType = typename cl::sycl::multi_ptr<
    T, cl::sycl::access::address_space::constant_space>::pointer;
template <typename T>
using localMultiPtrType = typename cl::sycl::multi_ptr<
    T, cl::sycl::access::address_space::local_space>::pointer;
template <typename T>
using privateMultiPtrType = typename cl::sycl::multi_ptr<
    T, cl::sycl::access::address_space::private_space>::pointer;
template <typename T, int dims>
using vectorType = typename cl::sycl::vec<T, dims>::vector_t;

/**
 * @brief Trivially-copyable standard layout custom type
 */
struct simple_struct {
  int a;
  float b;
};

/** tests the kernel execution for OpenCL inter-op
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

      if (ctsContext.is_host()) {
        log.note("OpenCL interop doesn't work on host");
        return;
      }

      {
        const size_t bufferSize = 32;
        int bufferData[bufferSize] = {0};

        auto queue = util::get_cts_object::queue(ctsSelector);
        auto context = queue.get_context();
        auto device = queue.get_device();

        cl::sycl::buffer<int, 1> buffer(bufferData,
                                        cl::sycl::range<1>(bufferSize));

        cl_program clProgram{};
        if (online_compiler_supported(device.get(), log)) {
          cl::sycl::string_class kernelSource = R"(
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

          if (!create_built_program(kernelSource, context.get(), device.get(),
                                    clProgram, log)) {
            FAIL(log, "create_built_program failed");
          }
        } else {
          cl::sycl::string_class programBinaryFile =
              "opencl_interop_kernel.bin";

          if (!create_program_with_binary(programBinaryFile, context.get(),
                                          device.get(), clProgram, log)) {
            cl::sycl::string_class errorMsg =
                "create_program_with_binary failed.";
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

        cl::sycl::kernel kernel(clKernel, context);

        /** test single_task(kernel)
         */
        queue.submit([&](cl::sycl::handler &handler) {
          auto bufferAccessor =
              buffer.get_access<cl::sycl::access::mode::read_write,
                                cl::sycl::access::target::global_buffer>(
                  handler);

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
        queue.submit([&](cl::sycl::handler &handler) {
          auto bufferAccessor =
              buffer.get_access<cl::sycl::access::mode::read_write,
                                cl::sycl::access::target::global_buffer>(
                  handler);

          simple_struct simpleStruct{19, 13.37f};

          /** check the set_args() method
           */
          handler.set_args(bufferAccessor, 15.0f, 17, simpleStruct);

          cl::sycl::range<1> myRange(1024);
          handler.parallel_for(myRange, kernel);
        });

        queue.wait_and_throw();
      }

      {
        if (!util::get_cts_object::queue(ctsSelector)
                 .get_device()
                 .get_info<cl::sycl::info::device::image_support>()) {
          log.note("Device does not support images");
        } else {
          static constexpr size_t imageSideSize = 32;
          static constexpr size_t imgAccElemSize = 4;  // rgba
          static constexpr auto imageSize =
              (imgAccElemSize * imageSideSize * imageSideSize);
          float imageData[imageSize] = {0.0f};

          auto queue = util::get_cts_object::queue(ctsSelector);
          auto context = queue.get_context();
          auto device = queue.get_device();

          cl::sycl::image<2> image(
              imageData, cl::sycl::image_channel_order::rgba,
              cl::sycl::image_channel_type::fp32,
              cl::sycl::range<2>(imageSideSize, imageSideSize));

          cl_program clProgram{};
          if (online_compiler_supported(device.get(), log)) {
            cl::sycl::string_class kernelSource = R"(
              struct simple_struct {
                int a;
                float b;
              };

              __kernel void opencl_interop_image_kernel_kernel(read_only image2d_t arg0,
                                                           sampler_t arg1)
              {}
              )";

            if (!create_built_program(kernelSource, context.get(), device.get(),
                                      clProgram, log)) {
              FAIL(log, "create_built_program failed");
            }
          } else {
            cl::sycl::string_class programBinaryFile =
                "opencl_interop_image_kernel.bin";

            if (!create_program_with_binary(programBinaryFile, context.get(),
                                            device.get(), clProgram, log)) {
              cl::sycl::string_class errorMsg =
                  "create_program_with_binary failed.";
              errorMsg +=
                  " Since online compile is not supported, expecting to find " +
                  programBinaryFile + " in same path as the executable binary";
              FAIL(log, errorMsg.c_str());
            }
          }

          cl_kernel clKernel{};
          if (!create_kernel(clProgram, "opencl_interop_image_kernel_kernel",
                             clKernel, log)) {
            FAIL(log, "create_kernel failed");
          }

          cl::sycl::kernel kernel(clKernel, context);

          /** test single_task(kernel)
           */
          queue.submit([&](cl::sycl::handler &handler) {
            auto imageAccessor =
                image
                    .get_access<cl::sycl::float4, cl::sycl::access::mode::read>(
                        handler);

            cl::sycl::sampler sampler(
                cl::sycl::coordinate_normalization_mode::unnormalized,
                cl::sycl::addressing_mode::none,
                cl::sycl::filtering_mode::nearest);

            /** check the set_arg() methods
             */

            // set_args(int, image)
            handler.set_arg(0, imageAccessor);
            // set_args(int, sampler)
            handler.set_arg(1, sampler);

            handler.single_task(kernel);
          });

          /** test parallel_for(const nd range<dimensions>&, kernel)
           */
          queue.submit([&](cl::sycl::handler &handler) {
            auto imageAccessor =
                image
                    .get_access<cl::sycl::float4, cl::sycl::access::mode::read>(
                        handler);

            cl::sycl::sampler sampler(
                cl::sycl::coordinate_normalization_mode::unnormalized,
                cl::sycl::addressing_mode::none,
                cl::sycl::filtering_mode::nearest);

            /** check the set_args() method
             */
            handler.set_args(imageAccessor, sampler);

            cl::sycl::range<1> myRange(1024);
            handler.parallel_for(myRange, kernel);
          });

          queue.wait_and_throw();
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

} /* namespace opencl_interop_kernel__ */
