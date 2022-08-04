/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "../common/common.h"
#include "../../util/opencl_helper.h"
#include "../../util/test_base_opencl.h"

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

// Forward declaration of the kernel
template <int N>
struct program_kernel_interop {
  void operator()() const {}
};

/** simple OpenCL test kernel
 */
const cl::sycl::string_class kernelName = "sample";
cl::sycl::string_class kernel_source = R"(
__kernel void sample(__global float * input)
{
    input[get_global_id(0)] = get_global_id(0);
}
)";

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

      auto ctsQueue = util::get_cts_object::queue(ctsSelector);
      auto context = ctsQueue.get_context();
      auto deviceList = context.get_devices();

      // Do ALL devices support online compiler / linker?
      bool compiler_available = is_compiler_available(deviceList);
      bool linker_available = is_linker_available(deviceList);

      const cl::sycl::string_class compileOptions = "-cl-opt-disable";
      const cl::sycl::string_class linkOptions = "-cl-fast-relaxed-math";

      {
        log.note(
            "link an OpenCL and a SYCL program without compile and link "
            "options");

        if (!compiler_available) {
          log.note("online compiler not available -- skipping check");
        }

        else {
          // obtain an existing OpenCL C program object
          cl_program myClProgram = nullptr;
          if (!create_compiled_program(
                  kernel_source, context.get(),
                  ctsQueue.get_device().get(), myClProgram, log)) {
            FAIL(log, "Didn't create the cl_program");
          }

          // Create a SYCL program object from a cl_program object
          cl::sycl::program myExternProgram(context,
                                            myClProgram);

          if (myExternProgram.get_state() !=
              cl::sycl::program_state::compiled) {
            FAIL(log, "Compiled interop program should be in compiled state");
          }

          // Add in the SYCL program object for our kernel
          cl::sycl::program mySyclProgram(context);
          mySyclProgram.compile_with_kernel_type<program_kernel_interop<0>>();

          if (mySyclProgram.get_state() != cl::sycl::program_state::compiled) {
            FAIL(log, "Compiled SYCL program should be in compiled state");
          }

          // Link myClProgram with the SYCL program object
          try {
            cl::sycl::program myLinkedProgram({myExternProgram, mySyclProgram});

            if (myLinkedProgram.get_state() !=
                cl::sycl::program_state::linked) {
              FAIL(log, "Program was not linked");
            }

            ctsQueue.submit([&](cl::sycl::handler &cgh) {
              cgh.single_task(program_kernel_interop<0>());
            });
            ctsQueue.wait_and_throw();

          } catch (const cl::sycl::feature_not_supported &fnse_link) {
            if (!linker_available) {
              log.note("online linker not available -- skipping check");
            } else {
              throw;
            }
          }
        }
      }

      {
        log.note(
            "link an OpenCL and a SYCL program with compile and link options");

        if (!compiler_available) {
          log.note("online compiler not available -- skipping check");
        }

        else {
          // obtain an existing OpenCL C program object
          cl_program myClProgram = nullptr;
          if (!create_compiled_program(
                  kernel_source, context.get(),
                  ctsQueue.get_device().get(), myClProgram, log)) {
            FAIL(log, "Didn't create the cl_program");
          }

          // Create a SYCL program object from a cl_program object
          cl::sycl::program myExternProgram(context,
                                            myClProgram);

          if (myExternProgram.get_state() !=
              cl::sycl::program_state::compiled) {
            FAIL(log, "Compiled interop program should be in compiled state");
          }

          // Add in the SYCL program object for our kernel
          cl::sycl::program mySyclProgram(context);
          mySyclProgram.compile_with_kernel_type<program_kernel_interop<1>>(
              compileOptions);

          if (mySyclProgram.get_state() != cl::sycl::program_state::compiled) {
            FAIL(log, "Compiled SYCL program should be in compiled state");
          }

          if (mySyclProgram.get_compile_options() != compileOptions) {
            FAIL(log,
                 "Compiled SYCL program did not store the compile options");
          }

          // Link myClProgram with the SYCL program object
          try {
            cl::sycl::program myLinkedProgram({myExternProgram, mySyclProgram},
                                              linkOptions);

            if (myLinkedProgram.get_state() !=
                cl::sycl::program_state::linked) {
              FAIL(log, "Program was not linked");
            }

            if (myLinkedProgram.get_link_options() != linkOptions) {
              FAIL(log, "Linked program did not store the link options");
            }

            ctsQueue.submit([&](cl::sycl::handler &cgh) {
              cgh.single_task(program_kernel_interop<1>());
            });
            ctsQueue.wait_and_throw();

          } catch (const cl::sycl::feature_not_supported &fnse_link) {
            if (!linker_available) {
              log.note("online linker not available -- skipping check");
            } else {
              throw;
            }
          }
        }
      }

      if (!context.is_host()) {
        log.note("check compiling and building from source");

        {  // Check compile_with_source(source)
          cl::sycl::program prog(context);
          try {
            prog.compile_with_source(kernel_source);
          } catch (const cl::sycl::feature_not_supported &fnse_compile) {
            if (!compiler_available) {
              log.note("online compiler not available -- skipping check");
            } else {
              throw;
            }
          }
        }
        {  // Check compile_with_source(source, options)
          cl::sycl::program prog(context);
          try {
            prog.compile_with_source(kernel_source, compileOptions);
          } catch (const cl::sycl::feature_not_supported &fnse_compile) {
            if (!compiler_available) {
              log.note("online compiler not available -- skipping check");
            } else {
              throw;
            }
          }
        }
        {  // Check build_with_source(source)
          cl::sycl::program prog(context);
          try {
            prog.build_with_source(kernel_source);
          } catch (const cl::sycl::feature_not_supported &fnse_build) {
            if (!compiler_available || !linker_available) {
              log.note(
                  "online compiler or linker not available -- skipping check");
            } else {
              throw;
            }
          }
        }
        {  // Check build_with_source(source, options)
          cl::sycl::program prog(context);

          try {
            prog.build_with_source(kernel_source, linkOptions);
          } catch (const cl::sycl::feature_not_supported &fnse_build) {
            if (!compiler_available || !linker_available) {
              log.note(
                  "online compiler or linker not available -- skipping check");
            } else {
              throw;
            }
          }
        }

        {  // Check retrieving kernel
          cl::sycl::program prog(context);

          try {
            prog.build_with_source(kernel_source);

            // Check has_kernel(string_class)
            bool hasKernel = prog.has_kernel(kernelName);
            if (!hasKernel) {
              FAIL(log,
                   "Program was not built properly (has_kernel(string_class))");
            }

            // Check get_kernel(string_class)
            cl::sycl::kernel k = prog.get_kernel(kernelName);

          } catch (const cl::sycl::feature_not_supported &fnse_build) {
            if (!compiler_available || !linker_available) {
              log.note(
                  "online compiler or linker not available -- skipping check");
            } else {
              throw;
            }
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

} /* namespace opencl_interop_kernel__ */
