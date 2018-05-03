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
      static const cl::sycl::string_class kernelSource =
          R"(
            __kernel void test_kernel(__global int *argOne,
                                      read_only image2d_t arg2,
                                      sampler_t arg3, float arg4)
            {
                ;
            })";

      const size_t bufferSize = 32;
      int bufferData[bufferSize] = {0};

      const size_t imageSize = 1024;
      float imageData[imageSize] = {0.0f};

      auto queue = util::get_cts_object::queue();

      cl::sycl::buffer<int, 1> buffer(bufferData,
                                      cl::sycl::range<1>(bufferSize));

      cl::sycl::image<imageSize> image(
          imageData, cl::sycl::image_channel_order::rgba,
          cl::sycl::image_channel_type::fp32, cl::sycl::range<2>(32, 32));

      cl_program clProgram = nullptr;
      if (!create_built_program(kernelSource, clProgram, log)) {
        FAIL(log, "create_built_program failed");
      }

      cl_kernel clKernel = nullptr;
      if (!create_kernel(clProgram, "test_kernel", clKernel, log)) {
        FAIL(log, "create_kernel failed");
      }

      cl::sycl::kernel kernel(clKernel);

      /** test single_task(kernel)
      */
      queue.submit([&](cl::sycl::handler &handler) {
        auto bufferAccessor =
            buffer.get_access<cl::sycl::access::mode::read_write,
                              cl::sycl::access::target::global_buffer>(handler);
        auto imageAccessor =
            image.get_access<cl::sycl::float4, cl::sycl::access::mode::read>(
                handler);

        cl::sycl::sampler sampler(false, cl::sycl::addressing_mode::none,
                                  cl::sycl::filtering_mode::nearest);

        /** check the set_arg() methods
        */
        handler.set_arg(0, bufferAccessor);
        handler.set_arg(1, imageAccessor);
        handler.set_arg(2, sampler);
        handler.set_arg(3, 15.0f);

        handler.single_task(kernel);
      });

      /** test parallel_for(const nd range<dimensions>&, kernel)
      */
      queue.submit([&](cl::sycl::handler &handler) {
        auto bufferAccessor =
            buffer.get_access<cl::sycl::access::mode::read_write,
                              cl::sycl::access::target::global_buffer>(handler);
        auto imageAccessor =
            image.get_access<cl::sycl::float4, cl::sycl::access::mode::read>(
                handler);

        cl::sycl::sampler sampler(false, cl::sycl::addressing_mode::none,
                                  cl::sycl::filtering_mode::nearest);

        /** check the set_arg() methods
        */
        handler.set_arg(0, bufferAccessor);
        handler.set_arg(1, imageAccessor);
        handler.set_arg(2, sampler);
        handler.set_arg(3, 15.0f);

        cl::sycl::range<1> myRange(1024);
        handler.parallel_for(myRange, kernel);
      });
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
