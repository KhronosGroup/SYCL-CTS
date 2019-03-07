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

      static const cl::sycl::string_class kernelSource = R"(
struct simple_struct {
  int a;
  float b;
};

__kernel void test_kernel(__global int* arg0_buffer,
                          read_only image2d_t arg1,
                          sampler_t arg2,
                          float arg3,
                          int arg4,
                          struct simple_struct arg5,
                          __global char* arg6_stream)
{}
)";

      const size_t bufferSize = 32;
      int bufferData[bufferSize] = {0};

      static constexpr size_t imageSideSize = 32;
      static constexpr size_t imgAccElemSize = 4;  // rgba
      static constexpr auto imageSize =
          (imgAccElemSize * imageSideSize * imageSideSize);
      float imageData[imageSize] = {0.0f};

      auto queue = util::get_cts_object::queue(ctsSelector);
      auto context = queue.get_context();
      auto device = queue.get_device();

      cl::sycl::buffer<int, 1> buffer(bufferData,
                                      cl::sycl::range<1>(bufferSize));

      cl::sycl::image<2> image(
          imageData, cl::sycl::image_channel_order::rgba,
          cl::sycl::image_channel_type::fp32,
          cl::sycl::range<2>(imageSideSize, imageSideSize));

      cl_program clProgram = nullptr;
      if (!create_built_program(kernelSource, context.get(), device.get(),
                                clProgram, log)) {
        FAIL(log, "create_built_program failed");
      }

      cl_kernel clKernel = nullptr;
      if (!create_kernel(clProgram, "test_kernel", clKernel, log)) {
        FAIL(log, "create_kernel failed");
      }

      cl::sycl::kernel kernel(clKernel, context);

      /** test single_task(kernel)
      */
      queue.submit([&](cl::sycl::handler &handler) {
        auto bufferAccessor =
            buffer.get_access<cl::sycl::access::mode::read_write,
                              cl::sycl::access::target::global_buffer>(handler);
        auto imageAccessor =
            image.get_access<cl::sycl::float4, cl::sycl::access::mode::read>(
                handler);

        cl::sycl::sampler sampler(
            cl::sycl::coordinate_normalization_mode::unnormalized,
            cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::nearest);

        simple_struct simpleStruct{19, 13.37f};

        cl::sycl::stream os(2048, 80, handler);

        /** check the set_arg() methods
        */

        // set_args(int, buffer)
        handler.set_arg(0, bufferAccessor);
        // set_args(int, image)
        handler.set_arg(1, imageAccessor);
        // set_args(int, sampler)
        handler.set_arg(2, sampler);
        // set_args(int, float)
        handler.set_arg(3, 15.0f);
        // set_args(int, int)
        handler.set_arg(4, 17);
        // set_args(int, simple_struct)
        handler.set_arg(5, simpleStruct);
        // set_args(int, stream)
        handler.set_arg(6, os);

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

        cl::sycl::sampler sampler(
            cl::sycl::coordinate_normalization_mode::unnormalized,
            cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::nearest);

        simple_struct simpleStruct{19, 13.37f};

        cl::sycl::stream os(2048, 80, handler);

        /** check the set_args() method
        */
        handler.set_args(bufferAccessor, imageAccessor, sampler, 15.0f, 17,
                         simpleStruct, os);

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
