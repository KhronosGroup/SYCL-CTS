/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_GET_CTS_OBJECT_H
#define __SYCLCTS_TESTS_COMMON_GET_CTS_OBJECT_H

// include our proxy to the real sycl header
#include "sycl.h"

#include "../common/cts_async_handler.h"
#include "../common/cts_selector.h"

/** @brief dummy kernel functor for checks that don't require a kernel
 */
template <class kernel_name = void>
struct dummy_functor {
  void operator()() {}
};

namespace sycl_cts {
namespace util {

/**
  @brief Object factory that returns SYCL objects using the CTS selector and CTS
  async handler
*/
struct get_cts_object {
  /**
    @brief Creates a SYCL device
    @param selector Device selector to use to create the device. Uses the CTS
    selector by default.
    @return Default SYCL device
  */
  static cl::sycl::device device(
      const cl::sycl::device_selector &selector = cts_selector()) {
    return cl::sycl::device(selector);
  }

  /**
    @brief Creates a SYCL platform
    @param selector Device selector to use to create the platform. Uses the CTS
    selector by default.
    @return Default SYCL platform
  */
  static cl::sycl::platform platform(
      const cl::sycl::device_selector &selector = cts_selector()) {
    return cl::sycl::platform(selector);
  }

  /**
    @brief Creates a SYCL queue using the CTS async handler
    @param selector Device selector to use to create the queue. Uses the CTS
    selector by default.
    @return Default SYCL queue
  */
  static cl::sycl::queue queue(
      const cl::sycl::device_selector &selector = cts_selector()) {
    static cts_async_handler asyncHandler;
    return cl::sycl::queue(selector, asyncHandler);
  }

  /**
    @brief Creates a SYCL context using the CTS async handler
    @param selector Device selector to use to create the context. Uses the CTS
    selector by default.
    @return Default SYCL context
  */
  static cl::sycl::context context(
      const cl::sycl::device_selector &selector = cts_selector()) {
    static cts_async_handler asyncHandler;
    return cl::sycl::context(selector.select_device(), asyncHandler);
  }

  /**
    @brief Helper class that holds different methods for creating different
    kernels
  */
  struct kernel {
    /**
      @brief Builds a kernel from the specified kernel name and returns it
      @tparam kernel_name Name of the kernel to build
      @param queue Queue that contains the context for the kernel
      @return The built kernel
    */
    template <class kernel_name>
    static cl::sycl::kernel prebuilt(cl::sycl::queue &queue) {
      cl::sycl::program program(queue.get_context());
      program.build_with_kernel_type<kernel_name>();
      return program.get_kernel<kernel_name>();
    }
  };

  /**
    @brief Helper class that holds different methods for creating different
    programs
  */
  struct program {
    /**
      @brief Creates and compiles a SYCL program
      @param ctx Context to create the program from
      @param compileOptions Options passed to the compilation of the program
      @param selector Device selector to use to create the queue where the
      program will be compiled on. Uses the CTS selector by default. Should use
      the same selector as the one used for the context.
      @return Compiled SYCL program
    */
    template <class kernel_name>
    static cl::sycl::program compiled(
        const cl::sycl::context &ctx,
        const cl::sycl::string_class &compileOptions = "",
        const cl::sycl::device_selector &selector = cts_selector()) {
      cl::sycl::program program(ctx);

      auto q = queue(selector);
      q.submit([](cl::sycl::handler &cgh) {
        cgh.single_task(dummy_functor<kernel_name>());
      });
      program.compile_with_kernel_type<dummy_functor<kernel_name>>(
          compileOptions);

      return program;
    }

    /**
      @brief Creates and builds a SYCL program
      @param ctx Context to create the program from
      @param buildOptions Options passed when building the program
      @param selector Device selector to use to create the queue where the
      program will be built on. Uses the CTS selector by default. Should use the
      same selector as the one used for the context.
      @return Compiled SYCL program
    */
    template <class kernel_name>
    static cl::sycl::program built(
        const cl::sycl::context &ctx,
        const cl::sycl::string_class &buildOptions = "",
        const cl::sycl::device_selector &selector = cts_selector()) {
      cl::sycl::program program(ctx);

      auto q = queue(selector);
      q.submit([](cl::sycl::handler &cgh) {
        cgh.single_task(dummy_functor<kernel_name>());
      });
      program.build_with_kernel_type<dummy_functor<kernel_name>>(buildOptions);

      return program;
    }
  };
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_COMMON_GET_CTS_OBJECT_H