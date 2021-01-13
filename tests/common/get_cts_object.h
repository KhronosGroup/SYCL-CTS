/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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
  void operator()(cl::sycl::group<3> g) {}
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
      q.wait_and_throw();
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
      q.wait_and_throw();
      program.build_with_kernel_type<dummy_functor<kernel_name>>(buildOptions);

      return program;
    }
  };

  /**
   * @brief Uniform way to retrieve a range of different dimensions by always
   *        specifying three components
   */
  template <int dimensions>
  struct range;
  /**
   * @brief Uniform way to retrieve an id of different dimensions
   */
  template <int dimensions>
  struct id;
};

/**
 * @brief Specialization that returns a range<1> from three components
 */
template <>
struct get_cts_object::range<1> {
  /**
   * @brief Constructs a range<1> by only using one component out of three
   * @param r0 Value of the first component of the range
   * @return range<1>
   */
  static cl::sycl::range<1> get(size_t r0, size_t, size_t) {
    return cl::sycl::range<1>(r0);
  }
  /**
   * @brief Constructs a range<1> by using any bigger range
   */
  template <int dims>
  static cl::sycl::range<1> get(const cl::sycl::range<dims>& range) {
    return cl::sycl::range<1>(range[0]);
  }
};

/**
 * @brief Specialization that returns a range<2> from three components
 */
template <>
struct get_cts_object::range<2> {
  /**
   * @brief Constructs a range<2> by only using two components out of three
   * @param r0 Value of the first component of the range
   * @param r1 Value of the second component of the range
   * @return range<2>
   */
  static cl::sycl::range<2> get(size_t r0, size_t r1, size_t) {
    return cl::sycl::range<2>(r0, r1);
  }
  /**
   * @brief Constructs a range<2> by using any bigger range
   */
  template <int dims>
  static cl::sycl::range<2> get(const cl::sycl::range<dims>& range) {
    return cl::sycl::range<2>(range[0], range[1]);
  }
};

/**
 * @brief Specialization that returns a range<3> from three components
 */
template <>
struct get_cts_object::range<3> {
  /**
   * @brief Constructs a range<3>
   * @param r0 Value of the first component of the range
   * @param r1 Value of the second component of the range
   * @param r2 Value of the third component of the range
   * @return range<3>
   */
  static cl::sycl::range<3> get(size_t r0, size_t r1, size_t r2) {
    return cl::sycl::range<3>(r0, r1, r2);
  }
  /**
   * @brief Common code support for bigger range usage
   */
  static cl::sycl::range<3> get(cl::sycl::range<3> range) {
    return range;
  }
};

/**
 * @brief Specialization that returns an id<1> from three components
 */
template <>
struct get_cts_object::id<1> {
  /**
   * @brief Constructs an id<1> by only using one component out of three
   * @param v0 Value of the first component of the id
   * @return id<1>
   */
  static cl::sycl::id<1> get(size_t v0, size_t, size_t) {
    return cl::sycl::id<1>(v0);
  }
  /**
   * @brief Constructs an id<1> by using any bigger id
   */
  template <int dims>
  static cl::sycl::id<1> get(const cl::sycl::id<dims>& id) {
    return cl::sycl::id<1>(id[0]);
  }
};
/**
 * @brief Specialization that returns an id<2> from three components
 */
template <>
struct get_cts_object::id<2> {
  /**
   * @brief Constructs an id<2> by only using two components out of three
   * @param v0 Value of the first component of the id
   * @param v1 Value of the second component of the id
   * @return id<2>
   */
  static cl::sycl::id<2> get(size_t v0, size_t v1, size_t) {
    return cl::sycl::id<2>(v0, v1);
  }
  /**
   * @brief Constructs an id<2> by using any bigger id
   */
  template <int dims>
  static cl::sycl::id<2> get(const cl::sycl::id<dims>& id) {
    return cl::sycl::id<2>(id[0], id[1]);
  }
};
/**
 * @brief Specialization that returns an id<3> from three components
 */
template <>
struct get_cts_object::id<3> {
  /**
   * @brief Constructs an id<3>
   * @param v0 Value of the first component of the id
   * @param v1 Value of the second component of the id
   * @param v2 Value of the third component of the id
   * @return id<3>
   */
  static cl::sycl::id<3> get(size_t v0, size_t v1, size_t v2) {
    return cl::sycl::id<3>(v0, v1, v2);
  }
  /**
   * @brief Common code support for bigger id usage
   */
  static cl::sycl::id<3> get(cl::sycl::id<3> id) {
    return id;
  }
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_COMMON_GET_CTS_OBJECT_H
