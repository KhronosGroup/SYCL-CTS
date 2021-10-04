/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_GET_CTS_OBJECT_H
#define __SYCLCTS_TESTS_COMMON_GET_CTS_OBJECT_H

#include <sycl/sycl.hpp>

#include "../common/cts_async_handler.h"
#include "../common/cts_selector.h"

#include <cassert>

/** @brief dummy kernel functor for checks that don't require a kernel
 */
template <class kernel_name = void>
struct dummy_functor {
  void operator()() const {}
  void operator()(sycl::group<3> g) const {}
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
  static sycl::device device(
      const sycl::device_selector &selector = cts_selector()) {
    return sycl::device(selector);
  }

  /**
    @brief Creates a SYCL platform
    @param selector Device selector to use to create the platform. Uses the CTS
    selector by default.
    @return Default SYCL platform
  */
  static sycl::platform platform(
      const sycl::device_selector &selector = cts_selector()) {
    return sycl::platform(selector);
  }

  /**
    @brief Creates a SYCL queue using the CTS async handler
    @param selector Device selector to use to create the queue. Uses the CTS
    selector by default.
    @return Default SYCL queue
  */
  static sycl::queue queue(
      const sycl::device_selector &selector = cts_selector()) {
    static cts_async_handler asyncHandler;
#if !defined(__HIPSYCL__)
    return sycl::queue(selector, asyncHandler, sycl::property_list{});
#else
    return sycl::queue(selector.select_device(), asyncHandler,
                       sycl::property_list{});
#endif
  }

  /**
    @brief Creates a SYCL context using the CTS async handler
    @param selector Device selector to use to create the context. Uses the CTS
    selector by default.
    @return Default SYCL context
  */
  static sycl::context context(
      const sycl::device_selector &selector = cts_selector()) {
    static cts_async_handler asyncHandler;
    return sycl::context(selector.select_device(), asyncHandler);
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
    static sycl::kernel prebuilt(sycl::queue &queue) {
      auto ctx = queue.get_context();
      auto kb_exe = sycl::get_kernel_bundle<
                      kernel_name, sycl::bundle_state::executable>(ctx);
      return kb_exe.get_kernel(sycl::get_kernel_id<kernel_name>());
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
  static sycl::range<1> get(size_t r0, size_t, size_t) {
    return sycl::range<1>(r0);
  }

  /**
   * @brief Constructs a range<1> by projecting any bigger range on its first
   *        dimension
   */
  template <int dims>
  static sycl::range<1> get(const sycl::range<dims>& range) {
    return sycl::range<1>(range[0]);
  }

  /**
   * @brief Constructs a range<1> by only using total size given
   * @tparam totalSize Value the size() call should return for the range
   * @return range<1>
   */
  template <size_t totalSize>
  static sycl::range<1> get_fixed_size(size_t, size_t) {
    return sycl::range<1>(totalSize);
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
  static sycl::range<2> get(size_t r0, size_t r1, size_t) {
    return sycl::range<2>(r0, r1);
  }

  /**
   * @brief Constructs a range<2> by projecting any bigger range on its 2 first
   *        dimensions
   */
  template <int dims>
  static sycl::range<2> get(const sycl::range<dims>& range) {
    return sycl::range<2>(range[0], range[1]);
  }

  /**
   * @brief Constructs a range<2> by only using first component and the
   *        total size given
   * @tparam totalSize Value the size() call should return for the range
   * @param r0 Value of the first component of the range
   * @return range<2>
   */
  template <size_t totalSize>
  static sycl::range<2> get_fixed_size(size_t r0, size_t) {
    assert("Parameters passed for fixed size range are not supported" &&
           (totalSize % r0 == 0));
    return sycl::range<2>(r0, totalSize / r0);
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
  static sycl::range<3> get(size_t r0, size_t r1, size_t r2) {
    return sycl::range<3>(r0, r1, r2);
  }

  /**
   * @brief Common code support for bigger range usage
   */
  static sycl::range<3> get(sycl::range<3> range) {
    return range;
  }

  /**
   * @brief Constructs a range<3> by only using two components and the
   *        total size given
   * @tparam totalSize Value the size() call should return for the range
   * @param r0 Value of the first component of the range
   * @param r0 Value of the second component of the range
   * @return range<3>
   */
  template <size_t totalSize>
  static sycl::range<3> get_fixed_size(size_t r0, size_t r1) {
    assert("Parameters passed for fixed size range are not supported" &&
           (totalSize % (r0 * r1) == 0));
    return sycl::range<3>(r0, r1, totalSize / r0 / r1);
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
  static sycl::id<1> get(size_t v0, size_t, size_t) {
    return sycl::id<1>(v0);
  }
  /**
   * @brief Constructs an id<1> by using any bigger id
   */
  template <int dims>
  static sycl::id<1> get(const sycl::id<dims>& id) {
    return sycl::id<1>(id[0]);
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
  static sycl::id<2> get(size_t v0, size_t v1, size_t) {
    return sycl::id<2>(v0, v1);
  }
  /**
   * @brief Constructs an id<2> by using any bigger id
   */
  template <int dims>
  static sycl::id<2> get(const sycl::id<dims>& id) {
    return sycl::id<2>(id[0], id[1]);
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
  static sycl::id<3> get(size_t v0, size_t v1, size_t v2) {
    return sycl::id<3>(v0, v1, v2);
  }
  /**
   * @brief Common code support for bigger id usage
   */
  static sycl::id<3> get(sycl::id<3> id) {
    return id;
  }
};

}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_COMMON_GET_CTS_OBJECT_H
