/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef SYCL_CTS_TESTS_ATOMIC_API_COMMON_H
#define SYCL_CTS_TESTS_ATOMIC_API_COMMON_H

#include "../common/common.h"
#include "atomic_common.h"

#include <cstring>

/**
 * @brief Helper struct for performing generic operations with regards to the
 *        access target
 * @tparam target Access target
 */
template <cl::sycl::access::target target>
struct target_map;

/**
 * @brief Helper struct for performing operations using a global_buffer target
 */
template <>
struct target_map<cl::sycl::access::target::global_buffer> {
  static constexpr auto target = cl::sycl::access::target::global_buffer;

  static constexpr auto addressSpace =
      cl::sycl::access::address_space::global_space;

  /**
   * @brief Retrieves a global accessor
   * @tparam T Underlying type of the buffer
   * @param buf Buffer to get access to
   * @param cgh Command group handler
   * @return Global accessor
   */
  template <typename T>
  static cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target>
  get_accessor(cl::sycl::buffer<T, 1> &buf, cl::sycl::handler &cgh) {
    return buf.template get_access<cl::sycl::access::mode::atomic, target>(cgh);
  }
};

/**
 * @brief Helper struct for performing operations using a local target
 */
template <>
struct target_map<cl::sycl::access::target::local> {
  static constexpr auto target = cl::sycl::access::target::local;

  static constexpr auto addressSpace =
      cl::sycl::access::address_space::local_space;

  /**
   * @brief Retrieves a local accessor
   * @tparam T Underlying type of the buffer
   * @param cgh Command group handler
   * @return Local accessor
   */
  template <typename T>
  static cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target>
  get_accessor(cl::sycl::buffer<T, 1> &, cl::sycl::handler &cgh) {
    return cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target>(
        cl::sycl::range<1>(1), cgh);
  }
};

/** Check atomic operations
 */
template <typename T, cl::sycl::access::target target,
          template <class, cl::sycl::access::target>
          class check_atomics_functor>
class check_atomics {
 public:
  void operator()(sycl_cts::util::logger &log, cl::sycl::queue &testQueue) {
    auto testDevice = testQueue.get_device();

    T data = 0;
    std::memset(&data, 0xFF, sizeof(T));

    cl::sycl::buffer<T, 1> buf(&data, cl::sycl::range<1>(1));

    testQueue.submit([&](cl::sycl::handler &cgh) {
      auto acc = target_map<target>::get_accessor(buf, cgh);
      auto f = check_atomics_functor<T, target>(acc);
      cgh.single_task(f);
    });
  }
};

template <typename T, template <class, cl::sycl::access::target>
                      class check_atomics_functor>
void generic_check_for_atomics(sycl_cts::util::logger &log,
                               cl::sycl::queue testQueue) {
  /** Check atomics for cl::sycl::access::target::global_buffer
   */
  check_atomics<T, cl::sycl::access::target::global_buffer,
                check_atomics_functor>{}(log, testQueue);

  /** Check atomics for cl::sycl::access::target::local
   */
  check_atomics<T, cl::sycl::access::target::local, check_atomics_functor>{}(
      log, testQueue);
}

/**
 * @brief Generic check wrapper for tests for 64bit types
 */
template <typename T, template <class, cl::sycl::access::target> class functor,
          typename... argsT>
void generic_check_for_atomics64(atomic64_bits_tag::yes, argsT &&... args) {
  generic_check_for_atomics<T, functor>(std::forward<argsT>(args)...);
}

template <typename, template <class, cl::sycl::access::target> class,
          typename... argsT>
void generic_check_for_atomics64(atomic64_bits_tag::no, argsT &&...) {
  // Skip 64bit checks for non-64bit types
}

/**
 * @brief Generic check wrapper for tests for non-64bit types
 */
template <typename T, template <class, cl::sycl::access::target> class functor,
          typename... argsT>
void generic_check_for_atomics32(atomic64_bits_tag::no, argsT &&... args) {
  generic_check_for_atomics<T, functor>(std::forward<argsT>(args)...);
}

template <typename, template <class, cl::sycl::access::target> class,
          typename... argsT>
void generic_check_for_atomics32(atomic64_bits_tag::yes, argsT &&...) {
  // Skip non-64bit checks for 64bit types
}

#endif  // SYCL_CTS_TESTS_ATOMIC_API_COMMON_H
