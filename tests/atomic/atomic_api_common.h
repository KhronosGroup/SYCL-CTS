/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#ifndef SYCL_CTS_TESTS_ATOMIC_API_COMMON_H
#define SYCL_CTS_TESTS_ATOMIC_API_COMMON_H

#include "../common/common.h"

namespace atomic_api_common {

/**
 * @brief Helper struct for performing generic operations with regards to the
 *        access target
 * @tparam target Access target
 */
template <sycl::target target>
struct target_map;

/**
 * @brief Helper struct for performing operations using a device target
 */
template <>
struct target_map<sycl::target::device> {
  static constexpr auto target = sycl::target::device;

  static constexpr auto addressSpace =
      sycl::access::address_space::global_space;

  /**
   * @brief Retrieves a global accessor
   * @tparam T Underlying type of the buffer
   * @param buf Buffer to get access to
   * @param cgh Command group handler
   * @return Global accessor
   */
  template <typename T>
  static sycl::accessor<T, 1, sycl::access_mode::atomic, target>
  get_accessor(sycl::buffer<T, 1> &buf, sycl::handler &cgh) {
    return buf.template get_access<sycl::access_mode::atomic, target>(cgh);
  }
};

/**
 * @brief Helper struct for performing operations using a local target
 */
template <>
struct target_map<sycl::target::local> {
  static constexpr auto target = sycl::target::local;

  static constexpr auto addressSpace =
      sycl::access::address_space::local_space;

  /**
   * @brief Retrieves a local accessor
   * @tparam T Underlying type of the buffer
   * @param cgh Command group handler
   * @return Local accessor
   */
  template <typename T>
  static sycl::accessor<T, 1, sycl::access_mode::atomic, target>
  get_accessor(sycl::buffer<T, 1> &, sycl::handler &cgh) {
    return sycl::accessor<T, 1, sycl::access_mode::atomic, target>(
        sycl::range<1>(1), cgh);
  }
};

/** Check atomic operations
 */
template <typename T, sycl::target target,
          template <class, sycl::target>
          class check_atomics_functor>
class check_atomics {
 public:
  void operator()(sycl_cts::util::logger &log, sycl::queue &testQueue) {
    auto testDevice = testQueue.get_device();

    T data = 0;
    std::memset(&data, 0xFF, sizeof(T));

    sycl::buffer<T, 1> buf(&data, sycl::range<1>(1));

    testQueue.submit([&](sycl::handler &cgh) {
      using functor = check_atomics_functor<T, target>;
      auto acc = target_map<target>::get_accessor(buf, cgh);
      auto f = functor(acc);
      cgh.single_task<functor>(f);
    });
  }
};

template <typename T, template <class, sycl::target>
                      class check_atomics_functor>
void generic_check_for_atomics(sycl_cts::util::logger &log,
                               sycl::queue testQueue) {
  /** Check atomics for sycl::target::device
   */
  check_atomics<T, sycl::target::device,
                check_atomics_functor>{}(log, testQueue);

  /** Check atomics for sycl::target::local
   */
  check_atomics<T, sycl::target::local, check_atomics_functor>{}(
      log, testQueue);
}

}  // namespace atomic_api_common

#endif  // SYCL_CTS_TESTS_ATOMIC_API_COMMON_H
