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

#ifndef SYCL_CTS_TESTS_ATOMIC_CONSTRUCTORS_COMMON_H
#define SYCL_CTS_TESTS_ATOMIC_CONSTRUCTORS_COMMON_H

#include "../common/common.h"

namespace atomic_constructors_common {

/** Check atomic constructors
 */
template <typename T, sycl::target target,
          sycl::access::address_space addressSpace>
class check_atomic_constructors {
  sycl::accessor<T, 1, sycl::access_mode::read_write, target> m_acc;

 public:
  check_atomic_constructors(
      sycl::accessor<T, 1, sycl::access_mode::read_write, target> acc)
      : m_acc(acc) {}

  void operator()() const {
    /** Check atomic constructor
     */
    sycl::atomic<T, addressSpace> a(m_acc.get_pointer());
  }
  void operator()(sycl::nd_item<1> item) const {
    /** Check atomic constructor
     */
    sycl::atomic<T, addressSpace> a(m_acc.get_pointer());
  }
};

/** Check atomic constructors
 */
template <typename T, sycl::target target>
class check_atomics {
 public:
  void operator()(sycl_cts::util::logger &log, sycl::queue &testQueue) {
    T data = 0;
    std::memset(&data, 0xFF, sizeof(T));

    sycl::buffer<T, 1> buf(&data, sycl::range<1>(1));

    /** Check atomic constructors
     */
    testQueue.submit([&](sycl::handler &cgh) {
      using functor =
          check_atomic_constructors<T, sycl::target::device,
                                    sycl::access::address_space::global_space>;
      sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::target::device>
          acc(buf, cgh);
      cgh.single_task<functor>(functor(acc));
    });
  }
};

/** Specialization for sycl::target::local
 */
template <typename T>
class check_atomics<T, sycl::target::local> {
 public:
  void operator()(sycl_cts::util::logger &log, sycl::queue &testQueue) {
    auto testDevice = testQueue.get_device();
    sycl::nd_range<1> nd_range(sycl::range<1>(1), sycl::range<1>(1));
    /** Check atomic constructors
     */
    testQueue.submit([&](sycl::handler &cgh) {
      using functor =
          check_atomic_constructors<T, sycl::target::local,
                                    sycl::access::address_space::local_space>;
      sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::target::local>
          acc(sycl::range<1>(1), cgh);
      cgh.parallel_for<functor>(nd_range, functor(acc));
    });
  }
};

}  // namespace atomic_constructors_common

#endif  // SYCL_CTS_TESTS_ATOMIC_CONSTRUCTORS_COMMON_H
