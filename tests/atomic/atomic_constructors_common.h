/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
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
#include <sstream>

namespace {

using namespace sycl_cts;

/** Check atomic constructors
*/
template <typename T, cl::sycl::access::target target,
          cl::sycl::access::address_space addressSpace>
class check_atomic_constructors {
  cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, target> m_acc;

 public:
  check_atomic_constructors(
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, target> acc)
      : m_acc(acc) {}

  void operator()() {
    /** Check atomic constructor
    */
    cl::sycl::atomic<T, addressSpace> a(m_acc.get_pointer());
  }
};

/** Check atomic constructors
*/
template <typename T, cl::sycl::access::target target>
class check_atomics {
 public:
  void operator()(util::logger &log, cl::sycl::queue &testQueue) {
    T data = 0;
    std::memset(&data, 0xFF, sizeof(T));

    cl::sycl::buffer<T, 1> buf(&data, cl::sycl::range<1>(1));

    /** Check atomic constructors
    */
    testQueue.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>
          acc(buf, cgh);

      check_atomic_constructors<T, cl::sycl::access::target::global_buffer,
                                cl::sycl::access::address_space::global_space>
          f(acc);

      cgh.single_task(f);
    });
  }
};

/** Specialization for cl::sycl::access::target::local
*/
template <typename T>
class check_atomics<T, cl::sycl::access::target::local> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &testQueue) {
    auto testDevice = testQueue.get_device();

    /** Check atomic constructors
    */
    testQueue.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          acc(cl::sycl::range<1>(1), cgh);

      check_atomic_constructors<T, cl::sycl::access::target::local,
                                cl::sycl::access::address_space::local_space>
          f(acc);

      cgh.single_task(f);
    });
  }
};

}  // namespace
