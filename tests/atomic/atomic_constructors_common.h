/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef SYCL_CTS_TESTS_ATOMIC_CONSTRUCTORS_COMMON_H
#define SYCL_CTS_TESTS_ATOMIC_CONSTRUCTORS_COMMON_H

#include "../common/common.h"

namespace atomic_constructors_common {

/** Check atomic constructors
 */
template <typename T, sycl::access::target target,
          sycl::access::address_space addressSpace>
class check_atomic_constructors {
  sycl::accessor<T, 1, sycl::access::mode::read_write, target> m_acc;

 public:
  check_atomic_constructors(
      sycl::accessor<T, 1, sycl::access::mode::read_write, target> acc)
      : m_acc(acc) {}

  void operator()() const {
    /** Check atomic constructor
     */
    sycl::atomic<T, addressSpace> a(m_acc.get_pointer());
  }
};

/** Check atomic constructors
 */
template <typename T, sycl::access::target target>
class check_atomics {
 public:
  void operator()(sycl_cts::util::logger &log, sycl::queue &testQueue) {
    T data = 0;
    std::memset(&data, 0xFF, sizeof(T));

    sycl::buffer<T, 1> buf(&data, sycl::range<1>(1));

    /** Check atomic constructors
     */
    testQueue.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access::mode::read_write,
                         sycl::access::target::global_buffer>
          acc(buf, cgh);

      check_atomic_constructors<T, sycl::access::target::global_buffer,
                                sycl::access::address_space::global_space>
          f(acc);

      cgh.single_task(f);
    });
  }
};

/** Specialization for sycl::access::target::local
 */
template <typename T>
class check_atomics<T, sycl::access::target::local> {
 public:
  void operator()(sycl_cts::util::logger &log, sycl::queue &testQueue) {
    auto testDevice = testQueue.get_device();

    /** Check atomic constructors
     */
    testQueue.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
          acc(sycl::range<1>(1), cgh);

      check_atomic_constructors<T, sycl::access::target::local,
                                sycl::access::address_space::local_space>
          f(acc);

      cgh.single_task(f);
    });
  }
};

}  // namespace atomic_constructors_common

#endif  // SYCL_CTS_TESTS_ATOMIC_CONSTRUCTORS_COMMON_H
