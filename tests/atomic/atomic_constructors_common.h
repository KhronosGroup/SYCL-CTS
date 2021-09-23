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
      sycl::accessor<T, 1, sycl::access_mode::read_write,
                         sycl::target::device>
          acc(buf, cgh);

      check_atomic_constructors<T, sycl::target::device,
                                sycl::access::address_space::global_space>
          f(acc);

      cgh.single_task(f);
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

    /** Check atomic constructors
     */
    testQueue.submit([&](sycl::handler &cgh) {
      sycl::accessor<T, 1, sycl::access_mode::read_write,
                         sycl::target::local>
          acc(sycl::range<1>(1), cgh);

      check_atomic_constructors<T, sycl::target::local,
                                sycl::access::address_space::local_space>
          f(acc);

      cgh.single_task(f);
    });
  }
};

}  // namespace atomic_constructors_common

#endif  // SYCL_CTS_TESTS_ATOMIC_CONSTRUCTORS_COMMON_H
