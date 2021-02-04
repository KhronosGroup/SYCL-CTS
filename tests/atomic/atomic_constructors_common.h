/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
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
