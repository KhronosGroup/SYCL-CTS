/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include <sstream>

#define TEST_NAME atomic_constructors

namespace TEST_NAMESPACE {

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

/** Specialisation for cl::sycl::access::target::local
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

/** Check the api for cl::sycl::atomic
*/
class TEST_NAME : public util::test_base {
 public:
  /** Return information about this test
  */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void check_atomics_for_type(util::logger &log, cl::sycl::queue testQueue) {
    /** Check atomic constructors for cl::sycl::access::target::global_buffer
    */
    check_atomics<T, cl::sycl::access::target::global_buffer>{}(log, testQueue);

    /** Check atomic constructors for cl::sycl::access::target::local
    */
    check_atomics<T, cl::sycl::access::target::local>{}(log, testQueue);
  }

  /** Execute the test
  */
  virtual void run(util::logger &log) override {
    try {
      auto testQueue = util::get_cts_object::queue();

      /** Check atomics for supported types
      */
      check_atomics_for_type<int>(log, testQueue);
      check_atomics_for_type<unsigned int>(log, testQueue);
      check_atomics_for_type<long>(log, testQueue);
      check_atomics_for_type<unsigned long>(log, testQueue);
      check_atomics_for_type<long long>(log, testQueue);
      check_atomics_for_type<unsigned long long>(log, testQueue);
      check_atomics_for_type<float>(log, testQueue);

      testQueue.wait_and_throw();

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// Construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
