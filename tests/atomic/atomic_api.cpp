/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include <sstream>

#define TEST_NAME atomic_api

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** Check base atomic operations
*/
template <typename T, cl::sycl::access::target target,
          cl::sycl::access::address_space addressSpace>
class check_base_atomics {
  cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target> m_acc;

 public:
  check_base_atomics(
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    cl::sycl::memory_order order = cl::sycl::memory_order::relaxed;
    cl::sycl::atomic<T, addressSpace> a = m_acc[0];

    /** Check atomic member functions
    */
    T old = a.load(order);
    a.store(static_cast<T>(0), order);
    old = a.exchange(static_cast<T>(1), order);
    bool res = a.compare_exchange_strong(old, static_cast<T>(1), order, order);
    old = a.fetch_add(static_cast<T>(1), order);
    old = a.fetch_sub(static_cast<T>(1), order);

    /** Check atomic global functions
    */
    old = cl::sycl::atomic_load(&a, order);
    cl::sycl::atomic_store(&a, static_cast<T>(0), order);
    old = cl::sycl::atomic_exchange(&a, static_cast<T>(1), order);
    old = cl::sycl::atomic_compare_exchange_strong(&a, &old, static_cast<T>(1),
                                         order, order);
    old = cl::sycl::atomic_fetch_add(&a, static_cast<T>(1), order);
    old = cl::sycl::atomic_fetch_sub(&a, static_cast<T>(1), order);
  }
};

/** Specialisation for float because most operations don't permit float type
*/
template <cl::sycl::access::target target,
          cl::sycl::access::address_space addressSpace>
class check_base_atomics<float, target, addressSpace> {
  cl::sycl::accessor<float, 1, cl::sycl::access::mode::atomic, target> m_acc;

 public:
  check_base_atomics(
      cl::sycl::accessor<float, 1, cl::sycl::access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    cl::sycl::memory_order order = cl::sycl::memory_order::relaxed;

    /** Check atomic member functions
    */
    cl::sycl::atomic<float, addressSpace> a = m_acc[0];
    float old = a.load(order);
    a.store(0.f, order);
    old = a.exchange(1.f, order);

    /** Check atomic global functions
    */
    old = atomic_load(&a, order);
    atomic_store(&a, 0.f, order);
    old = atomic_exchange(&a, 1.f, order);
  }
};

/** Check extended atomic operations
*/
template <typename T, cl::sycl::access::target target,
          cl::sycl::access::address_space addressSpace>
class check_extended_atomics {
  cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target> m_acc;

 public:
  check_extended_atomics(
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    cl::sycl::memory_order order = cl::sycl::memory_order::relaxed;
    cl::sycl::atomic<T, addressSpace> a = m_acc[0];

    /** Check atomic member functions
    */
    T old = a.load(order);
    old = a.fetch_and(static_cast<T>(0xFFFFFFFF), order);
    old = a.fetch_or(static_cast<T>(0), order);
    old = a.fetch_xor(static_cast<T>(0xFFFFFFFE), order);
    old = a.fetch_min(static_cast<T>(0xFFFFFFFF), order);
    old = a.fetch_max(static_cast<T>(0), order);

    /** Check atomic global functions
    */
    old = atomic_load(&a, order);
    old = atomic_fetch_and(&a, static_cast<T>(0xFFFFFFFF), order);
    old = atomic_fetch_or(&a, static_cast<T>(0), order);
    old = atomic_fetch_xor(&a, static_cast<T>(0xFFFFFFFE), order);
    old = atomic_fetch_min(&a, static_cast<T>(0xFFFFFFFF), order);
    old = atomic_fetch_max(&a, static_cast<T>(0), order);
  }
};

/** Specialisation for float because most operations don't permit float type
*/
template <cl::sycl::access::target target,
          cl::sycl::access::address_space addressSpace>
class check_extended_atomics<float, target, addressSpace> {
  cl::sycl::accessor<float, 1, cl::sycl::access::mode::atomic, target> m_acc;

 public:
  check_extended_atomics(
      cl::sycl::accessor<float, 1, cl::sycl::access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    /** extended atomics are not supported for float
    */
  }
};

/** Check atomic operations
*/
template <typename T, cl::sycl::access::target target>
class check_atomics;

/** Specialisation for cl::sycl::access::target::global_buffer
*/
template <typename T>
class check_atomics<T, cl::sycl::access::target::global_buffer> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &testQueue) {
    auto testDevice = testQueue.get_device();

    T data = 0;
    std::memset(&data, 0xFF, sizeof(T));

    cl::sycl::buffer<T, 1> buf(&data, cl::sycl::range<1>(1));

    /** Check base atomics
    */
    if ((sizeof(T) <= 4) ||
        testDevice.has_extension("cl_khr_int64_base_atomics")) {
      testQueue.submit([&](cl::sycl::handler &cgh) {
        cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic,
                           cl::sycl::access::target::global_buffer>
            acc(buf, cgh);

        check_base_atomics<T, cl::sycl::access::target::global_buffer,
                           cl::sycl::access::address_space::global_space>
            f(acc);

        cgh.single_task(f);
      });
    }

    /** Check extended atomics
    */
    if ((sizeof(T) <= 4) ||
        testDevice.has_extension("cl_khr_int64_extended_atomics")) {
      testQueue.submit([&](cl::sycl::handler &cgh) {
        cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic,
                           cl::sycl::access::target::global_buffer>
            acc(buf, cgh);

        check_extended_atomics<T, cl::sycl::access::target::global_buffer,
                               cl::sycl::access::address_space::global_space>
            f(acc);

        cgh.single_task(f);
      });
    }
  }
};

/** Specialisation for cl::sycl::access::target::local
*/
template <typename T>
class check_atomics<T, cl::sycl::access::target::local> {
 public:
  void operator()(util::logger &log, cl::sycl::queue &testQueue) {
    auto testDevice = testQueue.get_device();

    /** Check base atomics
    */
    if ((sizeof(T) <= 4) ||
        testDevice.has_extension("cl_khr_int64_base_atomics")) {
      testQueue.submit([&](cl::sycl::handler &cgh) {
        cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic,
                           cl::sycl::access::target::local>
            acc(cl::sycl::range<1>(1), cgh);

        check_base_atomics<T, cl::sycl::access::target::local,
                           cl::sycl::access::address_space::local_space>
            f(acc);

        cgh.single_task(f);
      });
    }

    /** Check extended atomics
    */
    if ((sizeof(T) <= 4) ||
        testDevice.has_extension("cl_khr_int64_extended_atomics")) {
      testQueue.submit([&](cl::sycl::handler &cgh) {
        cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic,
                           cl::sycl::access::target::local>
            acc(cl::sycl::range<1>(1), cgh);

        check_extended_atomics<T, cl::sycl::access::target::local,
                               cl::sycl::access::address_space::local_space>
            f(acc);

        cgh.single_task(f);
      });
    }
  }
};

/** Check the api for cl::sycl::atomic
*/
class TEST_NAME : public util::test_base {
 public:
  /** Return information about this test
    */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void check_atomics_for_type(util::logger &log, cl::sycl::queue testQueue) {
    /** Check atomics for cl::sycl::access::target::global_buffer
    */
    check_atomics<T, cl::sycl::access::target::global_buffer>{}(log, testQueue);

    /** Check atomics for cl::sycl::access::target::local
    */
    check_atomics<T, cl::sycl::access::target::local>{}(log, testQueue);
  }

  /** Execute the test
    */
  void run(util::logger &log) override {
    try {
      auto testQueue = util::get_cts_object::queue();

      /** Check cl::sycl::memory_order
      */
      check_enum_class_value(cl::sycl::memory_order::relaxed);

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
