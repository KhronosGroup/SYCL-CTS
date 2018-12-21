/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "atomic_api_common.h"
#include <sstream>

#define TEST_NAME atomic_api_32

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** Check base atomic operations
*/
template <typename T, cl::sycl::access::target target>
class check_atomics_32 {
  cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target> m_acc;

 public:
  check_atomics_32(
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    static constexpr auto addressSpace = target_map<target>::addressSpace;
    cl::sycl::memory_order order = cl::sycl::memory_order::relaxed;
    cl::sycl::atomic<T, addressSpace> a = m_acc[0];

    /** Check atomic member functions with default order
    */
    T old = a.load();
    a.store(T{0});
    old = a.exchange(T{1});
    bool res = a.compare_exchange_strong(old, T{1});
    old = a.fetch_add(T{1});
    old = a.fetch_sub(T{1});
    old = a.fetch_and(T{1});
    old = a.fetch_or(T{1});
    old = a.fetch_xor(T{1});
    old = a.fetch_min(T{1});
    old = a.fetch_max(T{1});

    /** Check atomic member functions
    */
    old = a.load(order);
    a.store(T{0}, order);
    old = a.exchange(T{1}, order);
    res = a.compare_exchange_strong(old, T{1}, order, order);
    old = a.fetch_add(T{1}, order);
    old = a.fetch_sub(T{1}, order);
    old = a.fetch_and(T{1}, order);
    old = a.fetch_or(T{1}, order);
    old = a.fetch_xor(T{1}, order);
    old = a.fetch_min(T{1}, order);
    old = a.fetch_max(T{1}, order);

    /** Check atomic global functions with default order
    */
    old = cl::sycl::atomic_load(a);
    cl::sycl::atomic_store(a, T{0});
    old = cl::sycl::atomic_exchange(a, T{1});
    old = cl::sycl::atomic_compare_exchange_strong(a, old, T{1});
    old = cl::sycl::atomic_fetch_add(a, T{1});
    old = cl::sycl::atomic_fetch_sub(a, T{1});
    old = cl::sycl::atomic_fetch_and(a, T{1});
    old = cl::sycl::atomic_fetch_or(a, T{1});
    old = cl::sycl::atomic_fetch_xor(a, T{1});
    old = cl::sycl::atomic_fetch_min(a, T{1});
    old = cl::sycl::atomic_fetch_max(a, T{1});

    /** Check atomic global functions
    */
    old = cl::sycl::atomic_load(a, order);
    cl::sycl::atomic_store(a, T{0}, order);
    old = cl::sycl::atomic_exchange(a, T{1}, order);
    old = cl::sycl::atomic_compare_exchange_strong(a, old, T{1}, order, order);
    old = cl::sycl::atomic_fetch_add(a, T{1}, order);
    old = cl::sycl::atomic_fetch_sub(a, T{1}, order);
    old = cl::sycl::atomic_fetch_and(a, T{1}, order);
    old = cl::sycl::atomic_fetch_or(a, T{1}, order);
    old = cl::sycl::atomic_fetch_xor(a, T{1}, order);
    old = cl::sycl::atomic_fetch_min(a, T{1}, order);
    old = cl::sycl::atomic_fetch_max(a, T{1}, order);
  }
};

/** Specialization for float because most operations don't permit float type
*/
template <cl::sycl::access::target target>
class check_atomics_32<float, target> {
  cl::sycl::accessor<float, 1, cl::sycl::access::mode::atomic, target> m_acc;

 public:
  check_atomics_32(
      cl::sycl::accessor<float, 1, cl::sycl::access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    static constexpr auto addressSpace = target_map<target>::addressSpace;
    cl::sycl::memory_order order = cl::sycl::memory_order::relaxed;
    cl::sycl::atomic<float, addressSpace> a = m_acc[0];

    /** Check atomic member functions with default order
    */
    float old = a.load();
    a.store(0.f);
    old = a.exchange(1.f);

    /** Check atomic member functions
    */
    old = a.load(order);
    a.store(0.f, order);
    old = a.exchange(1.f, order);

    /** Check atomic global functions with default order
    */
    old = cl::sycl::atomic_load(a);
    cl::sycl::atomic_store(a, 0.f);
    old = cl::sycl::atomic_exchange(a, 1.f);

    /** Check atomic global functions
    */
    old = cl::sycl::atomic_load(a, order);
    cl::sycl::atomic_store(a, 0.f, order);
    old = cl::sycl::atomic_exchange(a, 1.f, order);
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
    return generic_check_for_atomics<T, check_atomics_32>(log, testQueue);
  }

  /** Execute the test
    */
  void run(util::logger &log) override {
    try {
      auto testQueue = util::get_cts_object::queue();
      auto testDevice = testQueue.get_device();

      /** Check cl::sycl::memory_order
      */
      check_enum_class_value(cl::sycl::memory_order::relaxed);

      /** Check atomics for supported types
      */
      check_atomics_for_type<int>(log, testQueue);
      check_atomics_for_type<unsigned int>(log, testQueue);
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
