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

#define TEST_NAME atomic_api_64_extended

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** Check extended atomic operations
*/
template <typename T, cl::sycl::access::target target>
class check_extended_atomics {
  cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target> m_acc;

 public:
  check_extended_atomics(
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() {
    static constexpr auto addressSpace = target_map<target>::addressSpace;
    cl::sycl::memory_order order = cl::sycl::memory_order::relaxed;
    cl::sycl::atomic<T, addressSpace> a = m_acc[0];

    /** Check atomic member functions with default order
    */
    T old = a.load();
    (void)old;
    old = a.fetch_and(T{0xFFFFFFFF});
    old = a.fetch_or(T{0});
    old = a.fetch_xor(T{0xFFFFFFFE});
    old = a.fetch_min(T{0xFFFFFFFF});
    old = a.fetch_max(T{0});

    /** Check atomic member functions
    */
    old = a.load(order);
    old = a.fetch_and(T{0xFFFFFFFF}, order);
    old = a.fetch_or(T{0}, order);
    old = a.fetch_xor(T{0xFFFFFFFE}, order);
    old = a.fetch_min(T{0xFFFFFFFF}, order);
    old = a.fetch_max(T{0}, order);

    /** Check atomic global functions with default order
    */
    old = cl::sycl::atomic_load(a);
    old = cl::sycl::atomic_fetch_and(a, T{0xFFFFFFFF});
    old = cl::sycl::atomic_fetch_or(a, T{0});
    old = cl::sycl::atomic_fetch_xor(a, T{0xFFFFFFFE});
    old = cl::sycl::atomic_fetch_min(a, T{0xFFFFFFFF});
    old = cl::sycl::atomic_fetch_max(a, T{0});

    /** Check atomic global functions
    */
    old = cl::sycl::atomic_load(a, order);
    old = cl::sycl::atomic_fetch_and(a, T{0xFFFFFFFF}, order);
    old = cl::sycl::atomic_fetch_or(a, T{0}, order);
    old = cl::sycl::atomic_fetch_xor(a, T{0xFFFFFFFE}, order);
    old = cl::sycl::atomic_fetch_min(a, T{0xFFFFFFFF}, order);
    old = cl::sycl::atomic_fetch_max(a, T{0}, order);
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

  template <typename T,
            typename std::enable_if<sizeof(T) == 8>::type * = nullptr>
  void check_atomics_for_type(util::logger &log, cl::sycl::queue testQueue) {
    return generic_check_for_atomics<T, check_extended_atomics>(log, testQueue);
  }

  template <typename T,
            typename std::enable_if<sizeof(T) != 8>::type * = nullptr>
  void check_atomics_for_type(util::logger &log, cl::sycl::queue testQueue) {
    // generic_check_for_atomics assumes a 64-bit T type.
    return;
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
      if (testDevice.has_extension("cl_khr_int64_extended_atomics")) {
        check_atomics_for_type<long>(log, testQueue);
        check_atomics_for_type<unsigned long>(log, testQueue);
        check_atomics_for_type<long long>(log, testQueue);
        check_atomics_for_type<unsigned long long>(log, testQueue);
      }

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
