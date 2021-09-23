/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "atomic_api_common.h"

#include <climits>
#include <string>

#define TEST_NAME atomic_api_64_extended

namespace TEST_NAMESPACE {

using namespace atomic_api_common;
using namespace sycl_cts;

/** Check extended atomic operations
 */
template <typename T, sycl::target target>
class check_extended_atomics {
  sycl::accessor<T, 1, sycl::access_mode::atomic, target> m_acc;

 public:
  check_extended_atomics(
      sycl::accessor<T, 1, sycl::access_mode::atomic, target> acc)
      : m_acc(acc) {}

  void operator()() const {
    static constexpr auto addressSpace = target_map<target>::addressSpace;
    sycl::memory_order order = sycl::memory_order::relaxed;
    sycl::atomic<T, addressSpace> a = m_acc[0];

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
    old = sycl::atomic_load(a);
    old = sycl::atomic_fetch_and(a, T{0xFFFFFFFF});
    old = sycl::atomic_fetch_or(a, T{0});
    old = sycl::atomic_fetch_xor(a, T{0xFFFFFFFE});
    old = sycl::atomic_fetch_min(a, T{0xFFFFFFFF});
    old = sycl::atomic_fetch_max(a, T{0});

    /** Check atomic global functions
     */
    old = sycl::atomic_load(a, order);
    old = sycl::atomic_fetch_and(a, T{0xFFFFFFFF}, order);
    old = sycl::atomic_fetch_or(a, T{0}, order);
    old = sycl::atomic_fetch_xor(a, T{0xFFFFFFFE}, order);
    old = sycl::atomic_fetch_min(a, T{0xFFFFFFFF}, order);
    old = sycl::atomic_fetch_max(a, T{0}, order);
  }
};

/** Check the api for sycl::atomic
 */
class TEST_NAME : public util::test_base {
 public:
  /** Return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  void check_atomics_for_type(util::logger &log, sycl::queue testQueue) {
    return generic_check_for_atomics<T, check_extended_atomics>(log, testQueue);
  }

  /** Execute the test
   */
  void run(util::logger &log) override {
    auto testQueue = util::get_cts_object::queue();
    auto testDevice = testQueue.get_device();

    /** Check sycl::memory_order
     */
    check_enum_class_value(sycl::memory_order::relaxed);

    /** Check atomics for supported types
     */
    if (testDevice.has(sycl::aspect::atomic64)) {
      if constexpr (sizeof(long) * CHAR_BIT == 64 /*bits*/) {
        check_atomics_for_type<long>(log, testQueue);
        check_atomics_for_type<unsigned long>(log, testQueue);
      }
      check_atomics_for_type<long long>(log, testQueue);
      check_atomics_for_type<unsigned long long>(log, testQueue);
    }

    /** Check sycl::memory_order
     */
    check_enum_class_value(sycl::memory_order::relaxed);

    /** Check atomics for supported types
     */
    if (testDevice.has_extension("cl_khr_int64_extended_atomics")) {
      if constexpr (sizeof(long) * CHAR_BIT == 64 /*bits*/) {
        check_atomics_for_type<long>(log, testQueue);
        check_atomics_for_type<unsigned long>(log, testQueue);
      }
      check_atomics_for_type<long long>(log, testQueue);
      check_atomics_for_type<unsigned long long>(log, testQueue);
    }

    testQueue.wait_and_throw();
  }
};

// Construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
