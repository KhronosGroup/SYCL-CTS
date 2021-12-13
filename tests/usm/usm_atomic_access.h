/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for USM atomic access tests
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_USM_ATOMIC_ACCESS_H
#define __SYCL_CTS_TEST_USM_ATOMIC_ACCESS_H

#include "../../util/usm_helper.h"
#include "../common/type_coverage.h"
#include <chrono>
#include <cstring>
#include <limits.h>
#include <thread>
#include <type_traits>

namespace usm_atomic_access {

static auto get_scalar_types() {
  static const auto scalar_types =
      named_type_pack<int, unsigned int, long,
                      unsigned long, float, double, long long,
                      unsigned long long>({"int", "unsigned int", "long",
                                           "unsigned long", "float", "double",
                                           "long long", "unsigned long long"});
  return scalar_types;
}

/** @brief Return atomic aspect depending on the type of allocated memory
 *  @retval AllocMemT USM allocation type
 */
template <sycl::usm::alloc AllocMemT>
auto get_atomic_memory() {
  if constexpr (AllocMemT == sycl::usm::alloc::host) {
    return sycl::aspect::usm_atomic_host_allocations;
  } else if constexpr (AllocMemT == sycl::usm::alloc::shared) {
    return sycl::aspect::usm_atomic_shared_allocations;
  } else {
    static_assert(AllocMemT != AllocMemT,
                  "This USM allocation type does not supported");
  }
}

constexpr bool with_atomic64 = true;
constexpr bool without_atomic64 = false;
/** @brief Dummy functor that use in type coverage
 *  @tparam CounterT Variable type from type coverage
 *  @tparam UseAtomic64Flag std::integral_constant type
 */
template <typename CounterT>
struct run_all_tests {
  void operator()(sycl::queue &queue, sycl_cts::util::logger &log, bool use_atomic_flag,
                  const std::string &typeName) {
    if constexpr ((sizeof(CounterT) * CHAR_BIT) == 64) {
      if (!use_atomic_flag) {
        return;
      }
      if (!queue.get_device().has(sycl::aspect::atomic64)) {
        log.note("Device does not perform 64-bit atomic operations");
        return;
      }
    } else {
      if (use_atomic_flag) {
        return;
      }
    }
    run_test_with_chosen_mem_type<CounterT>(queue, log, typeName);
  }
};

/** @brief Run test with two allocation types
 *  @tparam CounterT Variable type from type coverage
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 */
template <typename CounterT>
void run_test_with_chosen_mem_type(sycl::queue &queue,
                                   sycl_cts::util::logger &log,
                                   const std::string &typeName) {
  check_atomic_access<sycl::usm::alloc::host, CounterT>(queue, log, typeName);
  check_atomic_access<sycl::usm::alloc::shared, CounterT>(queue, log, typeName);
}

/** @brief Kernel for device side interaction with common memory
 */
template <sycl::usm::alloc AllocMemT, typename CounterT>
struct kernel;

/** @brief Check concurrent access from host and one device and no check
 *         concurrent access from multiple devices with
 *         "usm_atomic_shared_allocations" aspect from the same context
 *  @tparam AllocMemT USM allocation type
 *  @tparam CounterT Variable type from type coverage
 *  @param queue sycl::queue class object
 *  @param log sycl_cts::util::logger class object
 *  @param typeName a string representing the currently tested type
 */
template <sycl::usm::alloc AllocMemT, typename CounterT>
void check_atomic_access(sycl::queue &queue, sycl_cts::util::logger &log,
                         const std::string &typeName) {
  static_assert(AllocMemT != sycl::usm::alloc::device,
                "This test can't be launched on USM device memory");
  if (!queue.get_device().has(get_atomic_memory<AllocMemT>())) {
    log.note("Device does not support atomic access to the unified " +
             std::string(usm_helper::get_allocation_description<AllocMemT>()) +
             " memory allocation");
    return;
  }

  auto flag{usm_helper::allocate_usm_memory<AllocMemT, int>(queue)};
  auto counter{usm_helper::allocate_usm_memory<AllocMemT, CounterT>(queue)};

  int increment_value{1};
  // For stable test results we limited counter increase number to int16_t value
  // when using floating point numbers number test accuracy does not get worse
  constexpr size_t number_of_repetitions{15000};
  static_assert(
      number_of_repetitions == static_cast<CounterT>(number_of_repetitions),
      "number_of_repetitions variable value too great, please decrease it");
  auto flag_ptr{flag.get()};
  auto counter_ptr{counter.get()};
  // initialize with default value to avoid random issues
  *flag_ptr = 0;
  *counter_ptr = 0;

  using sycl::memory_order;
  using sycl::memory_scope;
  using sycl::access::address_space;

  queue.submit([=](sycl::handler &cgh) {
    cgh.single_task<kernel<AllocMemT, CounterT>>([=]() {
      sycl::atomic_ref<CounterT, memory_order::relaxed, memory_scope::device,
                       address_space::global_space>
          atomic_ref_dev_side(*counter_ptr);
      sycl::atomic_ref<int, memory_order::relaxed, memory_scope::device,
                       address_space::global_space>
          flag_atomic_ref_dev_side(*flag_ptr);
      while (!flag_atomic_ref_dev_side.load(memory_order::seq_cst))
        ;
      for (size_t i = 0; i < number_of_repetitions; i++) {
        atomic_ref_dev_side.fetch_add(increment_value, memory_order::seq_cst);
      }
    });
  });
  sycl::atomic_ref<int, memory_order::relaxed, memory_scope::device,
                   address_space::global_space>
      flag_atomic_ref_host_side(*flag_ptr);
  flag_atomic_ref_host_side.fetch_add(increment_value, memory_order::seq_cst);
  // With sycl::atomic_ref we must use scalar data types.
  // This is mentioned in SYCL2020 rev. 3 spec (section #4.15.3. Atomic
  // references)
  sycl::atomic_ref<CounterT, memory_order::relaxed, memory_scope::device,
                   address_space::global_space>
      atomic_ref_host_side(*counter_ptr);
  for (size_t i = 0; i < number_of_repetitions; i++) {
    atomic_ref_host_side.fetch_add(increment_value, memory_order::seq_cst);
    // As it's more likely that the execution of the cycle will start first on
    // the host, we provoke a longer execution of the cycle on the host to
    // increase the chances of simultaneous overwriting of the variable value
    auto unblocking_time{std::chrono::high_resolution_clock::now() +
                         std::chrono::microseconds(20)};
    do {
      std::this_thread::yield();
    } while (std::chrono::high_resolution_clock::now() < unblocking_time);
  }
  queue.wait();
  if (!CHECK_VALUE_SCALAR(log, *counter_ptr,
                          2 * (number_of_repetitions * increment_value))) {
    log.note("Test for the USM " +
             std::string(usm_helper::get_allocation_description<AllocMemT>()) +
             " memory allocation failed for \"" + typeName +
             "\" underlying type");
  }
}

}  // namespace usm_atomic_access

#endif  // __SYCL_CTS_TEST_USM_ATOMIC_ACCESS_H
