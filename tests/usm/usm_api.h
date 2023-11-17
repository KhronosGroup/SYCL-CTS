/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for USM API tests
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_USM_USM_API_H
#define __SYCL_CTS_TEST_USM_USM_API_H

#include "../../util/exceptions.h"
#include "../../util/usm_helper.h"
#include "../common/common.h"
#include "../common/type_coverage.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace usm_api {

/** @brief Allocation type to use for tests, including non-USM allocation type
 */
enum class allocation : int { non_usm, host, device, shared };

/** @brief Maps internal usm_api::allocation type to the USM allocation type
 */
template <allocation alloc>
constexpr sycl::usm::alloc map_usm_allocation() {
  if constexpr (alloc == allocation::host) {
    return sycl::usm::alloc::host;
  } else if constexpr (alloc == allocation::device) {
    return sycl::usm::alloc::device;
  } else if constexpr (alloc == allocation::shared) {
    return sycl::usm::alloc::shared;
  } else {
    static_assert(alloc != alloc, "Unknown allocation");
  }
}

/** @brief Provides description string for the given allocation type
 */
template <allocation alloc>
std::string get_allocation_decription() {
  if constexpr (alloc == allocation::non_usm) {
    return "non-USM host pointer";
  } else {
    constexpr auto kind = map_usm_allocation<alloc>();
    return std::string(usm_helper::get_allocation_description<kind>()) +
           " allocation";
  }
}

/** @brief Verify device support requirement for the specific allocation case
 */
template <allocation alloc>
bool check_device_support(sycl_cts::util::logger &log,
                          const sycl::queue &queue) {
  if constexpr (alloc != allocation::non_usm) {
    // Only USM allocations require specific aspects to work
    constexpr auto kind = map_usm_allocation<alloc>();
    const auto &device = queue.get_device();

    if (!device.has(usm_helper::get_aspect<kind>())) {
      std::string message =
          "Device does not support " + get_allocation_decription<alloc>();
      log.note(message);
      return false;
    }
  }
  return true;
}

/** @brief Syntactic sugar for the number of events
 */
constexpr size_t operator"" _events(unsigned long long value) {
  if (value > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Too big number of events used");
  }
  return static_cast<size_t>(value);
}

/** @brief Encapsulates different callers to use within tests
 */
namespace caller {

/** @brief Caller to use for the queue member functions' tests
 */
struct queue {
  using type = sycl::queue;

  template <typename actionT>
  static void submit(sycl::queue &queue, actionT action) {
    action(queue);
  }
};

/** @brief Caller to use for the handler member functions' tests
 */
struct handler {
  using type = sycl::handler;

  template <typename actionT>
  static void submit(sycl::queue &queue, actionT action) {
    queue.submit([&](sycl::handler &cgh) { action(cgh); });
  }
};

}  // namespace caller

/** @brief Kernel for the device-side initialization of USM data
 */
template <typename, size_t, allocation>
struct kernel_storage_initialization {};

/** @brief Kernel for the device-side copy of USM data to the host
 */
template <typename, size_t, allocation>
struct kernel_storage_copy {};

/** @brief Storage helper for USM and non-USM pointers
 *  @tparam T Underlying data type
 *  @tparam count Size of allocation given in number of items
 *  @tparam alloc Allocation type
 */
template <typename T, size_t count, allocation alloc>
struct storage {
  /** @brief Allocate the USM or non-USM memory with the appropriate deleter
   */
  static auto get(sycl::queue &queue) {
    if constexpr (alloc == allocation::non_usm) {
      return std::make_unique<T[]>(count);
    } else {
      constexpr auto kind = map_usm_allocation<alloc>();
      return usm_helper::allocate_usm_memory<kind, T>(queue, count);
    }
  }

  /** @brief Allocate memory and initialize it with the value given
   */
  static auto get(sycl::queue &queue, T initialValue) {
    auto storage = get(queue);
    T *ptr = storage.get();

    if constexpr (alloc == allocation::non_usm) {
      std::fill_n(ptr, count, initialValue);
    } else {
      using kernel_name = kernel_storage_initialization<T, count, alloc>;
      const sycl::range<1> range{count};

      queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<kernel_name>(
            range, [=](sycl::id<1> idx) { ptr[idx] = initialValue; });
      });
      queue.wait_and_throw();
    }

    return storage;
  }

  /** @brief Type of allocation the get() member function will return
   */
  using type = decltype(get(std::declval<sycl::queue &>()));

  /** @brief Copy values from the USM allocation to the host-side array
   */
  static auto copy_values(sycl::queue &queue, const type &instance) {
    std::array<T, count> values;
    const T *ptr = instance.get();

    if constexpr (alloc != allocation::device) {
      std::copy_n(ptr, count, values.begin());
    } else {
      const sycl::range<1> range(count);
      sycl::buffer<T, 1> buf(values.data(), range);

      queue.submit([&](sycl::handler &cgh) {
        auto acc = buf.template get_access<sycl::access_mode::write>(cgh);
        cgh.parallel_for<kernel_storage_copy<T, count, alloc>>(
            range, [=](sycl::id<1> idx) {
              const size_t linearIndex = idx[0];
              acc[idx] = ptr[linearIndex];
            });
      });
      // buffer destructor waits for the kernel submitted
    }
    return values;
  }
};

/** @brief Encapsulates the member function-specific logic, for example logic
 *         specific to the memcpy() and memset() tests
 */
namespace tests {

/** @brief Encapsulates the implementation details
 */
namespace detail {

/** @brief Provides verification for the given pointer data against the given
 *         reference values
 */
template <size_t count, typename T, class ReferenceIt>
void check_values(const T *begin, const ReferenceIt &reference) {
  const auto end = begin + count;
  auto diff = std::mismatch(begin, end, reference);
  if (diff.first != end) {
    std::string message{"Mismatch on "};
    message += std::to_string(end - begin);
    message += ": retrieved " + std::to_string(*diff.first);
    message += ", expected " + std::to_string(*diff.second);
    throw sycl_cts::util::fail_check(message);
  }
}

/** @brief Trait for tests that have no additional device requirements
 *
 * Tests implementing this trait do not require any device support beyond
 * the destination memory location.
 */
struct noAdditionalDeviceRequirements {
  static bool supports_device(sycl_cts::util::logger&, const sycl::queue&) {
    return true;
  }
};

/** @brief Trait for tests that require support for a given USM alloc type
 *
 * Tests implementing this trait require device support for a specific
 * type of allocation (e.g. as a source memory location).
 */
template <allocation allocationType>
struct requiresUsmAllocationSupport {
  static bool supports_device(sycl_cts::util::logger& log,
                              const sycl::queue& queue) {
    return check_device_support<allocationType>(log, queue);
  }
};

/** @brief Provides generic copy test logic for memcpy() and copy() tests
 *  @tparam sourceAllocation Allocation type to use for data source
 */
template <typename T, size_t count, allocation sourceAllocation>
class copyGeneric : public requiresUsmAllocationSupport<sourceAllocation> {
 protected:
  using storage_t = storage<T, count, sourceAllocation>;
  const typename storage_t::type source;

  std::array<T, count> reference;

 public:
  copyGeneric(sycl::queue &queue, int seed)
      : source(storage_t::get(queue, T{seed})) {
    auto values = storage_t::copy_values(queue, source);
    reference.swap(values);
  }

  void check(const T *ptr) { check_values<count>(ptr, reference.begin()); }
};

/** @brief Provides generic test logic for the copy() member function tests
 */
template <typename T, size_t count, allocation sourceAllocation>
class copy : public copyGeneric<T, count, sourceAllocation> {
  using baseT = copyGeneric<T, count, sourceAllocation>;

 public:
  copy(sycl::queue &queue, int seed) : baseT(queue, seed) {}

  /** @brief The copy() member function supports non-USM pointers
   */
  static constexpr bool has_non_usm_support() { return true; }

  template <allocation alloc>
  static std::string description() {
    return "copy from " + get_allocation_decription<sourceAllocation>() +
           " to " + get_allocation_decription<alloc>();
  }

  template <typename parentT, typename... depEventsT>
  auto call(parentT &parent, T *ptr, depEventsT &&... events) const {
    return parent.copy(this->source.get(), ptr, count, events...);
  }
};

/** @brief Provides generic test logic for the memcpy() member function tests
 */
template <typename T, size_t count, allocation sourceAllocation>
class memcpy : public copyGeneric<T, count, sourceAllocation> {
  using baseT = copyGeneric<T, count, sourceAllocation>;

 public:
  memcpy(sycl::queue &queue, int seed) : baseT(queue, seed) {}

  /** @brief The memcpy() member function supports non-USM pointers
   */
  static constexpr bool has_non_usm_support() { return true; }

  template <allocation alloc>
  static std::string description() {
    return "memcpy from " + get_allocation_decription<sourceAllocation>() +
           " to " + get_allocation_decription<alloc>();
  }

  template <typename parentT, typename... depEventsT>
  auto call(parentT &parent, T *ptr, depEventsT &&... events) const {
    constexpr auto size = sizeof(T) * count;
    return parent.memcpy(ptr, this->source.get(), size,
                         std::forward<depEventsT>(events)...);
  }
};
}  // namespace detail

template <typename T, size_t count>
using copy_from_non_usm = detail::copy<T, count, allocation::non_usm>;
template <typename T, size_t count>
using copy_from_host = detail::copy<T, count, allocation::host>;
template <typename T, size_t count>
using copy_from_device = detail::copy<T, count, allocation::device>;
template <typename T, size_t count>
using copy_from_shared = detail::copy<T, count, allocation::shared>;

template <typename T, size_t count>
using memcpy_from_non_usm = detail::memcpy<T, count, allocation::non_usm>;
template <typename T, size_t count>
using memcpy_from_host = detail::memcpy<T, count, allocation::host>;
template <typename T, size_t count>
using memcpy_from_device = detail::memcpy<T, count, allocation::device>;
template <typename T, size_t count>
using memcpy_from_shared = detail::memcpy<T, count, allocation::shared>;

/** @brief Provides test logic for the fill() member function tests
 */
template <typename T, size_t count>
class fill : public detail::noAdditionalDeviceRequirements {
  const T value;

 public:
  fill(sycl::queue &, int seed) : value{seed} {}

  /** @brief The fill() member function doesn't support non-USM pointers
   */
  static constexpr bool has_non_usm_support() { return false; }

  template <allocation alloc>
  static std::string description() {
    return "fill using " + get_allocation_decription<alloc>();
  }

  template <typename parentT, typename... depEventsT>
  auto call(parentT &parent, T *ptr, depEventsT &&... events) const {
    return parent.fill(ptr, value, count, std::forward<depEventsT>(events)...);
  }

  void check(const T *ptr) const {
    T reference[count];
    std::fill_n(reference, count, value);

    detail::check_values<count>(ptr, reference);
  }
};

/** @brief Provides test logic for the memset() member function tests
 */
template <typename T, size_t count>
class memset : public detail::noAdditionalDeviceRequirements {
  const int value;

 public:
  static constexpr size_t size = count * sizeof(T);

  memset(sycl::queue &, int seed) : value(seed) {}

  /** @brief The memset() member function doesn't support non-USM pointers
   */
  static constexpr bool has_non_usm_support() { return false; }

  template <allocation alloc>
  static std::string description() {
    return "memset using " + get_allocation_decription<alloc>();
  }

  template <typename parentT, typename... depEventsT>
  auto call(parentT &parent, T *ptr, depEventsT &&... events) const {
    return parent.memset(ptr, value, size, std::forward<depEventsT>(events)...);
  }

  void check(const T *ptr) const {
    T reference[count];
    std::memset(reference, value, size);

    detail::check_values<count>(ptr, reference);
  }
};

/** @brief Provides test logic for the prefetch() member function tests
 */
template <typename T, size_t count>
class prefetch : public detail::noAdditionalDeviceRequirements {
 public:
  static constexpr size_t size = count * sizeof(T);

  prefetch(sycl::queue &, int) {}

  /** @brief The prefetch() member function doesn't support non-USM pointers
   */
  static constexpr bool has_non_usm_support() { return false; }

  template <allocation alloc>
  static std::string description() {
    return "prefetch using " + get_allocation_decription<alloc>();
  }

  template <typename parentT, typename... depEventsT>
  auto call(parentT &parent, T *ptr, depEventsT &&... events) const {
    return parent.prefetch(ptr, size, std::forward<depEventsT>(events)...);
  }

  void check(const T *) const {
    // Results are implementation-defined
  }
};

/** @brief Provides test logic for the mem_advise() member function tests
 */
template <typename T, size_t count>
class mem_advise : public detail::noAdditionalDeviceRequirements {
 public:
  static constexpr size_t size = count * sizeof(T);

  mem_advise(sycl::queue &, int) {}

  /** @brief The mem_advise() member function doesn't support non-USM pointers
   */
  static constexpr bool has_non_usm_support() { return false; }

  template <allocation alloc>
  static std::string description() {
    return "mem_advise using " + get_allocation_decription<alloc>();
  }

  template <typename parentT, typename... depEventsT>
  auto call(parentT &parent, T *ptr, depEventsT &&... events) const {
    const int advice = 0;  // Reset to defaults according to the SYCL 2020 spec
    return parent.mem_advise(ptr, size, advice,
                             std::forward<depEventsT>(events)...);
  }

  void check(const T *) const {
    // Effects are implementation-defined
  }
};

}  // namespace tests

/** @brief Kernel name for event_generator::init() member function
 */
template <typename T, size_t gen_buf_size>
struct init_kernel_name;

/** @brief Kernel name for event_generator::copy_arrays() member function
 */
template <typename T, size_t gen_buf_size>
struct copy_arrays_kernel_name;

/** @brief The number of events passed to the function being checked.
 */
constexpr auto multiple_events = 10_events;

/** @brief Helper class providing number of elements for generators buffers if
 *         numEvents > 0
 */
template <size_t numEvents>
struct gen_buf_size_selector {
  static constexpr size_t value = 10000;
};

/** @brief Helper class specialization providing number of elements for
 *         generators buffers if numEvents == 0
 */
template <>
struct gen_buf_size_selector<0> {
  static constexpr size_t value = 0;
};

/** @brief Helper class for generation of 'sycl::event's by calling
 *         queue::submit with long operation in kernel execution.
 *  @tparam buf_size Size of arrays to process
 */
template <typename T, size_t buf_size>
class event_generator {
  // Change vector<bool> to vector <unsigned char>, since vector<bool> is
  // stored bit-wise, and does not have data() member function.
  using ContainType =
      typename std::conditional_t<std::is_same_v<bool, T>, unsigned char, T>;
  sycl::range<1> rng{buf_size};
  std::vector<ContainType> arr_src;
  std::vector<ContainType> arr_dst;
  sycl::buffer<ContainType, 1> buf_src;
  sycl::buffer<ContainType, 1> buf_dst;

 public:
  event_generator()
      : rng{buf_size},
        arr_src(buf_size, ContainType{0}),
        arr_dst(buf_size, ContainType{0}),
        buf_src{arr_src.data(), rng},
        buf_dst{arr_dst.data(), rng} {}

  /** @brief Initialize arr_src with init_value and return sycl::event of
   *         queue::submit()
   *  @param value Some non-zero (non-default) value of type ContainType
   */
  sycl::event init(sycl::queue& queue, ContainType value) {
    return queue.submit([&](sycl::handler &cgh) {
      auto acc_src = buf_src.template get_access<sycl::access_mode::write>(cgh);
      // single_task is used to make process long enough for testing purpose
      // The function being tested must wait for this task to complete.
      cgh.single_task<init_kernel_name<ContainType, buf_size>>([=] {
        for (size_t i = 0; i < buf_size; ++i) {
          acc_src[i] = value;
        }
      });
    });
  }

  /** @brief Copy data from arr_src to arr_dst for future check
   */
  void copy_arrays(sycl::queue &queue) {
    queue.submit([&](sycl::handler& cgh) {
      using kernel_name = copy_arrays_kernel_name<ContainType, buf_size>;
      auto acc_src = buf_src.template get_access<sycl::access_mode::read>(cgh);
      auto acc_dst = buf_dst.template get_access<sycl::access_mode::write>(cgh);
      // Copy should be much faster then algorithm in 'init()' member function
      // to detect situation when tested function doesn't wait for events
      // provided as arguments
      cgh.parallel_for<kernel_name>(rng, [=](sycl::id<1> idx) {
        const size_t i = idx[0];
        acc_dst[i] = acc_src[i];
      });
    });
  }

  /** @brief Check that elements of arrays are equal to each other and to
   *         value
   *  @param value The same value as value passed to init() member function
   */
  bool check(ContainType value) {
    bool result = true;
    auto acc_src = buf_src.template get_access<sycl::access_mode::read>(rng);
    auto acc_dst = buf_dst.template get_access<sycl::access_mode::read>(rng);
    for (size_t i = buf_size - 1; i + 1 > 0; --i) {
      result = result && (acc_src[i] == acc_dst[i]);
      result = result && (acc_dst[i] == value);
    }
    return result;
  }
};

/** @brief Provides the root test logic for every member function, caller,
 *         number of events and allocation type
 *  @tparam count Size of the allocation, in number of items
 *  @tparam alloc Allocation to use for the pointer. In case the member function
 *          we should test requires more than a single pointer allocation, this
 *          parameter defines allocation to use for the pointer we will use to
 *          check the member function results (target pointer for the copy
 *          operations for example)
 */
template <typename T, size_t count, template <typename, size_t> class checkT,
          typename caller, allocation alloc, size_t numEvents>
struct test {
  static void run(sycl_cts::util::logger &log) {
    using verifier_t = checkT<T, count>;
    using storage_t = storage<T, count, alloc>;
    using events_t = std::vector<sycl::event>;
    // Elements to process in event_generator. Large number is selected to make
    // kernel execution longer if numEvents > 0
    constexpr size_t gen_buf_size = gen_buf_size_selector<numEvents>::value;
    using gen_arr_t = std::array<event_generator<T, gen_buf_size>, numEvents>;

    log.debug([&] {
      std::string message("Running test for ");
      message += verifier_t::template description<alloc>() + " for ";
      message += std::string(typeid(T).name()) + " ...";
      return message;
    });

    try {
      auto queue = sycl_cts::util::get_cts_object::queue();

      if (!verifier_t::supports_device(log, queue)) return;
      if (!check_device_support<alloc>(log, queue)) return;

      log.debug("... allocate USM memory storage");

      auto scopedStorage = storage_t::get(queue, T{0});
      const auto ptr = scopedStorage.get();

      log.debug("... prepare specific verifier instance");
      verifier_t verifier(queue, 1);

      // Event generators array
      gen_arr_t gens{};
      // Non-zero (non default) value to initialize event generators source
      // buffers
      T init_value{1};

      auto action = [&](typename caller::type &parent) {
        if constexpr (numEvents == 0) {
          log.debug("... submit USM operation with 0 events");
          verifier.call(parent, ptr);
        } else {
          static_assert(std::is_same_v<typename caller::type, sycl::queue>,
                        "Tests with events are not acceptible for handler");
          // List of events that tested usm operation should wait
          events_t depEvents(numEvents);
          // Run time-consuming tasks to generate events
          for (size_t i = 0; i < depEvents.size(); ++i) {
            depEvents[i] = gens[i].init(parent, init_value);
          }

          if constexpr (numEvents == 1) {
            auto event = verifier.call(parent, ptr, depEvents[0]);
            event.wait();
          } else {
            auto event = verifier.call(parent, ptr, depEvents);
            event.wait();
          }

          // Fast copy arrays for future check. If tested function does not
          // wait for the passed events to complete, then an incompletely
          // initialized 'src_arr' will be copied to 'dst_arr' and the test
          // will fail during validation. Reverse order is used to increase
          // probability of data race.
          for (size_t i = numEvents - 1; i + 1 > 0; --i) {
            gens[i].copy_arrays(parent);
          }
        }
      };

      caller::submit(queue, action);

      log.debug("... wait for all actions submitted");
      queue.wait_and_throw();  // avoid data race

      if constexpr (alloc != allocation::device) {
        // we can access the pointer from the host side
        log.debug("... verify USM operation results");
        verifier.check(ptr);
      } else {
        log.debug("... retrieve USM operation results from device allocation");
        auto values = storage_t::copy_values(queue, scopedStorage);

        log.debug("... verify USM operation results");
        verifier.check(values.data());
      }

      bool events_result = true;
      for (size_t i = 0; i < numEvents; ++i)
        events_result = events_result && gens[i].check(init_value);

      if (!events_result)
        FAIL(log, "One or more generators completed work before the verifier."
                  "Detected mismatch in buffers.");

      log.debug("... destruct the queue");
    } catch (sycl_cts::util::fail_check &ex) {
      std::string message{verifier_t::template description<alloc>()};
      message += std::string(" failed for ") + typeid(T).name();
      message += ". ";
      message += ex.what();
      FAIL(log, message);
    } catch (...) {
      std::string message{verifier_t::template description<alloc>()};
      message += std::string(" got an exception for ") + typeid(T).name();
      log.note(message);
      throw;
    }
  }
};

/** @brief Functor used as the entry point to run every test
 *  @tparam T Underlying data type for USM or non-USM pointer
 *  @tparam checkT Defines the actual member function to check and the specific
 *          test to run, for example memcpy() with the specific source
 *          allocation
 *  @tparam caller Defines caller to use for the member function calls, so for
 *          example we can either test queue::memcpy() or handler::memcpy()
 *  @tparam numEvents Number of events to use for the test; defines an exact
 *          overload of the member function given
 */
template <typename T, template <typename, size_t> class checkT, typename caller,
          size_t numEvents>
struct run_all_tests {
  static constexpr size_t count = 8;

  template <allocation alloc>
  using test_t = test<T, count, checkT, caller, alloc, numEvents>;

  template <typename... argsT>
  void operator()(argsT &&... args) const {
    if constexpr (checkT<T, count>::has_non_usm_support()) {
      test_t<allocation::non_usm>::run(std::forward<argsT>(args)...);
    }
    test_t<allocation::host>::run(std::forward<argsT>(args)...);
    test_t<allocation::shared>::run(std::forward<argsT>(args)...);
    test_t<allocation::device>::run(std::forward<argsT>(args)...);
  }
};

}  // namespace usm_api

#endif  // __SYCL_CTS_TEST_USM_USM_API_H
