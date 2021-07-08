/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides different tools for usm allocation/free/usm_allocator tests
//
*******************************************************************************/
#include "../common/common.h"
#include "usm.h"
#include <cstddef>

namespace usm_alloc_help {
using namespace sycl_cts;

/** @brief Custom type to create usm_allocator with other T type
 */
struct custom_type {
  int value;
};

/** @brief Kernel index to differentiate kernel names in member functions
 */
enum class memb_func_index : int { fill = 0, simple, device, host };

/** @brief Get description of index given
 */
template <memb_func_index index>
static std::string get_index_name() {
  if constexpr (index == memb_func_index::fill) {
    return "fill";
  } else if constexpr (index == memb_func_index::simple) {
    return "simple";
  } else if constexpr (index == memb_func_index::device) {
    return "device";
  } else if constexpr (index == memb_func_index::host) {
    return "host";
  } else {
    static_assert(index != index, "Wrong memb_func_index value!");
  }
}

/** @brief Get usm::alloc kind which differs from current (only used by tests
 *         for usm_allocator)
 */
template <sycl::usm::alloc kind>
inline constexpr sycl::usm::alloc other_kind() {
  if constexpr (kind == sycl::usm::alloc::host) {
    return sycl::usm::alloc::shared;
  } else if constexpr (kind == sycl::usm::alloc::shared) {
    return sycl::usm::alloc::host;
  } else {
    static_assert(kind != kind, "'kind' is supposed to be 'host' or 'shared'");
  }
}

/** @brief Check usm::alloc kind of allocated memory
 *  @tparam UPtr Expecting std::unique_ptr type
 */
template <typename UPtr>
static bool check_ptr_kind(const UPtr &ptr, sycl::context &ctx,
                           sycl::usm::alloc ref_kind) {
  return sycl::get_pointer_type(ptr.get(), ctx) == ref_kind;
}

/** @brief Check that two memory allocations have the same usm::alloc kind
 *  @tparam UPtrT Expecting std::unique_ptr type
 *  @tparam UPtrU Expecting other std::unique_ptr type
 */
template <typename UPtrT, typename UPtrU>
static bool compare_ptrs_kind(const UPtrT &lhs, const UPtrU &rhs,
                              sycl::context ctx) {
  return sycl::get_pointer_type(lhs.get(), ctx) ==
         sycl::get_pointer_type(rhs.get(), ctx);
}

/** @brief Checks if allocation of this usm::alloc kind is supported
 */
template <sycl::usm::alloc kind>
static bool allocation_supported(util::logger &log, sycl::queue &q) {
  if (!q.get_device().has(usm::get_aspect<kind>())) {
    log.note("Device does not support " +
             std::string(usm::get_allocation_description<kind>()) +
             " allocations");
    return false;
  }
  return true;
}

/** @brief Fills allocated memory
 *  @tparam T Testing type
 *  @tparam count Count of allocated elements
 *  @tparam kernel_name Unique kernel name
 */
template <typename T, std::size_t count, typename kernel_name>
void fill(T *ptr, sycl::queue &q) {
  sycl::range<1> ndRng{count};
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_name>(ndRng, [=](sycl::id<1> idx) {
      ptr[idx] = idx[0];
    });
  });
  q.wait_and_throw();
}

/** @brief Checks that allocated memory is accessible
 *  @tparam T Testing type
 *  @tparam count Count of allocated elements
 *  @tparam kernel_name Unique kernel name
 */
template <typename T, std::size_t count, typename kernel_name>
bool check_simple(util::logger &log, T *ptr, sycl::queue &q) {
  sycl::range<1> ndRng{count};
  T result_arr[count]{};
  {
    sycl::buffer<T, 1> buf(result_arr, ndRng);
    q.submit([&](sycl::handler &cgh) {
      auto acc = buf.template get_access<sycl::access_mode::write>(cgh);
      cgh.parallel_for<kernel_name>(ndRng, [=](sycl::id<1> idx) {
        acc[idx] = ptr[idx];
      });
    });
  }
  bool result = true;
  for (int i = 0; i < count; ++i) {
    result = result && (result_arr[i] == i);
  }
  return result;
}

/** @brief Allocate memory of requred usm::alloc kind and store it in
 *         std::unique_ptr
 *  @brief T Testing type
 *  @brief index Defines the allocation type
 */
template <typename T, memb_func_index index>
auto allocate_by_index(sycl::queue &q, std::size_t count) {
  if constexpr (index == memb_func_index::device) {
    return usm::allocate_usm_memory<sycl::usm::alloc::device, T>(q, count);
  } else if constexpr (index == memb_func_index::host) {
    return usm::allocate_usm_memory<sycl::usm::alloc::host, T>(q, count);
  } else {
    static_assert(index != index,
                  "'index' is supposed to be 'host' or 'device'");
  }
}

/** @brief Checks that memory is accessible for memory allocated with another
 *         usm::alloc kind
 *  @tparam T Testing type
 *  @tparam index Defines the allocation type
 *  @tparam count Count of elements to allocate
 *  @tparam kernel_name Unique kernel name
 */
template <typename T, memb_func_index index, std::size_t count,
          typename kernel_name>
bool check_by_index(util::logger &log, T *ptr, sycl::queue &q) {
  sycl::range<1> ndRng{count};
  // Save input values
  T ref_arr[count]{};
  for (int i = 0; i < count; ++i) ref_arr[i] = ptr[i];

  auto check_ptr = allocate_by_index<T, index>(q, count);
  if (!check_ptr) {
    log.note("allocation returned nullptr (check skipped)");
    return true;
  }
  T *check_raw_ptr = check_ptr.get();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel_name>(ndRng, [=](sycl::id<1> idx) {
      const int i = idx[0];
      check_raw_ptr[i] = ptr[i] * 2;
      ptr[i] = check_raw_ptr[i];
    });
  });
  q.wait_and_throw();
  bool result = true;
  for (int i = 0; i < count; ++i) {
    result = result && (ptr[i] == ref_arr[i] * 2);
  }
  return result;
}

/** @brief Check that pointer address aligned with given value
 */
template <typename T>
bool check_alignment(T *ptr, std::size_t align) {
  return (reinterpret_cast<std::size_t>(ptr) % align) == 0;
}

/** @brief Custom deleter for automatical deallocation of memory allocated by
 *         usm_allocator
 */
template <typename T, typename Allocator, std::size_t count>
class usm_custom_deleter {
  Allocator& allocator;
 public:
  usm_custom_deleter(Allocator& allctr) : allocator(allctr) {}

  void operator()(T* ptr) {
    allocator.deallocate(ptr, count);
  }
};

}  // namespace usm_alloc_help
