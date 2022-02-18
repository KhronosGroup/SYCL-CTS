/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common code to simplify the interaction with USM
//
*******************************************************************************/

#ifndef __SYCL_CTS_UTIL_USM_HELPER_H
#define __SYCL_CTS_UTIL_USM_HELPER_H

#include <memory>
#include <string_view>
#include <sycl/sycl.hpp>

namespace usm_helper {

/** @brief Return std::unique_ptr with allocated USM object
 *  @tparam USM allocation type
 *  @param queue sycl::queue class object
 */
template <sycl::usm::alloc alloc, typename elems_typeT>
auto allocate_usm_memory(const sycl::queue &queue, size_t num_elements = 1) {
  const auto &context{queue.get_context()};
  // According to the SYCL 2020 the first device in context should be
  // used for alloc::host
  const auto &device{(alloc == sycl::usm::alloc::host)
                         ? queue.get_context().get_devices()[0]
                         : queue.get_device()};

  auto deleter = [=](elems_typeT *ptr) { sycl::free(ptr, context); };

  if constexpr (alloc == sycl::usm::alloc::shared) {
    std::unique_ptr<elems_typeT, decltype(deleter)> usm_memory(
        sycl::malloc_shared<elems_typeT>(num_elements, device, context),
        deleter);
    return usm_memory;
  } else if constexpr (alloc == sycl::usm::alloc::device) {
    std::unique_ptr<elems_typeT, decltype(deleter)> usm_memory(
        sycl::malloc_device<elems_typeT>(num_elements, queue), deleter);
    return usm_memory;
  } else if constexpr (alloc == sycl::usm::alloc::host) {
    std::unique_ptr<elems_typeT, decltype(deleter)> usm_memory(
        sycl::malloc_host<elems_typeT>(num_elements, context), deleter);
    return usm_memory;
  } else {
    static_assert(alloc != alloc, "Unknown USM allocation type");
  }
};

/** @brief Returns an aspect depending on the type of allocated memory
 *  @tparam alloc USM allocation type
 *  @retval SYCL aspect that corresponds to allocated memory
 */
template <const sycl::usm::alloc alloc>
constexpr auto get_aspect() {
  if constexpr (alloc == sycl::usm::alloc::shared) {
    return sycl::aspect::usm_shared_allocations;
  } else if constexpr (alloc == sycl::usm::alloc::device) {
    return sycl::aspect::usm_device_allocations;
  } else if constexpr (alloc == sycl::usm::alloc::host) {
    return sycl::aspect::usm_host_allocations;
  } else {
    static_assert(alloc != alloc, "Unknown USM allocation type");
  }
}

/** @brief Return string's description depending on the type of allocated memory
 *  @tparam alloc USM allocation type
 *  @retval String description of allocated memory
 */
template <sycl::usm::alloc alloc>
constexpr std::string_view get_allocation_description() {
  if constexpr (alloc == sycl::usm::alloc::shared) {
    return "shared";
  } else if constexpr (alloc == sycl::usm::alloc::device) {
    return "device";
  } else if constexpr (alloc == sycl::usm::alloc::host) {
    return "host";
  } else {
    static_assert(alloc != alloc, "Unknown USM allocation type");
  }
}

}  // namespace usm_helper

#endif  // __SYCL_CTS_UTIL_USM_HELPER_H
