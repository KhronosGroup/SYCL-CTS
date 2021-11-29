/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide generic compile-time aspect sets support
//
*******************************************************************************/

#include "aspect_set.h"

#include <stdexcept>

namespace sycl_cts {
namespace util {
namespace aspect {

namespace detail {

#define ASPECT_SET_IMPL_MAP_NAME(aspectName) \
  case sycl::aspect::aspectName:             \
    result = TOSTRING(aspectName);           \
    break

inline std::string map_name(sycl::aspect value) {
  std::string result{"n/a"};

  switch (value) {
    ASPECT_SET_IMPL_MAP_NAME(cpu);
    ASPECT_SET_IMPL_MAP_NAME(gpu);
    ASPECT_SET_IMPL_MAP_NAME(accelerator);
    ASPECT_SET_IMPL_MAP_NAME(custom);
    // ASPECT_SET_IMPL_MAP_NAME(emulated);
    // ASPECT_SET_IMPL_MAP_NAME(host_debuggable);
    ASPECT_SET_IMPL_MAP_NAME(fp16);
    ASPECT_SET_IMPL_MAP_NAME(fp64);
    ASPECT_SET_IMPL_MAP_NAME(atomic64);
    ASPECT_SET_IMPL_MAP_NAME(image);
    ASPECT_SET_IMPL_MAP_NAME(online_compiler);
    ASPECT_SET_IMPL_MAP_NAME(online_linker);
    ASPECT_SET_IMPL_MAP_NAME(queue_profiling);
    ASPECT_SET_IMPL_MAP_NAME(usm_device_allocations);
    ASPECT_SET_IMPL_MAP_NAME(usm_host_allocations);
    ASPECT_SET_IMPL_MAP_NAME(usm_atomic_host_allocations);
    ASPECT_SET_IMPL_MAP_NAME(usm_shared_allocations);
    ASPECT_SET_IMPL_MAP_NAME(usm_atomic_shared_allocations);
    ASPECT_SET_IMPL_MAP_NAME(usm_system_allocations);
    default:
      throw std::logic_error("Failed to map_name");
  }
  return result;
}

#undef ASPECT_SET_IMPL_MAP_NAME

}  // namespace detail

std::string to_string(sycl::aspect asp) { return detail::map_name(asp); }

std::string to_string(const aspect_set &asp_set) {
  static const std::string delimiter = ", ";
  // Check we have any aspects to safely proceed with delimiters
  if (asp_set.empty()) return "none";

  std::string result;
  for (const auto &asp : asp_set) {
    result += detail::map_name(asp);
    result += delimiter;
  }

  // Remove the latest delimiter
  result.resize(result.size() - delimiter.size());
  return result;
}

}  // namespace aspect
}  // namespace util
}  // namespace sycl_cts
