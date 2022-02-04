/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide common extension verification logic
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_EXTENSIONS_H
#define __SYCLCTS_UTIL_EXTENSIONS_H

#include <sycl/sycl.hpp>
#include "logger.h"

#include <string>

namespace sycl_cts {
namespace util {

/**
 * @brief Namespace that defines extension support
 *        Tag members provide common extension runtime verification support
 */
namespace extensions {
/**
 * @brief Tags can be used as a compile-time guards for different code paths
 */
namespace tag {
// Tag for default path
struct generic {};
// Tag for no extensions
struct core : generic {};
// Tags for extensions
struct atomic64 : generic {};
struct fp16 : generic {};
struct fp64 : generic {};
} // namespace tag

/**
 * @brief Retrieve extension name by tag
 */
template <typename tagT>
inline const sycl::aspect aspect();
template <>
inline const sycl::aspect aspect<tag::atomic64>() {
  return sycl::aspect::atomic64;
}
template <>
inline const sycl::aspect aspect<tag::fp16>() {
  return sycl::aspect::fp16;
}
template <>
inline const sycl::aspect aspect<tag::fp64>() {
  return sycl::aspect::fp64;
}

/**
 * @brief Retrieve description for logs by tag
 */
template <typename tagT>
inline std::string description();
template <>
inline std::string description<tag::atomic64>() {
  return "64-bit atomic operations";
}
template <>
inline std::string description<tag::fp16>() {
  return "half precision floating point operations";
}
template <>
inline std::string description<tag::fp64>() {
  return "double precision floating point operations";
}

/**
 * @brief Provide runtime extension availability check by tag
 */
template <typename tagT>
struct availability {
  /**
   *  @brief Verify extension availability without log messages
   */
  static inline bool check(const sycl::queue& queue) {
    return queue.get_device().has(aspect<tagT>());
  }
  /**
   *  @brief Verify extension availability with default log messages
   */
  static inline bool check(const sycl::queue& queue,
                           sycl_cts::util::logger& log) {
    const bool result = check(queue);
    if (!result)
      log.note("Device does not support " + description<tagT>());
    return result;
  }
};

/**
 * @brief Specialization for common code support
 */
template <>
struct availability <tag::core> {
  template <typename ... argsT>
  static inline bool check(argsT&&...) {
    return true;
  }
};
/**
 * @brief Explicitly cancel extension availability verification for generic
 *        tag to avoid possible issues with object slicing
 */
template <>
struct availability<tag::generic> {};

} //namespace extensions
} // namespace util
} // namespace sycl_cts

#endif // __SYCLCTS_UTIL_EXTENSIONS_H
