/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tools for string representation for some types
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_GET_CTS_STRING_H
#define __SYCLCTS_TESTS_COMMON_GET_CTS_STRING_H

#include "common.h"

namespace sycl_cts {
namespace get_cts_string {

enum class verbosity { brief = 0, detail };

/** @brief Stringify bool value
 *  @tparam bool flag Value to convert
 *  @retval String with interpretation of bool
 */
inline std::string for_bool(bool flag) { return flag ? "true" : "false"; }

/** @brief Return string's description depending on the type of bundle_state and
 * verbosity level
 *  @tparam State bundle state
 *  @tparam Level verbosity level
 *  @retval string's description of bundle state
 */
template <sycl::bundle_state State, verbosity Level = verbosity::brief>
inline std::string for_bundle_state() {
  std::string result;
  if constexpr (State == sycl::bundle_state::input) {
    result += "input";
  } else if constexpr (State == sycl::bundle_state::object) {
    result += "object";
  } else if constexpr (State == sycl::bundle_state::executable) {
    result += "executable";
  } else {
    static_assert(State != State, "incorrect kernel bundle state");
  }

  if constexpr (Level > verbosity::brief) {
    result += " kernel bundle state";
  }

  return result;
}

/** @brief Return string's description depending on the type of address space
 *  @tparam AddressSpace type of address space that needs to be converted to
 * string
 *  @retval String description of given address space
 */
template <sycl::access::address_space AddressSpace>
inline std::string_view for_address_space() {
  if constexpr (AddressSpace == sycl::access::address_space::global_space) {
    return "global_space";
  } else if constexpr (AddressSpace ==
                       sycl::access::address_space::local_space) {
    return "local_space";
  } else if constexpr (AddressSpace ==
                       sycl::access::address_space::private_space) {
    return "private_space";
  } else if constexpr (AddressSpace ==
                       sycl::access::address_space::generic_space) {
    return "generic_space";
  } else if constexpr (AddressSpace ==
                       sycl::access::address_space::constant_space) {
    return "constant_space(deprecated)";
  } else {
    static_assert(AddressSpace != AddressSpace, "Unknown address space type");
  }
}

/** @brief Return string's description depending on the type of decorated
 *  @tparam Decorated value that needs to be converted to string
 *  @retval String description of address space
 */
template <sycl::access::decorated Decorated>
inline std::string_view for_decorated() {
  if constexpr (Decorated == sycl::access::decorated::yes) {
    return "yes";
  } else if constexpr (Decorated == sycl::access::decorated::no) {
    return "no";
  } else if constexpr (Decorated == sycl::access::decorated::legacy) {
    return "legacy";
  } else {
    static_assert(Decorated != Decorated, "Unknown decorated type");
  }
}

/** @brief Return string's description depending on the type of Mode
 *  @tparam Mode value that needs to be converted to string
 *  @retval String description of mode
 */
template <sycl::access::mode Mode>
inline std::string_view for_mode() {
  if constexpr (Mode == sycl::access::mode::read) {
    return "read";
  } else if constexpr (Mode == sycl::access::mode::read_write) {
    return "read_write";
  } else if constexpr (Mode == sycl::access::mode::write) {
    return "write";
  } else if constexpr (Mode == sycl::access::mode::atomic) {
    return "atomic(deprecated)";
  } else if constexpr (Mode == sycl::access::mode::discard_read_write) {
    return "discard_read_write(deprecated)";
  } else if constexpr (Mode == sycl::access::mode::discard_write) {
    return "discard_write(deprecated)";
  } else {
    static_assert(Mode != Mode, "Unknown mode type");
  }
}

/** @brief Return string's description depending on the type of Target
 *  @tparam Target value that needs to be converted to string
 *  @retval String description of target
 */
template <sycl::access::target Target>
inline std::string_view for_target() {
  if constexpr (Target == sycl::access::target::constant_buffer) {
    return "constant_buffer";
  } else if constexpr (Target == sycl::access::target::device) {
    return "device";
  } else if constexpr (Target == sycl::access::target::host_buffer) {
    return "host_buffer";
  } else if constexpr (Target == sycl::access::target::host_image) {
    return "host_image";
  } else if constexpr (Target == sycl::access::target::image) {
    return "image";
  } else if constexpr (Target == sycl::access::target::image_array) {
    return "image_array";
  } else if constexpr (Target == sycl::access::target::local) {
    return "local";
  } else if (Target == sycl::access::target::global_buffer) {
    return "global_buffer(deprecated)";
  } else {
    static_assert(Target != Target, "Unknown target type");
  }
}

/** @brief Return string's description depending on the type of isPlaceHolder
 *  @tparam isPlaceHolder value that needs to be converted to string
 *  @retval String description of placeholder
 */
template <sycl::access::placeholder placeholder>
inline std::string_view for_placeholder() {
  if constexpr (placeholder == sycl::access::placeholder::false_t) {
    return "false_t";
  } else if constexpr (placeholder == sycl::access::placeholder::true_t) {
    return "true_t";
  } else {
    static_assert(placeholder != placeholder, "Unknown placeholder type");
  }
}

}  // namespace get_cts_string
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_COMMON_GET_CTS_STRING_H
