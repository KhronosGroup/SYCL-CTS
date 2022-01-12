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

/** @brief Return string's description depending on the type of address space
 *  @tparam AddressSpace address space type
 *  @retval String description of address space
 */
template <sycl::access::decorated Decorated>
constexpr std::string_view for_decorated() {
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

}  // namespace get_cts_string
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_COMMON_GET_CTS_STRING_H
