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

namespace sycl_cts::get_cts_string {

enum class verbosity { brief = 0, detail };

inline std::string for_bool(bool flag) {
  return flag ? "true" : "false";
}

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

}  // namespace sycl_cts::get_cts_string

#endif  // __SYCLCTS_TESTS_COMMON_GET_CTS_STRING_H
