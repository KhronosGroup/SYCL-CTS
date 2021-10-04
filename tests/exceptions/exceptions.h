/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common code for exceptions tests
//
*******************************************************************************/

#ifndef __SYCL_CTS_TEST_EXCEPTIONS_H
#define __SYCL_CTS_TEST_EXCEPTIONS_H

#include "../common/common.h"
#include <vector>

/** @brief Provide common code for all tests for exceptions
 */
namespace {

const std::array<sycl::errc, 15>& get_err_codes() {
  static const std::array all_err_codes{
      sycl::errc::success,
      sycl::errc::runtime,
      sycl::errc::kernel,
      sycl::errc::accessor,
      sycl::errc::nd_range,
      sycl::errc::event,
      sycl::errc::kernel_argument,
      sycl::errc::build,
      sycl::errc::invalid,
      sycl::errc::memory_allocation,
      sycl::errc::platform,
      sycl::errc::profiling,
      sycl::errc::feature_not_supported,
      sycl::errc::kernel_not_supported,
      sycl::errc::backend_mismatch};
  return all_err_codes;
}

std::string errc_to_string(const sycl::errc& errc) {
  switch (errc) {
    case sycl::errc::success:
      return "success";
    case sycl::errc::runtime:
          return "runtime";
    case sycl::errc::kernel:
      return "kernel";
    case sycl::errc::accessor:
      return "accessor";
    case sycl::errc::nd_range:
      return "nd_range";
    case sycl::errc::event:
      return "event";
    case sycl::errc::kernel_argument:
      return "kernel_argument";
    case sycl::errc::build:
      return "build";
    case sycl::errc::invalid:
      return "invalid";
    case sycl::errc::memory_allocation:
      return "memory_allocation";
    case sycl::errc::platform:
      return "platform";
    case sycl::errc::profiling:
      return "profiling";
    case sycl::errc::feature_not_supported:
      return "feature_not_supported";
    case sycl::errc::kernel_not_supported:
      return "kernel_not_supported";
    case sycl::errc::backend_mismatch:
      return "backend_mismatch";
  }
}

}  // namespace

#endif  // __SYCL_CTS_TEST_EXCEPTIONS_H
