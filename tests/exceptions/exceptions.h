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

inline const std::vector<sycl::errc> all_err_codes{
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

}  // namespace

#endif  // __SYCL_CTS_TEST_EXCEPTIONS_H
