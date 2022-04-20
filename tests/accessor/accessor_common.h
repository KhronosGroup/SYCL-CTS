/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common functions for the accessor tests.
//
*******************************************************************************/

#ifndef SYCL_CTS_ACCESSOR_COMMON_H
#define SYCL_CTS_ACCESSOR_COMMON_H

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/value_helper.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers.hpp"

namespace accessor_tests_common {
using namespace sycl_cts;

/**
 * @brief Factory function for getting type_pack with all (including zero)
 *        dimensions values
 */
inline auto get_all_dimensions() {
  static const auto dimensions = integer_pack<0, 1, 2, 3>::generate_unnamed();
  return dimensions;
}
}  // namespace accessor_tests_common

#endif  // SYCL_CTS_ACCESSOR_COMMON_H
