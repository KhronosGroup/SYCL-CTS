/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Factory methods for objects created once per translation unit
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_ONCE_PER_UNIT_H
#define __SYCLCTS_TESTS_COMMON_ONCE_PER_UNIT_H

#include "../common/get_cts_object.h"

namespace {
  /**
   * All symbols have internal linkage here;
   * special attention to the ODR rules should be made
   */
namespace once_per_unit {
  /**
   * @brief Factory method; provides unique queue instance per compilation unit
   */
  inline sycl::queue& get_queue() {
    static auto q = sycl_cts::util::get_cts_object::queue();
    return q;
  }
} // namespace once_per_unit
} // namespace

#endif // __SYCLCTS_TESTS_COMMON_ONCE_PER_UNIT_H
