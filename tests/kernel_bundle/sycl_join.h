/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tools for sycl::join tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SYCL_JOIN_H
#define __SYCLCTS_TESTS_SYCL_JOIN_H

#include "../common/common.h"

namespace sycl_cts::tests::sycl_join {

/** @brief Provides full comparison of device images in two kernel bundles
 *  @tparam State sycl::bundle_state
 *  @details Checks that rhs_kb contains all device images form lhs_kb.
 */
template <sycl::bundle_state State>
inline bool check_dev_images_equal(sycl_cts::util::logger &log,
                                   sycl::kernel_bundle<State> &lhs_kb,
                                   sycl::kernel_bundle<State> &rhs_kb) {
  bool ok = (std::distance(lhs_kb.begin(), lhs_kb.end()) ==
             std::distance(rhs_kb.begin(), rhs_kb.end()));
  if (!ok) return ok;
  // TODO: we can't use std::set<sycl::kernel_bundle> due to the fact that
  // sycl::kernel_bundle has no comparative operators.
  // When this issue will be resolved the following statement can be used insted
  // of cycle with std::count calling:
  //  std::sort(rhs_kb.begin(), rhs_kb.end());
  //  ok = std::adjacent_find(rhs_kb.begin(), rhs_kb.end()) != rhs_kb.end();
  for (const auto &dev_img : lhs_kb) {
    ok &= (std::find(rhs_kb.begin(), rhs_kb.end(), dev_img) != rhs_kb.end());
  }

  return ok;
}

}  // namespace sycl_cts::tests::sycl_join

#endif  // __SYCLCTS_TESTS_SYCL_JOIN_H
