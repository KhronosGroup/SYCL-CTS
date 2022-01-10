/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common functions for sycl::kernel_bundle tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_KERNEL_BUNDLE_H
#define __SYCLCTS_TESTS_KERNEL_BUNDLE_H

#include "../../util/exceptions.h"
#include "../common/common.h"

namespace sycl_cts {
namespace tests {
namespace kernel_bundle {

/** @brief Submit dummy kernel with specific kernel name
 */
template <typename KernelName>
void define_kernel(sycl::queue &queue) {
  queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<KernelName>([=]() {}); });
}

}  // namespace kernel_bundle
}  // namespace tests
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_KERNEL_BUNDLE_H
