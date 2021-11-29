/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for zero kernels test case
//
//  These tests all assume that there are no kernels defined anywhere in the
//  application. (Note that there are no calls to submit any kernels to a queue
//  in this file.) Therefore, this test must not be compiled together with any
//  other test that defines a kernel.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_ZERO_KERNELS_H
#define __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_ZERO_KERNELS_H

#include "../common/common.h"
#include "has_kernel_bundle.h"

namespace sycl_cts {
namespace tests {
namespace has_kernel_bundle {

namespace check {

static const std::string error_message{
    "sycl::has_kernel_bundle return true for zero kernels"};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context and
 *         verify that has_kernel_bundle return false
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct zero_kernels<KernelDescriptorT, BundleState, overload::id::ctx_only> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev) {
    if (sycl::has_kernel_bundle<BundleState>(ctx)) {
      FAIL(log, error_message);
    }
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context and
 *         kernel_id and verify that has_kernel_bundle return false
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct zero_kernels<KernelDescriptorT, BundleState, overload::id::ctx_kid> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev) {
    std::vector<sycl::kernel_id> k_ids{};

    if (sycl::has_kernel_bundle<BundleState>(ctx, {k_ids})) {
      FAIL(log, error_message);
    }
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context
 *         and device and verify that has_kernel_bundle return false
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct zero_kernels<KernelDescriptorT, BundleState, overload::id::ctx_dev> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev) {
    if (sycl::has_kernel_bundle<BundleState>(ctx, {dev})) {
      FAIL(log, error_message);
    }
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context,
 *         device and kernel_id and verify that has_kernel_bundle return false
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct zero_kernels<KernelDescriptorT, BundleState, overload::id::ctx_dev_kid> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev) {
    std::vector<sycl::kernel_id> k_ids{};

    if (sycl::has_kernel_bundle<BundleState>(ctx, {dev}, {k_ids})) {
      FAIL(log, error_message);
    }
  }
};

}  // namespace check

}  // namespace has_kernel_bundle
}  // namespace tests
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_ZERO_KERNELS_H
