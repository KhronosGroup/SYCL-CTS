/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for test case where exceptions will be thrown
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_NOTHROW_H
#define __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_NOTHROW_H

#include "../common/common.h"
#include "has_kernel_bundle.h"

namespace sycl_cts {
namespace tests {
namespace has_kernel_bundle {

namespace check {

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context
 *         and device and verify that exception was thrown
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct throws<KernelDescriptorT, BundleState, overload::id::ctx_dev> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev) {
    bool ex_was_thrown = false;
    sycl::queue queue{ctx, ctx.get_devices()[0]};

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<BundleState>(
            "<bundle_state>(context, devices, selector)"),
        [&] { sycl::has_kernel_bundle<BundleState>(ctx, {dev}); });

    kernel_bundle::define_kernel<KernelDescriptorT, BundleState>(queue);
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context,
 *         device and kernel_id and verify that exception was thrown
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct throws<KernelDescriptorT, BundleState, overload::id::ctx_dev_kid> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev) {
    using kernel_functor = typename KernelDescriptorT::type;
    auto k_id = sycl::get_kernel_id<kernel_functor>();
    bool ex_was_thrown = false;
    sycl::queue queue{ctx, ctx.get_devices()[0]};

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<BundleState>(
            "<bundle_state>(context, devices, selector)"),
        [&] { sycl::has_kernel_bundle<BundleState>(ctx, {dev}, {k_id}); });

    kernel_bundle::define_kernel<KernelDescriptorT, BundleState>(queue);
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context,
 *         device and kernel_name and verify that exception was thrown
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct throws<KernelDescriptorT, BundleState, overload::id::ctx_dev_kname> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev) {
    using kernel_functor = typename KernelDescriptorT::type;
    bool ex_was_thrown = false;
    sycl::queue queue{ctx, ctx.get_devices()[0]};

    // We expect that exception with sycl::errc::invalid will be thrown

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<BundleState>(
            "<kernel_name, bundle_state>(context, devices, selector)"),
        [&] {
          sycl::has_kernel_bundle<kernel_functor, BundleState>(ctx, {dev});
        });

    kernel_bundle::define_kernel<KernelDescriptorT, BundleState>(queue);
  }
};

}  // namespace check

}  // namespace has_kernel_bundle
}  // namespace tests
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_NOTHROW_H
