/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for sycl::use_kernel_bundle tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_USE_KERNEL_BUNDLE_H
#define __SYCLCTS_TESTS_USE_KERNEL_BUNDLE_H

#include "../../util/exceptions.h"
#include "../common/common.h"
#include "../common/type_coverage.h"
#include "kernel_bundle.h"
#include "kernels.h"

#include <cstddef>

namespace sycl_cts::tests::use_kernel_bundle {

/** @brief Kernel, that will be used to get kernel bundle, that will be used
 *         for the case when we get kernel bundle for kernel that won't be used
 *         in queue::submit
 */
class kernel_for_kernel_bundle {
 public:
  void operator()() const {}
};

/** @brief Another kernel, that will be used for the case when we get kernel
 *         bundle for kernel that won't be used in queue::submit
 */
class kernel {
 public:
  void operator()() const {}
};

static const std::string unexpected_exception_msg{
    "unexpected SYCL exception error code was caught"};

static const std::string skip_test_for_builtin_kernels_msg{
    "Test for built-in kernels will be skipped due to kernel bundle is "
    "empty."};

inline auto user_def_kernels =
    named_type_pack<kernels::kernel_cpu_descriptor,
                    kernels::kernel_gpu_descriptor,
                    kernels::kernel_accelerator_descriptor>{
        "kernel_cpu_descriptor", "kernel_gpu_descriptor",
        "kernel_accelerator_descriptor"};

template <sycl::bundle_state BundleState>
class TestCaseDescription
    : public sycl_cts::tests::kernel_bundle::TestCaseDescriptionBase<
          BundleState> {
 public:
  constexpr TestCaseDescription(std::string_view functionOverload)
      : sycl_cts::tests::kernel_bundle::TestCaseDescriptionBase<BundleState>(
            "sycl::use_kernel_bundle", functionOverload) {
    m_print_bundle_state = false;
  };
};

/** @brief Constructing empty kernel bundle
 *  @param ctx Context that will be used for kernel bundle
 *  @param num_bundles Number kernel bundles that will be constructed
 */
inline sycl::kernel_bundle<sycl::bundle_state::executable> get_empty_bundle(
    const sycl::context &ctx) {
  std::vector<sycl::kernel_id> kernel_ids{};
  auto k_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctx, ctx.get_devices(), kernel_ids);
  if (!k_bundle.empty()) {
    throw sycl_cts::util::skip_check("Test skipped due to bundle is not empty");
  }

  return k_bundle;
}

/** @brief Constructing non-empty kernel bundle with provided kernel
 *  @tparam KernelT Kernel that will be used in kernel bundle
 *  @param ctx Context that will be used for kernel bundle
 */
template <typename KernelT>
inline sycl::kernel_bundle<sycl::bundle_state::executable> get_non_empty_bundle(
    const sycl::context &ctx) {
  auto kernel_id = sycl::get_kernel_id<KernelT>();
  auto k_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctx, ctx.get_devices(), {kernel_id});
  if (k_bundle.empty()) {
    throw sycl_cts::util::skip_check("Test skipped due to bundle is empty");
  }
  return k_bundle;
}

/** @brief Constructing kernel bundle with built-in kernels
 *  @tparam KernelT Kernel that will be used in kernel bundle
 *  @param ctx Context that will be used for kernel bundle
 */
inline sycl::kernel_bundle<sycl::bundle_state::executable>
get_bundle_with_built_in_kernels(const sycl::context &ctx) {
  auto dev = ctx.get_devices()[0];
  auto kernel_ids = dev.get_info<sycl::info::device::built_in_kernel_ids>();
  auto k_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctx, {dev}, kernel_ids);
  return k_bundle;
}

}  // namespace sycl_cts::tests::use_kernel_bundle

#endif  // __SYCLCTS_TESTS_USE_KERNEL_BUNDLE_H
