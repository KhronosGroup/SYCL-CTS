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

inline auto user_def_kernels = named_type_pack<
    kernels::kernel_fp16_descriptor, kernels::kernel_fp64_descriptor,
    kernels::kernel_atomic64_descriptor, kernels::kernel_image_descriptor,
    kernels::kernel_online_compiler_descriptor,
    kernels::kernel_online_linker_descriptor,
    kernels::kernel_queue_profiling_descriptor,
    kernels::kernel_usm_device_allocations_descriptor,
    kernels::kernel_usm_host_allocations_descriptor,
    kernels::kernel_usm_atomic_host_allocations_descriptor,
    kernels::kernel_usm_shared_allocations_descriptor,
    kernels::kernel_usm_atomic_shared_allocations_descriptor,
    kernels::kernel_usm_system_allocations_descriptor>::
    generate("kernel_fp16_descriptor", "kernel_fp64_descriptor",
             "kernel_atomic64_descriptor", "kernel_image_descriptor",
             "kernel_online_compiler_descriptor",
             "kernel_online_linker_descriptor",
             "kernel_queue_profiling_descriptor",
             "kernel_usm_device_allocations_descriptor",
             "kernel_usm_host_allocations_descriptor",
             "kernel_usm_atomic_host_allocations_descriptor",
             "kernel_usm_shared_allocations_descriptor",
             "kernel_usm_atomic_shared_allocations_descriptor",
             "kernel_usm_system_allocations_descriptor");

template <sycl::bundle_state BundleState>
class TestCaseDescription
    : public sycl_cts::tests::kernel_bundle::TestCaseDescriptionBase<
          BundleState> {
 public:
  constexpr TestCaseDescription(std::string_view functionOverload)
      : sycl_cts::tests::kernel_bundle::TestCaseDescriptionBase<BundleState>(
            "sycl::use_kernel_bundle", functionOverload) {
    this->m_print_bundle_state = false;
  };
};

/** @brief Constructing empty kernel bundle
 *  @param ctx Context that will be used for kernel bundle
 *  @param num_bundles Number kernel bundles that will be constructed
 */
inline sycl::kernel_bundle<sycl::bundle_state::executable> get_empty_bundle(
    const sycl::context &ctx) {
  std::vector<sycl::kernel_id> kernel_ids;
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
  try {
    auto k_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        ctx, ctx.get_devices(), {kernel_id});
    return k_bundle;
  } catch (const sycl::exception &e) {
    if (sycl::errc::invalid == e.code())
      SKIP(
          "Test skipped because no device is compatible with requested kernel");
  }
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
