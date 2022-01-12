/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for test case where verifies sycl::has_kernel_bundle
//  result.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_CORE_H
#define __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_CORE_H

#include "../common/common.h"
#include "has_kernel_bundle.h"

namespace sycl_cts::tests::has_kernel_bundle {

static const auto kernels_types_sub_group = named_type_pack<
    kernels::kernel_likely_unsupported_sub_group_size_descriptor,
    kernels::kernel_likely_supported_sub_group_size_descriptor>{
    "kernel_likely_unsupported_sub_group_size_descriptor",
    "kernel_likely_supported_sub_group_size_descriptor"};

static const auto kernels_types_work_group = named_type_pack<
    kernels::kernel_likely_supported_work_group_size_descriptor,
    kernels::kernel_likely_unsupported_work_group_size_descriptor>{
    "kernel_large_work_group_size_descriptor",
    "kernel_small_work_group_size_descriptor"};

static const auto kernels_types_fp16_fp64_at64 =
    named_type_pack<kernels::kernel_fp16_descriptor,
                    kernels::kernel_fp64_descriptor,
                    kernels::kernel_atomic64_descriptor>{
        "kernel_fp16", "kernel_fp64", "kernel_atomic64"};

static const auto kernels_types_for_aspect_required = named_type_pack<
    kernels::kernel_cpu_descriptor, kernels::kernel_gpu_descriptor,
    kernels::kernel_accelerator_descriptor, kernels::kernel_custom_descriptor,
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
    kernels::kernel_usm_system_allocations_descriptor>{
    "kernel_cpu_descriptor",
    "kernel_gpu_descriptor",
    "kernel_accelerator_descriptor",
    "kernel_custom_descriptor",
    "kernel_fp16_descriptor",
    "kernel_fp64_descriptor",
    "kernel_atomic64_descriptor",
    "kernel_image_descriptor",
    "kernel_online_compiler_descriptor",
    "kernel_online_linker_descriptor",
    "kernel_queue_profiling_descriptor",
    "kernel_usm_device_allocations_descriptor",
    "kernel_usm_host_allocations_descriptor",
    "kernel_usm_atomic_host_allocations_descriptor",
    "kernel_usm_shared_allocations_descriptor",
    "kernel_usm_atomic_shared_allocations_descriptor",
    "kernel_usm_system_allocations_descriptor"};

namespace check {

/** @brief Print message to the console and fail test
 */
inline void log_err_message(sycl_cts::util::logger &log, bool has_kb_result,
                            bool dev_is_compat,
                            const std::string &string_with_restrictions,
                            const std::string &compatible_info,
                            const std::string &kernel_name) {
  FAIL(log, "sycl::has_kernel_bundle() result: " +
                sycl_cts::get_cts_string::for_bool(has_kb_result) +
                ", but device compatible status: " +
                sycl_cts::get_cts_string::for_bool(dev_is_compat) + ". for " +
                kernel_name + ", with restrictions:" +
                string_with_restrictions + " and device " + compatible_info);
}

/** @brief Verify has_kernel_bundle result by rules that depend on
 *         sycl::bundle_state
 *  @details If we got unexpected has_kernel_bundle result then this function
 *           returns TRUE
 *  @tparam BundleStateT sycl::bundle_state enum's enumeration's field
 *  @param has_kb_result Result of sycl::has_kernel_bundle calling
 *  @param dev_is_compat Current device compatible status
 *  @retval Bool value that corresponds to has_kernel_bundle result that does
 *          NOT equal to device compatible status according to rules that depend
 *          on sycl::bundle_state
 */
template <sycl::bundle_state BundleStateT>
bool result_has_kb_is_wrong(bool has_kb_result, bool dev_is_compat) {
  // The implementation must expose all kernels in "executable" state, so the
  // only reason "has_kernel_bundle()" could return false is when the kernel is
  // incompatible with the device.
  if (BundleStateT == sycl::bundle_state::executable) {
    return dev_is_compat != has_kb_result;
  } else {
    // There is no guarantee that the application represents the kernel in
    // bundle state "input" or "object", so "has_kernel_bundle()" might return
    // false simply because the kernel is not representable in this state. If it
    // returns "true", though, the kernel must be compatible with the device.
    return has_kb_result && !dev_is_compat;
  }
}

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context and
 *         verify that sycl::has_kernel_bundle result is equal to current
 *         pair of kernel and device compatible status
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 *  @param kernel_name String representation of current tested kernel
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct core<KernelDescriptorT, BundleState, overload::id::ctx_only> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev, const std::string &kernel_name) {
    using kernel_functor = typename KernelDescriptorT::type;
    auto k_id{sycl::get_kernel_id<kernel_functor>()};
    auto restrictions{
        kernel_bundle::get_restrictions<KernelDescriptorT, BundleState>()};
    std::string compat_info;
    sycl::queue queue(ctx, dev);

    bool dev_is_compat{restrictions.is_compatible(dev, compat_info)};
    bool has_kb_result{sycl::has_kernel_bundle<BundleState>(ctx)};

    if (result_has_kb_is_wrong<BundleState>(has_kb_result, dev_is_compat)) {
      log_err_message(log, has_kb_result, dev_is_compat,
                      restrictions.to_string(), compat_info, kernel_name);
    }
    kernel_bundle::define_kernel<KernelDescriptorT, BundleState>(queue);
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context and
 *         kernel_id and verify that sycl::has_kernel_bundle result is equal to
 *         current pair of kernel and device compatible status
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 *  @param kernel_name String representation of current tested kernel
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct core<KernelDescriptorT, BundleState, overload::id::ctx_kid> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev, const std::string &kernel_name) {
    using kernel_functor = typename KernelDescriptorT::type;
    auto k_id{sycl::get_kernel_id<kernel_functor>()};
    auto restrictions{
        kernel_bundle::get_restrictions<KernelDescriptorT, BundleState>()};
    std::string compat_info;
    sycl::queue queue(ctx, dev);

    bool dev_is_compat{restrictions.is_compatible(dev, compat_info)};
    bool has_kb_result{sycl::has_kernel_bundle<BundleState>(ctx, {k_id})};

    if (result_has_kb_is_wrong<BundleState>(has_kb_result, dev_is_compat)) {
      log_err_message(log, has_kb_result, dev_is_compat,
                      restrictions.to_string(), compat_info, kernel_name);
    }
    kernel_bundle::define_kernel<KernelDescriptorT, BundleState>(queue);
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context
 *         and device and verify that sycl::has_kernel_bundle result is equal to
 *         current pair of kernel and device compatible status
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 *  @param kernel_name String representation of current tested kernel
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct core<KernelDescriptorT, BundleState, overload::id::ctx_dev> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev, const std::string &kernel_name) {
    using kernel_functor = typename KernelDescriptorT::type;
    auto k_id{sycl::get_kernel_id<kernel_functor>()};
    auto restrictions{
        kernel_bundle::get_restrictions<KernelDescriptorT, BundleState>()};
    std::string compat_info;
    sycl::queue queue(ctx, dev);

    bool dev_is_compat{restrictions.is_compatible(dev, compat_info)};
    bool has_kb_result{sycl::has_kernel_bundle<BundleState>(ctx, {dev})};

    if (result_has_kb_is_wrong<BundleState>(has_kb_result, dev_is_compat)) {
      log_err_message(log, has_kb_result, dev_is_compat,
                      restrictions.to_string(), compat_info, kernel_name);
    }
    kernel_bundle::define_kernel<KernelDescriptorT, BundleState>(queue);
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context,
 *         device and kernel_id and verify that sycl::has_kernel_bundle result
 *         is equal to current pair of kernel and device compatible status
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 *  @param kernel_name String representation of current tested kernel
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct core<KernelDescriptorT, BundleState, overload::id::ctx_dev_kid> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev, const std::string &kernel_name) {
    using kernel_functor = typename KernelDescriptorT::type;
    auto k_id{sycl::get_kernel_id<kernel_functor>()};
    auto restrictions{
        kernel_bundle::get_restrictions<KernelDescriptorT, BundleState>()};
    std::string compat_info;
    sycl::queue queue(ctx, dev);

    bool dev_is_compat{restrictions.is_compatible(dev, compat_info)};
    bool has_kb_result{
        sycl::has_kernel_bundle<BundleState>(ctx, {dev}, {k_id})};

    if (result_has_kb_is_wrong<BundleState>(has_kb_result, dev_is_compat)) {
      log_err_message(log, has_kb_result, dev_is_compat,
                      restrictions.to_string(), compat_info, kernel_name);
    }
    kernel_bundle::define_kernel<KernelDescriptorT, BundleState>(queue);
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context and
 *         kernel_name and verify that sycl::has_kernel_bundle result
 *         is equal to current pair of kernel and device compatible status
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 *  @param kernel_name String representation of current tested kernel
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct core<KernelDescriptorT, BundleState, overload::id::ctx_kname> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev, const std::string &kernel_name) {
    using kernel_functor = typename KernelDescriptorT::type;
    auto k_id{sycl::get_kernel_id<kernel_functor>()};
    auto restrictions{
        kernel_bundle::get_restrictions<KernelDescriptorT, BundleState>()};
    std::string compat_info;
    sycl::queue queue(ctx, dev);

    bool dev_is_compat{restrictions.is_compatible(dev, compat_info)};
    bool has_kb_result{
        sycl::has_kernel_bundle<kernel_functor, BundleState>(ctx)};

    if (result_has_kb_is_wrong<BundleState>(has_kb_result, dev_is_compat)) {
      log_err_message(log, has_kb_result, dev_is_compat,
                      restrictions.to_string(), compat_info, kernel_name);
    }
    kernel_bundle::define_kernel<KernelDescriptorT, BundleState>(queue);
  }
};

/** @brief Call sycl::has_kernel_bundle with sycl::bundle_state, context,
 *         device and kernel_name and verify that sycl::has_kernel_bundle result
 *         is equal to current pair of kernel and device compatible status
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param ctx sycl::context class object
 *  @param dev sycl::device class object
 *  @param kernel_name String representation of current tested kernel
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
struct core<KernelDescriptorT, BundleState, overload::id::ctx_dev_kname> {
  void operator()(sycl_cts::util::logger &log, const sycl::context &ctx,
                  const sycl::device &dev, const std::string &kernel_name) {
    using kernel_functor = typename KernelDescriptorT::type;
    auto k_id = sycl::get_kernel_id<kernel_functor>();
    auto restrictions{
        kernel_bundle::get_restrictions<KernelDescriptorT, BundleState>()};
    std::string compat_info;
    sycl::queue queue(ctx, dev);

    bool dev_is_compat{restrictions.is_compatible(dev, compat_info)};
    bool has_kb_result{
        sycl::has_kernel_bundle<kernel_functor, BundleState>(ctx, {dev})};

    if (result_has_kb_is_wrong<BundleState>(has_kb_result, dev_is_compat)) {
      log_err_message(log, has_kb_result, dev_is_compat,
                      restrictions.to_string(), compat_info, kernel_name);
    }
    kernel_bundle::define_kernel<KernelDescriptorT, BundleState>(queue);
  }
};

}  // namespace check

}  // namespace sycl_cts::tests::has_kernel_bundle

#endif  // __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_CORE_H
