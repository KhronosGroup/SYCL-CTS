/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides the common code for the tests on the `sycl::get_kernel_bundle`
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_GET_KERNEL_BUNDLE_H
#define __SYCLCTS_TESTS_GET_KERNEL_BUNDLE_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "kernel_bundle.h"
#include "kernels.h"

namespace sycl_cts {
namespace tests {
namespace get_kernel_bundle {

inline auto kernels_with_attributes = named_type_pack<
    kernels::kernel_cpu_descriptor, kernels::kernel_gpu_descriptor,
    kernels::kernel_accelerator_descriptor, kernels::simple_kernel_descriptor,
    kernels::simple_kernel_descriptor_second,
    kernels::kernel_likely_unsupported_sub_group_size_descriptor,
    kernels::kernel_likely_supported_sub_group_size_descriptor,
    kernels::kernel_likely_unsupported_work_group_size_descriptor,
    kernels::kernel_likely_supported_work_group_size_descriptor>{
    "kernel_cpu_descriptor",
    "kernel_gpu_descriptor",
    "kernel_accelerator_descriptor",
    "simple_kernel_descriptor",
    "simple_kernel_descriptor_second",
    "kernel_likely_unsupported_sub_group_size_descriptor",
    "kernel_likely_supported_sub_group_size_descriptor",
    "kernel_likely_unsupported_work_group_size_descriptor",
    "kernel_likely_supported_work_group_size_descriptor"};

inline auto kernels_without_attributes =
    named_type_pack<kernels::simple_kernel_descriptor,
                    kernels::simple_kernel_descriptor_second,
                    kernels::kernel_fp16_no_attr_descriptor,
                    kernels::kernel_fp64_no_attr_descriptor,
                    kernels::kernel_atomic64_no_attr_descriptor>{
        "simple_kernel_descriptor", "simple_kernel_descriptor_second",
        "kernel_fp16_no_attr_descriptor", "kernel_fp64_no_attr_descriptor",
        "kernel_atomic64_no_attr_descriptor"};

template <sycl::bundle_state BundleState>
class TestCaseDescription
    : public kernel_bundle::TestCaseDescriptionBase<BundleState> {
 public:
  constexpr TestCaseDescription(std::string_view functionOverload)
      : kernel_bundle::TestCaseDescriptionBase<BundleState>(
            "sycl::get_kernel_bundle", functionOverload){};
};

/** @brief Fills std::vector with kernels_ids if kernel compatible with the
 *         provided device. This let avoid situations when new kernel descriptor
 *         was added, but test developer forget to update kernel_ids for tests
 *         and this kernel will be skipped.
 *  @tparam KernelDescriptorT Kernel descriptor
 */
template <typename KernelDescriptorT>
struct fill_vector_with_user_defined_kernel_ids {
  void operator()(std::vector<sycl::kernel_id> &user_defined_kernel_ids,
                  const sycl::device &device, const std::string &kernel_name) {
    using kernel = typename KernelDescriptorT::type;
    const auto kernel_restrictions{
        kernel_bundle::get_restrictions<KernelDescriptorT,
                                        sycl::bundle_state::executable>()};

    if (kernel_restrictions.is_compatible(device)) {
      auto k_id{sycl::get_kernel_id<kernel>()};
      user_defined_kernel_ids.push_back(k_id);
    }
  }
};

/** @brief Do verifications for two sycl::get_kernel_bundle overloads that
 *         obtain KernelName as template parameter. For each provided kernel
 *         descriptor call sycl::get_kernel_bundle for each bundle state and
 *         verify obtained kernel bundle
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @param queue sycl::queue class object for obtaining context and current
 *         device
 *  @param kernel_name String with tested kernel representation
 */
template <typename KernelDescriptorT>
struct run_tests_for_overloads_that_obtain_kernel_name {
  void operator()(util::logger &log, sycl::queue &queue,
                  const std::string &kernel_name) {
    const auto context{queue.get_context()};
    const auto device{queue.get_device()};

    const auto kernel_restrictions{
        kernel_bundle::get_restrictions<KernelDescriptorT,
                                        sycl::bundle_state::executable>()};
    using kernel = typename KernelDescriptorT::type;
    auto k_id{sycl::get_kernel_id<kernel>()};

    // verifications for
    // sycl::get_kernel_bundle<KernelName,BundleState>(context) overload
    {
      auto kernel_bundle{
          sycl::get_kernel_bundle<kernel, sycl::bundle_state::executable>(
              context)};

      const bool kb_has_kernel{kernel_bundle.has_kernel(k_id)};
      const bool dev_compat_status{kernel_restrictions.is_compatible(device)};
      kernel_bundle::compare_dev_compat_and_has_kb_result(
          log, dev_compat_status, kb_has_kernel, kernel_name);
    }

    // verifications for sycl::get_kernel_bundle<KernelName,
    // BundleState>(context, devices) overload
    {
      auto kernel_bundle{
          sycl::get_kernel_bundle<kernel, sycl::bundle_state::executable>(
              context, {device})};

      const bool kb_has_kernel{kernel_bundle.has_kernel(k_id)};
      const bool dev_compat_status{kernel_restrictions.is_compatible(device)};

      kernel_bundle::compare_dev_compat_and_has_kb_result(
          log, dev_compat_status, kb_has_kernel, kernel_name);
    }
  }
};

/** @brief Observing kernel bundle with provided bundle state for each overloads
 *         that dont receive KernelName template parameter then call
 *         for "kernel_bundle::verify_that_kernel_in_bundle" with provided
 *         named_type_pack that filled with predefined kernel descriptors
 *  @tparam BundleState sycl::bundle_state enum's field
 *  @tparam KernelDescriptorsT That filled provided named_type_pack
 *  @param log sycl_cts::util::logger class object
 *  @param queue sycl::queue class object
 *  @param kernel_descriptors Named type pack with user-defined kernel
 *         descriptors
 */
template <sycl::bundle_state BundleState, typename... KernelDescriptorsT>
inline void run_test_for_all_overload_types(
    util::logger &log, sycl::queue &queue,
    const named_type_pack<KernelDescriptorsT...> &kernel_descriptors,
    const std::vector<sycl::kernel_id> &user_defined_kernel_ids) {
  const auto context{queue.get_context()};
  const auto device{queue.get_device()};
  const size_t number_devices{context.get_devices().size()};

  const auto selector = [](const sycl::device_image<BundleState> &) {
    return true;
  };

  {
    log.note(
        "Run test for sycl::get_kernel_bundle<BundleState>(context, devices, "
        "selector) overload");
    auto kernel_bundle{sycl::get_kernel_bundle<BundleState>(
        context, context.get_devices(), selector)};
    for_all_types<kernel_bundle::verify_that_kernel_in_bundle>(
        kernel_descriptors, log, kernel_bundle, number_devices);
  }
  {
    log.note(
        "Run test for sycl::get_kernel_bundle<BundleState>(context, "
        "selector) "
        "overload");
    auto kernel_bundle{sycl::get_kernel_bundle<BundleState>(context, selector)};
    for_all_types<kernel_bundle::verify_that_kernel_in_bundle>(
        kernel_descriptors, log, kernel_bundle, number_devices);
  }
  {
    log.note(
        "Run test for sycl::get_kernel_bundle<BundleState>(context, devices, "
        "kernel_ids) overload");
    auto kernel_bundle{sycl::get_kernel_bundle<BundleState>(
        context, context.get_devices(), user_defined_kernel_ids)};
    for_all_types<kernel_bundle::verify_that_kernel_in_bundle>(
        kernel_descriptors, log, kernel_bundle, number_devices);
  }
  {
    log.note(
        "Run test for sycl::get_kernel_bundle<BundleState>(context, devices) "
        "overload");
    auto kernel_bundle{
        sycl::get_kernel_bundle<BundleState>(context, context.get_devices())};
    for_all_types<kernel_bundle::verify_that_kernel_in_bundle>(
        kernel_descriptors, log, kernel_bundle, number_devices);
  }
  {
    log.note(
        "Run test for sycl::get_kernel_bundle<BundleState>(context, "
        "kernel_ids) overload");
    auto kernel_bundle{
        sycl::get_kernel_bundle<BundleState>(context, user_defined_kernel_ids)};
    for_all_types<kernel_bundle::verify_that_kernel_in_bundle>(
        kernel_descriptors, log, kernel_bundle, number_devices);
  }
  {
    log.note(
        "Run test for sycl::get_kernel_bundle<BundleState>(context) "
        "overload");
    auto kernel_bundle{sycl::get_kernel_bundle<BundleState>(context)};
    for_all_types<kernel_bundle::verify_that_kernel_in_bundle>(
        kernel_descriptors, log, kernel_bundle, number_devices);
  }
}

}  // namespace get_kernel_bundle
}  // namespace tests
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_GET_KERNEL_BUNDLE_H
