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

namespace sycl_cts::tests::kernel_bundle::get_kernel_bundle {

inline auto kernels_with_attributes = named_type_pack<
    kernels::kernel_cpu_descriptor, kernels::kernel_gpu_descriptor,
    kernels::kernel_accelerator_descriptor, kernels::simple_kernel_descriptor,
    kernels::simple_kernel_descriptor_second,
    kernels::kernel_likely_unsupported_sub_group_size_descriptor,
    kernels::kernel_likely_supported_sub_group_size_descriptor,
    kernels::kernel_likely_unsupported_work_group_size_descriptor,
    kernels::kernel_likely_supported_work_group_size_descriptor>::generate(
    "kernel_cpu_descriptor",
    "kernel_gpu_descriptor",
    "kernel_accelerator_descriptor",
    "simple_kernel_descriptor",
    "simple_kernel_descriptor_second",
    "kernel_likely_unsupported_sub_group_size_descriptor",
    "kernel_likely_supported_sub_group_size_descriptor",
    "kernel_likely_unsupported_work_group_size_descriptor",
    "kernel_likely_supported_work_group_size_descriptor");

inline auto kernels_without_attributes =
    named_type_pack<kernels::simple_kernel_descriptor,
                    kernels::simple_kernel_descriptor_second,
                    kernels::kernel_fp16_no_attr_descriptor,
                    kernels::kernel_fp64_no_attr_descriptor,
                    kernels::kernel_atomic64_no_attr_descriptor>::generate(
        "simple_kernel_descriptor", "simple_kernel_descriptor_second",
        "kernel_fp16_no_attr_descriptor", "kernel_fp64_no_attr_descriptor",
        "kernel_atomic64_no_attr_descriptor");

template <sycl::bundle_state BundleState>
class TestCaseDescription
    : public kernel_bundle::TestCaseDescriptionBase<BundleState> {
 public:
  constexpr TestCaseDescription(std::string_view functionOverload)
      : kernel_bundle::TestCaseDescriptionBase<BundleState>(
            "sycl::get_kernel_bundle", functionOverload){};
};

/** @brief Fills an std::vector with kernels_ids for all kernels compatible with
 *         the provided device. This avoids failures when new kernel descriptors
 *         are added, but the test developer forgot to update kernel_ids for
 *         tests skipping the new kernels.
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

template <typename KernelDescriptorT, sycl::bundle_state BundleState,
          unsigned int KernelDescriptorCount>
struct helper_device_image_has_kernel {
  /**
   * Sets \p has_kernel[index] if \p device_image.has_kernel(id)
   * for the kernel \p id described by \p KernelDescriptorT. */
  void operator()(const sycl::device_image<BundleState> &device_image,
                  std::bitset<KernelDescriptorCount> &has_kernel,
                  unsigned int index) {
    using kernel = typename KernelDescriptorT::type;
    auto kernel_id{sycl::get_kernel_id<kernel>()};

    if (device_image.has_kernel(kernel_id)) {
      has_kernel[index] = true;
    }
  }
};

template <typename KernelDescriptorT, sycl::bundle_state BundleState,
          unsigned int KernelDescriptorCount>
struct helper_device_image_has_kernel_dev {
  /**
   * Sets \p has_kernel_dev[index] if \p device_image.has_kernel_dev(id, device)
   * for the kernel \p id described by \p KernelDescriptorT. */
  void operator()(const sycl::device_image<BundleState> &device_image,
                  std::bitset<KernelDescriptorCount> &has_kernel_dev,
                  unsigned int index, const sycl::device &device) {
    using kernel = typename KernelDescriptorT::type;
    auto kernel_id{sycl::get_kernel_id<kernel>()};

    if (device_image.has_kernel(kernel_id, device)) {
      has_kernel_dev[index] = true;
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
  constexpr unsigned int kernel_descriptor_count =
      sizeof...(KernelDescriptorsT);

  // represent has_kernel(kernel_id) and has_kernel(kernel_id, device)
  std::bitset<kernel_descriptor_count> has_kernel;
  std::bitset<kernel_descriptor_count> has_kernel_dev;

  // the usage of this selector in get_kernel_bundle must set all elements of
  // has_kernel and has_kernel_dev to true
  const auto selector =
      [&](const sycl::device_image<BundleState> &device_image) {
        unsigned int index;

        // for each kernel_id in KernelDescriptorsT, set its respective element
        // in has_kernel, if device_image.has_kernel(kernel_id)
        index = 0;
        ((helper_device_image_has_kernel<KernelDescriptorsT, BundleState,
                                         kernel_descriptor_count>{}(
              device_image, has_kernel, index),
          ++index),
         ...);

        // for each kernel_id in KernelDescriptorsT, set its respective element
        // in has_kernel_dev, if device_image.has_kernel(kernel_id, device)
        index = 0;
        ((helper_device_image_has_kernel_dev<KernelDescriptorsT, BundleState,
                                             kernel_descriptor_count>{}(
              device_image, has_kernel_dev, index, device),
          ++index),
         ...);

        return true;
      };

  {
    log.note(
        "Run test for sycl::get_kernel_bundle<BundleState>(context, devices, "
        "selector) overload");
    has_kernel.reset();
    has_kernel_dev.reset();
    auto kernel_bundle{sycl::get_kernel_bundle<BundleState>(
        context, context.get_devices(), selector)};
    for_all_types<kernel_bundle::verify_that_kernel_in_bundle>(
        kernel_descriptors, log, kernel_bundle, number_devices);
    CHECK((has_kernel.all() && has_kernel_dev.all()));
  }
  {
    log.note(
        "Run test for sycl::get_kernel_bundle<BundleState>(context, "
        "selector) overload");
    has_kernel.reset();
    has_kernel_dev.reset();
    auto kernel_bundle{sycl::get_kernel_bundle<BundleState>(context, selector)};
    for_all_types<kernel_bundle::verify_that_kernel_in_bundle>(
        kernel_descriptors, log, kernel_bundle, number_devices);
    CHECK((has_kernel.all() && has_kernel_dev.all()));
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

}  // namespace sycl_cts::tests::kernel_bundle::get_kernel_bundle

#endif  // __SYCLCTS_TESTS_GET_KERNEL_BUNDLE_H
