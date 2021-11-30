/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for tests on kernel bundle
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_KERNEL_BUNDLE_H
#define __SYCLCTS_TESTS_KERNEL_BUNDLE_H

namespace sycl_cts {
namespace tests {
namespace kernel_bundle {

/** @brief Call queue::submit for each user-defined kernel from provided named
 *         type pack and then verify that this kernel was invoked (if device
 *         compatible with current kernel requires)
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @param log sycl_cts::util::logger class object
 *  @param queue sycl::queue class object
 *  @param kernel_name String with tested kernel representation
 */
template <typename KernelDescriptorT>
struct execute_kernel_and_verify_executions {
  void operator()(util::logger &log, sycl::queue &queue,
                  const std::string &kernel_name) {
    const auto kernel_restrictions{
        get_restrictions<KernelDescriptorT, sycl::bundle_state::executable>()};
    const bool dev_compat_status{
        kernel_restrictions.is_compatible(queue.get_device())};
    const bool kernel_was_invoked{
        define_kernel<KernelDescriptorT, sycl::bundle_state::executable>(
            queue, submit_kernel::yes)};

    if (dev_compat_status && !kernel_was_invoked) {
      FAIL(log, kernel_name + "kernel was not invoked");
    }
  }
};

/** @brief Compare device compatible status and containing kernel in kernel
 *         bundle, if they are different then fail test and print failure
 *         message
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @param log sycl_cts::util::logger class object
 *  @param queue sycl::queue class object
 *  @param kernel_name String with tested kernel representation
 */
inline void compare_dev_compat_and_has_kb_result(
    util::logger &log, bool dev_compat_status, bool has_k_b_result,
    const std::string &kernel_name) {
  if (has_k_b_result != dev_compat_status) {
    FAIL(log, "For kernel " + kernel_name +
                  " containing in bundle status is: " +
                  get_cts_string::for_bool(has_k_b_result) +
                  ", but device compatible status is: " +
                  get_cts_string::for_bool(dev_compat_status));
  }
}

/** @brief Verify that provided sycl::kernel_bundle presented kernel if it
 *         device compatible with kernel requirements. Also verifies that number
 *         of devicec in kernel bundle is equal to the passed devices number.
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @param log sycl_cts::util::logger class object
 *  @param kernel_bundle kernel bundle that was obtained from different
 *         overloads of sycl::get_kernel_bundle function
 *  @param number_devices Number of devices that was provided to the
 *         sycl::get_kernel_bundle
 *  @param kernel_name String with tested kernel representation
 */
template <typename KernelDescriptorT>
struct verify_that_kernel_in_bundle {
  template <typename KernelBundleT>
  void operator()(util::logger &log, const KernelBundleT &kernel_bundle,
                  size_t number_devices, const std::string &kernel_name) {
    const auto kernel_restrictions{
        get_restrictions<KernelDescriptorT, sycl::bundle_state::executable>()};
    if (kernel_bundle.get_devices().size() != number_devices) {
      FAIL(log,
           "Test failed due to kernel bundle devices length not equal to " +
               std::to_string(number_devices));
    } else {
      using kernel = typename KernelDescriptorT::type;
      auto k_id{sycl::get_kernel_id<kernel>()};

      const bool kb_has_kernel{kernel_bundle.has_kernel(k_id)};
      const bool dev_compat_status{
          kernel_restrictions.is_compatible(kernel_bundle.get_devices()[0])};
      compare_dev_compat_and_has_kb_result(log, dev_compat_status,
                                           kb_has_kernel, kernel_name);
    }
  }
};

}  // namespace kernel_bundle
}  // namespace tests
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_KERNEL_BUNDLE_H
