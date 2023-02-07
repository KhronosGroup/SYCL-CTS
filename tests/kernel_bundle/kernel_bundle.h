/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

//  Provides common code for tests on kernel bundle

#ifndef __SYCLCTS_TESTS_KERNEL_BUNDLE_H
#define __SYCLCTS_TESTS_KERNEL_BUNDLE_H

#include "../../util/aspect_set.h"
#include "../../util/kernel_restrictions.h"

#include "../common/assertions.h"
#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "../common/type_coverage.h"
#include "kernels.h"

#include <string>
#include <string_view>

namespace sycl_cts::tests::kernel_bundle {

inline auto kernels_for_link_and_build =
    named_type_pack<kernels::simple_kernel_descriptor,
                    kernels::simple_kernel_descriptor_second>::
        generate("simple_kernel_descriptor", "simple_kernel_descriptor_second");

using cpu_kernel = kernels::kernel_cpu_descriptor::type;
using gpu_kernel = kernels::kernel_gpu_descriptor::type;
using accelerator_kernel = kernels::kernel_accelerator_descriptor::type;
using first_simple_kernel = kernels::simple_kernel_descriptor::type;
using second_simple_kernel = kernels::simple_kernel_descriptor_second::type;
using fp16_kernel = kernels::kernel_fp16_no_attr_descriptor::type;
using fp64_kernel = kernels::kernel_fp64_no_attr_descriptor::type;
using atomic64_kernel = kernels::kernel_atomic64_no_attr_descriptor::type;

/** @brief Provides device aspects required for the bundle state given
 */
template <sycl::bundle_state BundleState>
inline auto get_bundle_state_aspects() {
  using aspect_set_t = sycl_cts::util::aspect::aspect_set;

  if constexpr (BundleState == sycl::bundle_state::object) {
    return aspect_set_t{sycl::aspect::online_linker};
  } else if constexpr (BundleState == sycl::bundle_state::input) {
    return aspect_set_t{sycl::aspect::online_compiler};
  } else {
    return aspect_set_t{};
  }
}

/** @brief Constructing restrictions that depends on sycl::bundle_state for
 *         current test case
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @retval Constructed restrictions
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
sycl_cts::util::kernel_restrictions get_restrictions() {
  auto restrictions = KernelDescriptorT::get_restrictions();
  const auto extra_aspects = get_bundle_state_aspects<BundleState>();

  restrictions.add_aspects(extra_aspects);
  return restrictions;
}

/** @brief Provide enumeration that let choosing between defining kernel and it
 *         invoking
 */
enum class submit_kernel { yes, no };

/** @brief Call sycl::queue::submit only if dev_is_compat flag is true
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @tparam BundleState sycl::bundle_state enum's enumeration's field
 *  @param queue sycl::queue class object
 *  @param dev sycl::device class object
 */
template <typename KernelDescriptorT, sycl::bundle_state BundleState>
bool define_kernel(sycl::queue &queue,
                   submit_kernel also_submit = submit_kernel::no) {
  auto restrictions{get_restrictions<KernelDescriptorT, BundleState>()};
  const bool dev_is_compat = restrictions.is_compatible(queue.get_device());
  // We use sycl::queue::submit only for compatible pair of kernel and device
  // but even for a case when device is not compatible we are sure that we can
  // call sycl::get_kernel_id
  if (also_submit == submit_kernel::yes && dev_is_compat) {
    using kernel_functor = typename KernelDescriptorT::type;
    using res_type = typename kernel_functor::element_type;
    res_type result = kernel_functor::init_val;
    {
      sycl::buffer<res_type, 1> res_buffer(&result, sycl::range<1>(1));
      queue.submit([&](sycl::handler &cgh) {
        auto acc =
            res_buffer.template get_access<sycl::access_mode::read_write>(cgh);
        cgh.parallel_for<kernel_functor>(sycl::range<1>(1),
                                         kernel_functor{acc});
      });
    }
    return result == kernel_functor::expected_val;
  }
  return false;
}

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
                  std::string(get_cts_string::for_bool(has_k_b_result)) +
                  ", but device compatible status is: " +
                  std::string(get_cts_string::for_bool(dev_compat_status)));
  }
}

/** @brief Verify that the provided sycl::kernel_bundle has the specified kernel
 *         and that the associated device satisfy all kernel requirements. Also
 *         verifies that number of devices in kernel bundle is equal to the
 *         passed devices number.
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @param log sycl_cts::util::logger class object
 *  @param kernel_bundle kernel bundle that was obtained from different
 *         overloads of sycl::get_kernel_bundle function
 *  @param dev_vector Devices that was provided to the sycl::get_kernel_bundle
 *  @param kernel_name String with tested kernel representation
 */
template <typename KernelDescriptorT>
struct verify_that_kernel_in_bundle {
  template <typename KernelBundleT>
  void operator()(util::logger &log, const KernelBundleT &kernel_bundle,
                  const std::vector<sycl::device> &dev_vector,
                  const std::string &kernel_name) {
    const auto kernel_restrictions{
        get_restrictions<KernelDescriptorT, sycl::bundle_state::executable>()};
    if (kernel_bundle.get_devices().size() != dev_vector.size()) {
      FAIL(log,
           "Test failed due to kernel bundle devices length not equal to "
           "device vector size");
    } else {
      using kernel = typename KernelDescriptorT::type;
      auto k_id{sycl::get_kernel_id<kernel>()};

      const bool kb_has_kernel{kernel_bundle.has_kernel(k_id)};
      const bool dev_compat_status{
          std::all_of(dev_vector.begin(), dev_vector.end(),
                      [&](const sycl::device &device) {
                        return kernel_restrictions.is_compatible(device);
                      })};

      compare_dev_compat_and_has_kb_result(log, dev_compat_status,
                                           kb_has_kernel, kernel_name);
    }
  }
};

/** @brief Submit dummy kernel, without any requires with specific kernel name
 *  @tparam KernelName Name of the kernel that will be defined
 *  @param queue sycl::queue class object
 */
template <typename KernelName>
void define_kernel(sycl::queue &queue) {
  queue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<KernelName>([=]() {}); });
}

template <sycl::bundle_state BundleState>
class TestCaseDescriptionBase : public ITestCaseDescription {
 protected:
  // Storing string_view is only valid with constexpr constructor
  const std::string_view m_functionName;
  const std::string_view m_functionOverload;
  bool m_print_bundle_state{true};

 public:
  constexpr TestCaseDescriptionBase(std::string_view functionName,
                                    std::string_view functionOverload)
      : m_functionName(functionName), m_functionOverload(functionOverload) {}

  std::string to_string() const override {
    std::string result(m_functionName);

    if (!m_functionOverload.empty()) {
      result += m_functionOverload;
    }
    if (m_print_bundle_state) {
      result += " with bundle state ";
      result += sycl_cts::get_cts_string::for_bundle_state<BundleState>();
    }

    return result;
  }
};

}  // namespace sycl_cts::tests::kernel_bundle

#endif  // __SYCLCTS_TESTS_KERNEL_BUNDLE_H
