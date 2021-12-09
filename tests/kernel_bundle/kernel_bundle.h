/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for tests on kernel bundle
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_KERNEL_BUNDLE_H
#define __SYCLCTS_TESTS_KERNEL_BUNDLE_H

#include "../../util/aspect_set.h"
#include "../../util/kernel_restrictions.h"

#include "../common/assertions.h"
#include "../common/common.h"
#include "../common/get_cts_string.h"

#include <string>
#include <string_view>

namespace sycl_cts::tests::kernel_bundle {

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
    res_type result = kernel_functor::INIT_VAL;
    {
      sycl::buffer<res_type, 1> res_buffer(&result, sycl::range<1>(1));
      queue.submit([&](sycl::handler &cgh) {
        auto acc =
            res_buffer.template get_access<sycl::access_mode::read_write>(cgh);
        cgh.parallel_for(sycl::range<1>(1), kernel_functor{acc});
      });
    }
    return result == kernel_functor::EXPECTED_VAL;
  }
  return false;
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
