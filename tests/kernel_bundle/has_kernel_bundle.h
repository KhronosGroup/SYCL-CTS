/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for tests on sycl::has_kernel_bundle
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_H
#define __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_H

#include "../../util/aspect_set.h"
#include "../../util/device_set.h"
#include "../../util/kernel_restrictions.h"
#include "../common/common.h"
#include "../common/type_coverage.h"
#include "kernel_bundle.h"
#include "kernels.h"

namespace sycl_cts::tests::has_kernel_bundle {

static const std::string unexpected_exception_msg{
    "unexpected SYCL exception error code was caught"};

static const auto simple_kernel =
    named_type_pack<kernels::simple_kernel_descriptor>::generate(
        "simple_kernel_descriptor");

template <sycl::bundle_state BundleState>
class TestCaseDescription
    : public kernel_bundle::TestCaseDescriptionBase<BundleState> {
 public:
  constexpr TestCaseDescription(std::string_view functionOverload)
      : kernel_bundle::TestCaseDescriptionBase<BundleState>(
            "sycl::has_kernel_bundle", functionOverload){};
};

namespace overload {

/** @brief Enum for sycl::has_kernel_bundle overload used
 */
enum class id {
  ctx_only,
  ctx_kid,
  ctx_dev,
  ctx_dev_kid,
  ctx_kname,
  ctx_dev_kname
};

}  // namespace overload

namespace check {

/** @brief Verification logic for zero device
 */
template <typename KernelDescriptorT, sycl::bundle_state, overload::id>
struct zero_device;

/** @brief Verification logic for zero kernels
 */
template <typename KernelDescriptorT, sycl::bundle_state, overload::id>
struct zero_kernels;

/** @brief Verification logic for cases when exception expected
 */
template <typename KernelDescriptorT, sycl::bundle_state, overload::id>
struct throws;

/** @brief Verification logic for core check on call result
 */
template <typename KernelDescriptorT, sycl::bundle_state, overload::id>
struct core;

/** @brief Verification logic for case of multiple kernel ids call
 */
template <sycl::bundle_state, overload::id>
struct multiple_kernels;

}  // namespace check

}  // namespace sycl_cts::tests::has_kernel_bundle

#endif  // __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_H
