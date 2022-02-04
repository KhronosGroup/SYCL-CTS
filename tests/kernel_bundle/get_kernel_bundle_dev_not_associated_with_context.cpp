/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls these sycl::get_kernel_bundle overloads:
//    1) sycl::get_kernel_bundle<BundleState>(context, devices, selector)
//    2) sycl::get_kernel_bundle<BundleState>(context, devices)
//    3) sycl::get_kernel_bundle<BundleState>(context, devices, kernelIds)
//    4) sycl::get_kernel_bundle<KernelName, BundleState>(context, devices)
//  With device that are not associated with provided context (devices from
//  different platforms).
//
//  The test verifies that the exceptions with sycl::errc::invalid are thrown.
//
*******************************************************************************/

#include "../common/assertions.h"
#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "get_kernel_bundle.h"
#include "kernel_bundle.h"
#include "kernels.h"

#define TEST_NAME get_kernel_bundle_dev_not_associated_with_context

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::kernel_bundle;
using namespace sycl_cts::tests::kernel_bundle::get_kernel_bundle;

/** @brief Calls get_kernel_bundle overloads when provided device is not
 *         associated with provided context and verify that exception was thrown
 *  @tparam BundleState Bundle state
 *  @param log sycl_cts::util::logger class object
 *  @param context Context that will be used in sycl::get_kernel_bundle
 *  @param devices std::vector sycl::device with that will be used in
 *         sycl::get_kernel_bundle
 */
template <sycl::bundle_state BundleState>
void run_test_for_all_overload_types(
    util::logger &log, const sycl::context &context,
    const std::vector<sycl::device> &devices,
    const std::vector<sycl::kernel_id> &user_defined_kernel_ids) {
  const auto selector = [](const sycl::device_image<BundleState> &) {
    return true;
  };

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("(context, devices, selector)"),
      [&] {
        sycl::get_kernel_bundle<BundleState>(context, devices, selector);
      });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("(context, devices, kernel_ids)"),
      [&] {
        sycl::get_kernel_bundle<BundleState>(context, devices,
                                             user_defined_kernel_ids);
      });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("(context, devices)"),
      [&] { sycl::get_kernel_bundle<BundleState>(context, devices); });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("<KernelName>(context, devices)"),
      [&] { sycl::get_kernel_bundle<kernel, BundleState>(context, devices); });
}

class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    const std::vector<sycl::device> devices{sycl::device::get_devices()};
    if (devices.size() <= 1) {
      log.note("Test skipped due to only zero or one device was found");
      return;
    }

    sycl::context context(devices[0]);
    sycl::queue queue{devices[0]};

    auto k_id = sycl::get_kernel_id<typename simple_kernel_descriptor::type>();
    std::vector<sycl::kernel_id> user_defined_kernel_ids{k_id};

    run_test_for_all_overload_types<sycl::bundle_state::input>(
        log, context, {devices[1]}, user_defined_kernel_ids);
    run_test_for_all_overload_types<sycl::bundle_state::object>(
        log, context, {devices[1]}, user_defined_kernel_ids);
    run_test_for_all_overload_types<sycl::bundle_state::executable>(
        log, context, {devices[1]}, user_defined_kernel_ids);

    define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(
        queue);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
