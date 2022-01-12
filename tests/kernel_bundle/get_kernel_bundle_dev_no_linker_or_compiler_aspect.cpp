/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls these sycl::get_kernel_bundle overloads:
//    1) sycl::get_kernel_bundle<BundleState>(context, devices, selector)
//    2) sycl::get_kernel_bundle<BundleState>(context, selector)
//    3) sycl::get_kernel_bundle<BundleState>(context, devices)
//    4) sycl::get_kernel_bundle<BundleState>(context, kernelIds)
//    5) sycl::get_kernel_bundle<BundleState>(context, devices, kernelIds)
//    6) sycl::get_kernel_bundle<BundleState>(context)
//    7) sycl::get_kernel_bundle<KernelName, BundleState>(context)
//    8) sycl::get_kernel_bundle<KernelName, BundleState>(context, devices)
//  With bundle bundle_state::input if device doesn't has
//  sycl::aspect::online_compiler and with bundle_state::object if device
//  doesn't has sycl::aspect::online_linker.
//
//  The test verifies that an exception with sycl::errc::invalid is thrown if
//  device is not supported sycl::aspect::online_linker and/or
//  sycl::aspect::online_compiler aspects.
//
*******************************************************************************/

#include "../common/assertions.h"
#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "get_kernel_bundle.h"
#include "kernel_bundle.h"
#include "kernels.h"

#define TEST_NAME get_kernel_bundle_dev_no_linker_or_compiler_aspect

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::kernel_bundle;
using namespace sycl_cts::tests::get_kernel_bundle;

using kernel = simple_kernel_descriptor::type;

/** @brief Calls all get_kernel_bundle overloads in case when device does not
 *         support sycl::aspect::online_linker or sycl::aspect::online_compiler
 *         aspects and verifies that exception was thrown
 *  @tparam BundleState Bundle state
 *  @param log sycl_cts::util::logger class object
 *  @param queue sycl::queue class object
 */
template <sycl::bundle_state BundleState>
void run_test_for_all_overload_types(
    util::logger &log, sycl::queue &queue,
    const std::vector<sycl::kernel_id> &user_defined_kernel_ids) {
  static_assert(BundleState == sycl::bundle_state::input ||
                BundleState == sycl::bundle_state::object &&
                    "Provided bundle state not equal to input or object");

  const auto selector = [](const sycl::device_image<BundleState> &) {
    return true;
  };

  const auto context = queue.get_context();
  const auto device = queue.get_device();

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("(context, devices, selector)"),
      [&] {
        sycl::get_kernel_bundle<BundleState>(context, {device}, selector);
      });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("(context, selector)"),
      [&] { sycl::get_kernel_bundle<BundleState>(context, selector); });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("(context, devices, kernel_ids)"),
      [&] {
        sycl::get_kernel_bundle<BundleState>(context, {device},
                                             user_defined_kernel_ids);
      });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("(context, devices)"),
      [&] { sycl::get_kernel_bundle<BundleState>(context, {device}); });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("(context, kernel_ids)"), [&] {
        sycl::get_kernel_bundle<BundleState>(context, user_defined_kernel_ids);
      });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("(context)"),
      [&] { sycl::get_kernel_bundle<BundleState>(context); });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("<KernelName>(context)"),
      [&] { sycl::get_kernel_bundle<kernel, BundleState>(context); });

  expect_throws<sycl::errc::invalid>(
      log, TestCaseDescription<BundleState>("<KernelName>(context, devices)"),
      [&] { sycl::get_kernel_bundle<kernel, BundleState>(context, {device}); });
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
    auto queue = util::get_cts_object::queue();

    auto k_id = sycl::get_kernel_id<kernel>();
    std::vector<sycl::kernel_id> user_defined_kernel_ids{k_id};

    if (queue.get_device().has(sycl::aspect::online_compiler)) {
      log.skip(
          "Test for get_kernel_bundle<bundle_state::input> was skipped due to "
          "current device supports aspect::online_compiler");
    } else {
      run_test_for_all_overload_types<sycl::bundle_state::input>(
          log, queue, user_defined_kernel_ids);
    }
    if (queue.get_device().has(sycl::aspect::online_linker)) {
      log.skip(
          "Test for get_kernel_bundle<bundle_state::object> was skipped due to "
          "current device supports aspect::online_linker");
    } else {
      run_test_for_all_overload_types<sycl::bundle_state::object>(
          log, queue, user_defined_kernel_ids);
    }

    define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(
        queue);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
