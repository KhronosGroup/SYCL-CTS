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
//  For kernels that use fp16 and fp64 datatypes, a kernel that uses atomic_ref
//  (it requires atomic64 aspect) and for two kernels without any requirement
//  on device from the chosen platform.
//
//  The test verifies that no exception is thrown for bundle_state::input and
//  bundle_state::object.
//
*******************************************************************************/

#include "get_kernel_bundle.h"

#define TEST_NAME \
  get_kernel_bundle_without_kernel_attr_verify_that_exception_is_not_thrown

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::kernel_bundle;
using namespace sycl_cts::tests::kernel_bundle::get_kernel_bundle;

/** @brief Observing kernel bundle with provided bundle state for each overload
 *         that don't receive KernelName template parameter the then call
 *         "for_all_types" with "verify_that_kernel_in_bundle" function for
 *         check existence of each predefined kernel in result bundle
 *  @tparam BundleState sycl::bundle_state enum's field
 *  @param log sycl_cts::util::logger class object
 *  @param queue sycl::queue class object
 */
template <sycl::bundle_state BundleState>
void run_test_for_all_overload_types(
    util::logger &log, sycl::queue &queue,
    const std::vector<sycl::kernel_id> &user_defined_kernel_ids) {
  const auto context = queue.get_context();
  const auto device = queue.get_device();

  const auto always_device_image_selector =
      [](const sycl::device_image<BundleState> &) { return true; };

  expect_not_throws(
      log, TestCaseDescription<BundleState>("(context, devices, selector)"),
      [&] {
        sycl::get_kernel_bundle<BundleState>(context, {device},
                                             always_device_image_selector);
      });

  expect_not_throws(
      log, TestCaseDescription<BundleState>("(context, selector)"), [&] {
        sycl::get_kernel_bundle<BundleState>(context,
                                             always_device_image_selector);
      });

  expect_not_throws(
      log, TestCaseDescription<BundleState>("(context, devices, kernel_ids)"),
      [&] {
        sycl::get_kernel_bundle<BundleState>(context, {device},
                                             user_defined_kernel_ids);
      });

  expect_not_throws(
      log, TestCaseDescription<BundleState>("(context, devices)"),
      [&] { sycl::get_kernel_bundle<BundleState>(context, {device}); });

  expect_not_throws(
      log, TestCaseDescription<BundleState>("(context, kernel_ids)"), [&] {
        sycl::get_kernel_bundle<BundleState>(context, user_defined_kernel_ids);
      });

  expect_not_throws(log, TestCaseDescription<BundleState>("(context)"),
                    [&] { sycl::get_kernel_bundle<BundleState>(context); });

  expect_not_throws(
      log, TestCaseDescription<BundleState>("<KernelName>(context)"),
      [&] { sycl::get_kernel_bundle<kernel, BundleState>(context); });

  expect_not_throws(
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

    if (!queue.get_device().has(sycl::aspect::online_linker) or
        !queue.get_device().has(sycl::aspect::online_compiler)) {
      log.skip(
          "Test skipped due to device does not support online_linker or/and "
          "online_compiler aspect.");
      return;
    }

    std::vector<sycl::kernel_id> user_defined_kernel_ids{};

    // fill vector with kernel ids with for pre-defined kernels
    for_all_types<fill_vector_with_user_defined_kernel_ids>(
        kernels_without_attributes, user_defined_kernel_ids,
        queue.get_device());

    run_test_for_all_overload_types<sycl::bundle_state::input>(
        log, queue, user_defined_kernel_ids);
    run_test_for_all_overload_types<sycl::bundle_state::object>(
        log, queue, user_defined_kernel_ids);

    for_all_types<execute_kernel_and_verify_executions>(
        kernels_without_attributes, log, queue);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
