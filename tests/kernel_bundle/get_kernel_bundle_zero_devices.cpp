/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Check the sycl::get_kernel_bundle logic for the cases when:
//  - an empty device list was provided
//
*******************************************************************************/

#include "../common/common.h"
#include "get_kernel_bundle.h"

#define TEST_NAME get_kernel_bundle_zero_devices

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::kernel_bundle;
using namespace sycl_cts::tests::get_kernel_bundle;

template <sycl::bundle_state State>
void run_tests(util::logger &log, sycl::queue &queue) {
  const auto context = queue.get_context();
  const auto device = queue.get_device();

  log.debug([] {
    return "Running test for " + get_cts_string::for_bundle_state<State>() +
           " kernel bundle state...";
  });

  // Ensure we have some kernel in application to retrieve its id
  using kernelDescriptorT = typename kernels::simple_kernel_descriptor;
  define_kernel<kernelDescriptorT, State>(queue);

  using kernelNameT = typename kernelDescriptorT::type;
  const auto kernel_id = sycl::get_kernel_id<kernelNameT>();

  using descriptionT = TestCaseDescription<State>;
  const std::vector<sycl::device> zero_devices;

  // No checks for: sycl::get_kernel_bundle(context)
  // No checks for: sycl::get_kernel_bundle(context, kernel_ids)

  // Checks for: sycl::get_kernel_bundle(context, devices)
  expect_throws<sycl::errc::invalid>(
      log, descriptionT("with context, devices"),
      [&] { sycl::get_kernel_bundle<State>(context, zero_devices); });

  // Checks for: sycl::get_kernel_bundle(context, devices, kernel_ids)
  expect_throws<sycl::errc::invalid>(
      log, descriptionT("with context, devices, single kernel id"), [&] {
        sycl::get_kernel_bundle<State>(context, zero_devices, {kernel_id});
      });
  expect_throws<sycl::errc::invalid>(
      log, descriptionT("with context, devices, zero kernel ids"),
      [&] { sycl::get_kernel_bundle<State>(context, zero_devices, {}); });

  // No checks for: sycl::get_kernel_bundle<KernelName>(context)

  // Checks for: sycl::get_kernel_bundle<KernelName>(context, devices)
  expect_throws<sycl::errc::invalid>(
      log, descriptionT("with context, devices, kernel name"), [&] {
        sycl::get_kernel_bundle<kernelNameT, State>(context, zero_devices);
      });

  // No checks for: sycl::get_kernel_bundle(context, selector)

  // Checks for: sycl::get_kernel_bundle(context, devices, selector)
  expect_throws<sycl::errc::invalid>(
      log, descriptionT("with context, devices, selector"), [&] {
        const auto selector = [](const sycl::device_image<State> &) {
          return true;
        };

        sycl::get_kernel_bundle<State>(context, zero_devices, selector);
      });
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

    run_tests<sycl::bundle_state::executable>(log, queue);
    run_tests<sycl::bundle_state::input>(log, queue);
    run_tests<sycl::bundle_state::object>(log, queue);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
