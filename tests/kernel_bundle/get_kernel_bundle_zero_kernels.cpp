/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Check the sycl::get_kernel_bundle logic for the cases when:
//  - an empty kernel ids' list was provided, but
//  - device list is not empty
//
//  For the following sycl::get_kernel_bundle overloads:
//    1) sycl::get_kernel_bundle(context)
//    2) sycl::get_kernel_bundle(context, kernel_ids)
//    3) sycl::get_kernel_bundle(context, devices)
//    4) sycl::get_kernel_bundle(context, devices, kernel_ids)
//    5) sycl::get_kernel_bundle(context, selector)
//    6) sycl::get_kernel_bundle(context, devices, selector)
//
//  IMPORTANT: This file should not be merged with another *.cpp file because we
//  should have no kernel in application.
//
*******************************************************************************/

#include "../common/common.h"
#include "get_kernel_bundle.h"

#define TEST_NAME get_kernel_bundle_zero_kernels

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::kernel_bundle;
using namespace sycl_cts::tests::kernel_bundle::get_kernel_bundle;

template <sycl::bundle_state State>
void run_tests(util::logger &log, sycl::queue &queue) {
  const auto context = queue.get_context();
  const auto device = queue.get_device();

  log.debug([] {
    return "Running test for " + get_cts_string::for_bundle_state<State>() +
           " kernel bundle state...";
  });

  // Verify we can run tests using the device provided
  {
    sycl_cts::util::kernel_restrictions restrictions;
    restrictions.set_aspects(get_bundle_state_aspects<State>());

    if (!restrictions.is_compatible(device)) {
      log.skip("Test for " + get_cts_string::for_bundle_state<State>() +
               " kernel bundle state "
               "skipped. Device does not support " +
               restrictions.to_string());
      return;
    }
  }

  using descriptionT = TestCaseDescription<State>;
  const std::vector<sycl::kernel_id> zero_kernel_ids{};
  const std::vector<sycl::device> devices{device};
  const auto selector = [](const sycl::device_image<State> &) { return true; };

  // Checks for: sycl::get_kernel_bundle(context)
  expect_not_throws(log, descriptionT("with context"),
                    [&] { sycl::get_kernel_bundle<State>(context); });
  // Checks for: sycl::get_kernel_bundle(context, kernel_ids)
  expect_not_throws(log, descriptionT("with context, kernel ids"), [&] {
    sycl::get_kernel_bundle<State>(context, zero_kernel_ids);
  });
  // Checks for: sycl::get_kernel_bundle(context, devices)
  expect_not_throws(log, descriptionT("with context, devices"),
                    [&] { sycl::get_kernel_bundle<State>(context, devices); });
  // Checks for: sycl::get_kernel_bundle(context, devices, kernel_ids)
  expect_not_throws(
      log, descriptionT("with context, devices, kernel ids"), [&] {
        sycl::get_kernel_bundle<State>(context, devices, zero_kernel_ids);
      });

  // No checks for: sycl::get_kernel_bundle<KernelName>(context)
  // No checks for: sycl::get_kernel_bundle<KernelName>(context, devices)

  // Checks for: sycl::get_kernel_bundle(context, selector)
  expect_not_throws(log, descriptionT("with context, selector"),
                    [&] { sycl::get_kernel_bundle<State>(context, selector); });
  // Checks for: sycl::get_kernel_bundle(context, devices, selector)
  expect_not_throws(log, descriptionT("with context, devices, selector"), [&] {
    sycl::get_kernel_bundle<State>(context, devices, selector);
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

    // Ensure we have no user-defined kernels in a whole application
    const auto kernel_ids = sycl::get_kernel_ids();
    if (!kernel_ids.empty()) {
      FAIL(log,
           "Test precondition failed: at least one user-defined kernel found");
    }

    run_tests<sycl::bundle_state::executable>(log, queue);
    run_tests<sycl::bundle_state::input>(log, queue);
    run_tests<sycl::bundle_state::object>(log, queue);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
