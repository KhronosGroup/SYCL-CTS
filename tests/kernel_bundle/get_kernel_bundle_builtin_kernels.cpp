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
//  In this tests, use two user-defined kernels without any requirements for
//  device from chosen platform.
//
//  This test verifies that these overloads:
//    1) sycl::get_kernel_bundle<BundleState>(context)
//    2) sycl::get_kernel_bundle<BundleState>(context, devices)
//    3) sycl::get_kernel_bundle<BundleState>(context, devices, selector)
//    4) sycl::get_kernel_bundle<BundleState>(context, selector)
//  Do not contain any built-in kernel.
//  This test verifies that these overloads:
//    1) sycl::get_kernel_bundle<BundleState>(context, kernelIds)
//    2) sycl::get_kernel_bundle<BundleState>(context, devices, kernelIds)
//  Contain built-in kernels (if built-in kernel_ids were provided) and contain
//  built-in kernels + user-defined kernels (if built-in kernel_ids +
//  user-defined kernel_ids were provided).
//
*******************************************************************************/

#include "../common/common.h"
#include "get_kernel_bundle.h"
#include "kernels.h"
// for std::any_of
#include <algorithm>

#define TEST_NAME get_kernel_bundle_builtin_kernels

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::kernel_bundle::get_kernel_bundle;
using namespace sycl_cts::tests::kernel_bundle;

using first_simple_kernel = simple_kernel_descriptor::type;
using second_simple_kernel = simple_kernel_descriptor_second::type;

const auto first_simple_kernel_id = sycl::get_kernel_id<first_simple_kernel>();
const auto second_simple_kernel_id =
    sycl::get_kernel_id<second_simple_kernel>();
const std::vector<sycl::kernel_id> user_defined_kernel_ids{
    first_simple_kernel_id, second_simple_kernel_id};

template <sycl::bundle_state BundleState>
struct has_kernel {
  sycl::kernel_bundle<BundleState> m_k_bundle;
  has_kernel(sycl::kernel_bundle<BundleState> k_bundle)
      : m_k_bundle(k_bundle) {}

  bool operator()(const sycl::kernel_id &kernel_id) {
    return m_k_bundle.has_kernel(kernel_id);
  }
};

/** @brief Do some verifications for described sycl::get_kernel_bundle overloads
 *  @tparam BundleState sycl::bundle_state enumeration's field
 *  @param log sycl_cts::util::logger class object
 *  @param queue sycl::queue class object
 */
template <sycl::bundle_state BundleState>
void run_tests(util::logger &log, sycl::queue &queue) {
  const auto context{queue.get_context()};
  const auto device{queue.get_device()};
  bool selector_was_called_for_built_in{false};

  const std::vector<sycl::kernel_id> built_in_kernel_ids{
      device.template get_info<sycl::info::device::built_in_kernel_ids>()};

  const auto always_device_image_selector =
      [](const sycl::device_image<BundleState> &) { return true; };
  const auto no_built_in_device_image_selector_verifier =
      [&](const sycl::device_image<BundleState> &device_img) {
        for (auto &built_in_kernel_id : built_in_kernel_ids) {
          if (!device_img.has_kernel(built_in_kernel_id)) {
            selector_was_called_for_built_in = true;
            break;
          }
        }
        return true;
      };

  // Verify we can run tests using the device provided
  {
    sycl_cts::util::kernel_restrictions restrictions{};
    restrictions.set_aspects(get_bundle_state_aspects<BundleState>());

    if (!restrictions.is_compatible(device)) {
      log.note("Test for " + get_cts_string::for_bundle_state<BundleState>() +
               " kernel bundle state "
               "skipped. Device does not support " +
               restrictions.to_string());
      return;
    }
  }

  const std::vector<sycl::device> devices{device};

  log.note(
      "Run test for sycl::get_kernel_bundle<BundleState>(context) overload");
  // verify that obtained kernel bundle does not contain built-in kernels
  {
    auto k_bundle = sycl::get_kernel_bundle<BundleState>(context);

    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                    has_kb_verifier)) {
      FAIL(log, "Obtained kernel bundle has built-in kernel");
    }
  }

  log.note(
      "Run test for sycl::get_kernel_bundle<BundleState>(context, kernel_ids) "
      "overload");
  // with built-in kernels only
  {
    auto k_bundle =
        sycl::get_kernel_bundle<BundleState>(context, built_in_kernel_ids);

    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (!std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                     has_kb_verifier) &&
        !built_in_kernel_ids.empty()) {
      FAIL(log,
           "Obtained kernel bundle does not have one or more built-in kernels");
    }
  }

  // with built-in and user-defined kernels
  {
    auto all_available_kernels = built_in_kernel_ids;
    std::copy(user_defined_kernel_ids.begin(), user_defined_kernel_ids.end(),
              std::back_inserter(all_available_kernels));
    auto k_bundle =
        sycl::get_kernel_bundle<BundleState>(context, all_available_kernels);

    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (!std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                     has_kb_verifier) &&
        !built_in_kernel_ids.empty()) {
      FAIL(log,
           "Obtained kernel bundle does not have one or more built-in kernels");
    }

    if (!std::any_of(user_defined_kernel_ids.cbegin(),
                     user_defined_kernel_ids.cend(), has_kb_verifier)) {
      FAIL(log,
           "Obtained kernel bundle does not have one or more user-defined "
           "kernels without any requires");
    }
  }

  log.note(
      "Run test for sycl::get_kernel_bundle<BundleState>(context, devices) "
      "overload");
  // verify that obtained kernel bundle does not contain built-in kernels
  {
    auto k_bundle = sycl::get_kernel_bundle<BundleState>(context, devices);

    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                    has_kb_verifier)) {
      FAIL(log, "Obtained kernel bundle has built-in kernel");
    }
  }

  log.note(
      "Run test for sycl::get_kernel_bundle<BundleState>(context, devices, "
      "kernel_ids) overload");
  // with built-in kernels only
  {
    auto k_bundle =
        sycl::get_kernel_bundle<BundleState>(context, built_in_kernel_ids);
    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (!std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                     has_kb_verifier) &&
        !built_in_kernel_ids.empty()) {
      FAIL(log,
           "Obtained kernel bundle does not have one or more built-in kernels");
    }
  }
  // with built-in and user-defined kernels
  {
    auto all_available_kernels = built_in_kernel_ids;
    std::copy(user_defined_kernel_ids.begin(), user_defined_kernel_ids.end(),
              std::back_inserter(all_available_kernels));
    auto k_bundle =
        sycl::get_kernel_bundle<BundleState>(context, all_available_kernels);
    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (!std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                     has_kb_verifier) &&
        !built_in_kernel_ids.empty()) {
      FAIL(log,
           "Obtained kernel bundle does not have one or more built-in kernels");
    }

    if (!std::any_of(user_defined_kernel_ids.cbegin(),
                     user_defined_kernel_ids.cend(), has_kb_verifier)) {
      FAIL(log,
           "Obtained kernel bundle does not have one or more user-defined "
           "kernels without any requires");
    }
  }

  log.note(
      "Run test for sycl::get_kernel_bundle<BundleState>(context, selector) "
      "overload");
  // verify that obtained kernel bundle does not contain built-in kernels
  {
    auto k_bundle = sycl::get_kernel_bundle<BundleState>(
        context, always_device_image_selector);

    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                    has_kb_verifier)) {
      FAIL(log, "Obtained kernel bundle has built-in kernel");
    }
  }
  // selector should not be called for built-in kernels
  {
    selector_was_called_for_built_in = false;

    auto k_bundle = sycl::get_kernel_bundle<BundleState>(
        context, no_built_in_device_image_selector_verifier);

    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                    has_kb_verifier)) {
      FAIL(log, "Obtained kernel bundle has built-in kernel");
    }
    if (selector_was_called_for_built_in) {
      FAIL(log,
           "Kernel bundle was called for device image with built-in kernel");
    }
  }

  log.note(
      "Run test for sycl::get_kernel_bundle<BundleState>(context, devices, "
      "selector) overload");
  // verify that obtained kernel bundle does not contain built-in kernels
  {
    auto k_bundle = sycl::get_kernel_bundle<BundleState>(
        context, devices, always_device_image_selector);

    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                    has_kb_verifier)) {
      FAIL(log, "Obtained kernel bundle has built-in kernel");
    }
  }
  // selector should not be called for built-in kernels
  {
    selector_was_called_for_built_in = false;

    auto k_bundle = sycl::get_kernel_bundle<BundleState>(
        context, devices, no_built_in_device_image_selector_verifier);

    has_kernel<BundleState> has_kb_verifier(k_bundle);
    if (std::any_of(built_in_kernel_ids.cbegin(), built_in_kernel_ids.cend(),
                    has_kb_verifier)) {
      FAIL(log, "Obtained kernel bundle has built-in kernel");
    }
    if (selector_was_called_for_built_in) {
      FAIL(log,
           "Kernel bundle was callsed for device image with built-in kernel");
    }
  }
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

    define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(
        queue);
    define_kernel<simple_kernel_descriptor_second,
                  sycl::bundle_state::executable>(queue);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
