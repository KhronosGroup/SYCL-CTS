/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::link(vector<kernel_bundle<>>, std::vector<device>,
//  property_list); sycl::link(vector<kernel_bundle<>>, property_list);
//  sycl::link(kernel_bundle<>, std::vector<device>, property_list) and
//  sycl::link(kernel_bundle<>, property_list) overloads with two kernel bundles
//  that were obtained using different contexts.
//
//  The test verifies that an exception with sycl::errc::invalid was thrown.
//
*******************************************************************************/

#include "../common/assertions.h"
#include "../common/common.h"
#include "kernels.h"
#include "sycl_link.h"

#define TEST_NAME sycl_link_kernel_bundles_with_diff_ctx

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::sycl_link;
using namespace sycl_cts::tests::kernel_bundle;

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
      SKIP("Requires at least two devices");
    }

    sycl::context first_ctx(devices[0]);
    sycl::context second_ctx(devices[1]);
    sycl::queue q{devices[0]};

    auto first_kb = sycl::get_kernel_bundle<sycl::bundle_state::object>(
        first_ctx, first_ctx.get_devices());
    auto second_kb = sycl::get_kernel_bundle<sycl::bundle_state::object>(
        second_ctx, second_ctx.get_devices());

    std::vector<sycl::kernel_bundle<sycl::bundle_state::object>>
        kb_with_diff_ctx{first_kb, second_kb};

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(vector<kernel_bundle>, vector<sycl::device>)"),
        [&] { sycl::link(kb_with_diff_ctx, first_ctx.get_devices()); });
    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(vector<kernel_bundle>)"),
        [&] { sycl::link(kb_with_diff_ctx); });
    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(kernel_bundle, vector<sycl::device)>"),
        [&] {
          sycl::link(sycl::join<sycl::bundle_state::object>(kb_with_diff_ctx),
                     first_ctx.get_devices());
        });
    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>("(kernel_bundle)"),
        [&] {
          sycl::link(sycl::join<sycl::bundle_state::object>(kb_with_diff_ctx));
        });

    define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(q);
    define_kernel<simple_kernel_descriptor_second,
                  sycl::bundle_state::executable>(q);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
