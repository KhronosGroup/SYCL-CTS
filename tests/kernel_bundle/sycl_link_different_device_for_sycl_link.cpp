/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::link(vector<kernel_bundle<>>, std::vector<device>,
//  property_list) and sycl::link(kernel_bundle<>, std::vector<device>,
//  property_list). For obtained kernel bundle with object state will use
//  one device, but for sycl::link will be provided vector with device, that was
//  used for kernel bundle and different devices.
//
//  The test verifies that the exception with sycl::errc::invalid was thrown.
//
*******************************************************************************/

#include "sycl_link.h"
#include "../common/assertions.h"

#define TEST_NAME sycl_link_different_device_for_sycl_link

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
      log.note("Test skipped due to one device was found");
      return;
    }

    sycl::queue q{devices[0]};

    const auto first_simple_kernel_id =
        sycl::get_kernel_id<first_simple_kernel>();
    auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::object>(
        q.get_context(), {first_simple_kernel_id});
    vector_with_object_bundles vector_with_kb{kernel_bundle};

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(vector<kernel_bundle>, vector<sycl::device>)"),
        [&] { sycl::link(vector_with_kb, devices); });

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(kernel_bundle, vector<sycl::device>)"),
        [&] { sycl::link(kernel_bundle, devices); });

    define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(q);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
