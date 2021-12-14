/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::build(vector<kernel_bundle<>>, std::vector<device>).
//  For obtained kernel bundle with input state will use one device, but for
//  sycl::build will be provided vector with device, that was used for kernel
//  bundle and device from different platform.
//
//  The test verifies that the exception with sycl::errc::invalid was thrown.
//
*******************************************************************************/

#include "sycl_build.h"
#include "../common/assertions.h"
#include "../common/common.h"
#include "kernel_bundle.h"
#include "kernels.h"

#define TEST_NAME sycl_build_different_device_for_sycl_build

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::sycl_build;
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
    auto platforms = sycl::platform::get_platforms();
    if (platforms.size() <= 1) {
      log.note("Test skipped due to one platform was found");
      return;
    }
    sycl::queue q{platforms[0].get_devices()[0]};

    using kernel = simple_kernel_descriptor::type;
    const auto first_simple_kernel_id{sycl::get_kernel_id<kernel>()};
    auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        q.get_context(), platforms[0].get_devices(), {first_simple_kernel_id});

    std::vector<sycl::device> devices{platforms[0].get_devices()[0],
                                      platforms[1].get_devices()[0]};

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(kernel_bundle<>, vector<sycl::device>)"),
        [&] { sycl::build(kernel_bundle, devices); });

    define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(
        q, submit_kernel::yes);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
