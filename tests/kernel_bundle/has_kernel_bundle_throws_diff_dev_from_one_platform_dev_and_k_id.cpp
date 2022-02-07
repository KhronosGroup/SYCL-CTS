/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::has_kernel_bundle<all bundle_state>(context, devices,
//  kernelIds) with device that are not associated with provided context
//  (devices from same platform).
//
//  The test verifies that the exceptions with sycl::errc::invalid is thrown.
//
*******************************************************************************/

#include "../common/common.h"
#include "has_kernel_bundle_throws.h"

#define TEST_NAME \
  has_kernel_bundle_throws_diff_dev_from_one_platform_dev_and_k_id

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::has_kernel_bundle;
using namespace kernels;

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
    const auto platforms = sycl::platform::get_platforms();
    for (const auto &platform : platforms) {
      auto devices{platform.get_devices()};
      if (devices.size() <= 1) {
        log.note("Platform skipped due to one device was found");
        continue;
      }
      sycl::context ctx(devices[1]);

      check::throws<simple_kernel_descriptor, sycl::bundle_state::input,
                    overload::id::ctx_dev_kid>{}(log, ctx,
                                                 platforms[0].get_devices()[0]);
      check::throws<simple_kernel_descriptor, sycl::bundle_state::object,
                    overload::id::ctx_dev_kid>{}(log, ctx,
                                                 platforms[0].get_devices()[0]);
      check::throws<simple_kernel_descriptor, sycl::bundle_state::executable,
                    overload::id::ctx_dev_kid>{}(log, ctx,
                                                 platforms[0].get_devices()[0]);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
