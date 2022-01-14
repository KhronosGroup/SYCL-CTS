/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  The test checks that calling sycl::handler::use_kernel_bundle with a kernel
//  bundle which is associated with a context different from one associated with
//  a handler results in exception. This is done for kernel bundles that:
//    1) contain user kernels
//    2) contain builtin kernels
//    3) don't contain any kernel
//
//  The test verifies that the exception with sycl::errc::invalid was thrown.
//
*******************************************************************************/

#include "../common/common.h"
#include "use_kernel_bundle.h"
#include "use_kernel_bundle_queue_ctx_not_equal_to_kernel_bundle_ctx.h"

#define TEST_NAME \
  use_kernel_bundle_second_queue_ctx_not_equal_to_kernel_bundle_ctx

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::use_kernel_bundle;
using namespace use_kernel_bundle_queue_ctx_not_equal_kb_ctx;

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
    const std::vector<sycl::device> devices{sycl::device::get_devices()};
    if (devices.size() <= 1) {
      log.note("Test skipped due to only zero or one device was found");
      return;
    }
    sycl::context first_ctx(devices[0]);
    sycl::context second_ctx(devices[1]);

    sycl::kernel_bundle<sycl::bundle_state::executable> empty_bundle =
        get_empty_bundle(first_ctx);
    run_verification(log, first_ctx, second_ctx, empty_bundle);

    sycl::kernel_bundle<sycl::bundle_state::executable> non_empty_bundle =
        get_non_empty_bundle<kernel_for_kernel_bundle>(first_ctx);
    run_verification(log, first_ctx, second_ctx, non_empty_bundle);

    sycl::kernel_bundle<sycl::bundle_state::executable>
        bundle_with_built_in_kernel =
            get_bundle_with_built_in_kernels(first_ctx);
    run_verification(log, first_ctx, second_ctx, bundle_with_built_in_kernel);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
