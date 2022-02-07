/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification for invoking kernel that not contained in kernel bundle
//  the test do not used second queue. The test will be skipped if kernel for
//  execution will be found in obtained kernel bundle.
//
//  The test verifies that the exception with sycl::errc::kernel_not_supported
//  was thrown.
//
*******************************************************************************/

#include "../common/common.h"
#include "use_kernel_bundle.h"

#define TEST_NAME use_kernel_bundle_invoke_kernel_without_bundle_no_second_queue

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::use_kernel_bundle;

/** @brief Call use_kernel_bundle and invoke the kernel that not contained in
 *         provided kernel bundle
 *  @param log sycl_cts::util::logger class object
 *  @param ctx Context that will be used for constructing sycl::queue
 *  @param kernel_bundle empty, non-empty or kernel_bundle with built-in kernels
 */
void run_verification(
    sycl_cts::util::logger &log, const sycl::context &ctx,
    sycl::kernel_bundle<sycl::bundle_state::executable> &kernel_bundle) {
  sycl::queue queue(ctx, ctx.get_devices()[0]);

  auto kernel_id = sycl::get_kernel_id<kernel>();
  if (kernel_bundle.has_kernel(kernel_id)) {
    log.note(
        "Test skipped due to kernel bundle contain kernel that shouldn't be in "
        "this bundle");
    return;
  }

  bool ex_was_thrown = false;
  try {
    queue.submit([&](sycl::handler &cgh) {
      cgh.use_kernel_bundle(kernel_bundle);
      cgh.single_task<kernel>([=]() {});
    });

    // call queue::submit for have kernel in application and have possible to
    // use this kernel for receive kernel bundle
    queue.submit([&](sycl::handler &cgh) {
      cgh.single_task<kernel_for_kernel_bundle>([=] {});
    });
  } catch (const sycl::exception &e) {
    if (e.code() != sycl::errc::kernel_not_supported) {
      FAIL(log, unexpected_exception_msg);
      throw;
    }
    ex_was_thrown = true;
  }

  if (!ex_was_thrown) {
    FAIL(log, "Exception was not thrown");
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
    sycl::device dev = util::get_cts_object::device();
    sycl::context ctx(dev.get_platform().get_devices());

    sycl::kernel_bundle<sycl::bundle_state::executable> empty_bundle =
        get_empty_bundle(ctx);
    run_verification(log, ctx, empty_bundle);

    sycl::kernel_bundle<sycl::bundle_state::executable> non_empty_bundle =
        get_non_empty_bundle<kernel_for_kernel_bundle>(ctx);
    run_verification(log, ctx, non_empty_bundle);

    sycl::kernel_bundle<sycl::bundle_state::executable>
        bundle_with_built_in_kernel = get_bundle_with_built_in_kernels(ctx);
    if (bundle_with_built_in_kernel.empty()) {
      log.note(skip_test_for_builtin_kernels_msg);
      return;
    }
    run_verification(log, ctx, bundle_with_built_in_kernel);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
