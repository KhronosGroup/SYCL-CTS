/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::handler::use_kernel_bundle after calling
//  sycl::handler::set_specialization_constant with secondary queue.
//
//  The test verifies that the exception with sycl::errc::invalid was thrown.
//
*******************************************************************************/

#include "../common/common.h"
#include "use_kernel_bundle.h"

#define TEST_NAME \
  use_kernel_bundle_call_set_spec_const_before_use_kb_with_second_queue

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace tests_for_use_kernel_bundle;

constexpr sycl::specialization_id<int> spec_const;

/** @brief Call sycl::handler::set_specialization_constant after
 *         sycl::handler::use_kernel_bundle with secondary queue
 *  @param log sycl_cts::util::logger class object
 *  @param ctx Context that will be used for constructing sycl::queue
 *  @param kernel_bundle empty, non-empty or kernel_bundle with built-in kernels
 */
void run_verification(
    sycl_cts::util::logger &log, const sycl::context &ctx,
    sycl::kernel_bundle<sycl::bundle_state::executable> &kernel_bundle) {
  sycl::queue queue(ctx, ctx.get_devices()[0]);
  sycl::queue second_queue(ctx, ctx.get_devices()[0]);
  bool ex_was_thrown = false;
  try {
    queue.submit(
        [&](sycl::handler &cgh) {
          cgh.set_specialization_constant<spec_const>(0);
          cgh.use_kernel_bundle(kernel_bundle);
          cgh.single_task<kernel_for_kernel_bundle>([=]() {});
        },
        second_queue);
  } catch (const sycl::exception &e) {
    if (e.code() != sycl::errc::invalid) {
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
    run_verification(log, ctx, bundle_with_built_in_kernel);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
