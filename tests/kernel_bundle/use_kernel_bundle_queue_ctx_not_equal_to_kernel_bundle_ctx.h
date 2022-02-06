/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides verification function for the case when first/second sycl::queue
//  context not equal to kernel bundle context.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_USE_KERNEL_BUNDLE_QUEUE_CTX_NOT_EQUAL_KB_CTX_H
#define __SYCLCTS_TESTS_USE_KERNEL_BUNDLE_QUEUE_CTX_NOT_EQUAL_KB_CTX_H

#include "../common/assertions.h"
#include "../common/common.h"
#include "use_kernel_bundle.h"

namespace use_kernel_bundle_queue_ctx_not_equal_kb_ctx {

/** @brief Call sycl::handler::use_kernel_bundle with kernel bundle which
 *         context doesent associated with the sycl::handler context
 *  @param log sycl_cts::util::logger class object
 *  @param first_ctx Context that will used for sycl::queue
 *  @param second_ctx Context that will used for sycl::queue
 *  @param kernel_bundle which will be set using use_kernel_bundle
 */
inline void run_verification(
    sycl_cts::util::logger &log, const sycl::context &first_ctx,
    const sycl::context &second_ctx,
    sycl::kernel_bundle<sycl::bundle_state::executable> &kernel_bundle) {
  sycl::queue first_queue(first_ctx, first_ctx.get_devices()[0]);
  sycl::queue second_queue(second_ctx, second_ctx.get_devices()[0]);

  expect_throws<sycl::errc::invalid>(
      log,
      sycl_cts::tests::use_kernel_bundle::TestCaseDescription<
          sycl::bundle_state::executable>("(kernel_bundle<>)"),
      [&] {
        first_queue.submit(
            [&](sycl::handler &cgh) {
              cgh.use_kernel_bundle(kernel_bundle);
              cgh.single_task<
                  sycl_cts::tests::use_kernel_bundle::kernel_for_kernel_bundle>(
                  [=]() {});
            },
            second_queue);
      });
}

}  // namespace use_kernel_bundle_queue_ctx_not_equal_kb_ctx

#endif  // __SYCLCTS_TESTS_USE_KERNEL_BUNDLE_QUEUE_CTX_NOT_EQUAL_KB_CTX_H
