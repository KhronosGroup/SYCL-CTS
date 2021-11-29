/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  The test calls sycl::has_kernel_bundle<sycl::bundle_state>(context, device,
//  kernelId) with each one of all available kernels in the application and
//  unite each call result to varaible with name "expected" (unition using "|="
//  operator), then calls sycl::has_kernel_bundle<sycl::bndle_state>(context).
//
//  The test verifies that variable value "expected" is equal to
//  sycl::has_kernel_bundle<sycl::bundle_state>(context) calling result.
//
//  IMPORTANT: This file should not be merged with another *.cpp file because we
//  we must have a strictly defined kernels in the application because tested
//  overload uses all available kernels and we can't chose specific kernel.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "has_kernel_bundle_multiple_kernels.h"

#define TEST_NAME has_kernel_bundle_multiple_kernels_ctx_only

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::has_kernel_bundle;

template <sycl::bundle_state BundleState>
using verifier = check::multiple_kernels<BundleState, overload::id::ctx_only>;

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

    verifier<sycl::bundle_state::input>::run(log, dev);
    verifier<sycl::bundle_state::object>::run(log, dev);
    verifier<sycl::bundle_state::executable>::run(log, dev);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
