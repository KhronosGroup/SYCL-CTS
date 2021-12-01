/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::compile throwing exception in case device is not
//  in set of kernel_bundle::get_devices()
//
*******************************************************************************/

#include "../common/common.h"
#include "sycl_compile.h"
#include "kernel_bundle.h"

#define TEST_NAME sycl_compile_exception_if_dev_not_in_set

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_compile;

/** test sycl::compile throwing exception in case device is not in set of
 *  kernel_bundle::get_device()
 */
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
    sycl::context ctx(platforms[1].get_devices());

    if ( !sycl::has_kernel_bundle<sycl::bundle_state::input>(ctx) ) {
      log.note("No kernel bundle with input state for test (skipped).");
      return;
    }

    auto input_kb = sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx);

    TestCaseDescription desc{"(kernel_bundle, devices)"};
    expect_throws<sycl::errc::invalid>(log, desc, [&]() {

      sycl::compile(input_kb, {platforms[0].get_devices()[0]});
    });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
