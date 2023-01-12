/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::compile throwing exception in case zero devices
//  are passed
//
*******************************************************************************/

#include "../common/common.h"
#include "sycl_compile.h"
#include "kernel_bundle.h"

#define TEST_NAME sycl_compile_exception_for_zero_devices

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_compile;

/** test sycl::compile throwing exception for zero devices
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
    auto queue = util::get_cts_object::queue();

    if ( !sycl::has_kernel_bundle<sycl::bundle_state::input>(queue.get_context()) ) {
      SKIP("No kernel bundle with input state for test.");
    }

    auto input_kb = sycl::get_kernel_bundle<sycl::bundle_state::input>(queue.get_context());

    TestCaseDescription desc{"(kernel_bundle, {})"};
    expect_throws<sycl::errc::invalid>(log, desc, [&] {
      // Explicit type for vector of devices is required. Otherwise call is
      // ambiguous
      sycl::compile(input_kb, std::vector<sycl::device>{});
    });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
