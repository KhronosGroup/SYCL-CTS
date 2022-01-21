/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::build(kernel_bundle<>, std::vector<device>,
//  property_list) with empty device vector.
//
//  The test verifies that an exception with sycl::errc::invalid was thrown.
//
*******************************************************************************/

#include "../common/assertions.h"
#include "../common/common.h"
#include "kernel_bundle.h"
#include "kernels.h"
#include "sycl_build.h"

#define TEST_NAME sycl_build_zero_device

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
    auto q = util::get_cts_object::queue();

    std::vector<sycl::device> zero_device;
    const auto first_simple_kernel_id =
        sycl::get_kernel_id<first_simple_kernel>();

    auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        q.get_context(), {first_simple_kernel_id});

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(kernel_bundle<>, empty vector<sycl::device>)"),
        [&] { sycl::build(kernel_bundle, zero_device); });

    define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(
        q, submit_kernel::yes);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
