/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::link(vector<kernel_bundle<>>, std::vector<device>,
//  property_list) and  sycl::link(kernel_bundle<>, std::vector<device>,
//  property_list) with empty device vector.
//
//  The test verifies that an exception with sycl::errc::invalid was thrown.
//
*******************************************************************************/

#include "../common/assertions.h"
#include "../common/common.h"
#include "kernels.h"
#include "sycl_link.h"

#define TEST_NAME sycl_link_zero_device

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::sycl_link;
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

    const auto first_simple_kernel_id =
        sycl::get_kernel_id<first_simple_kernel>();

    auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::object>(
        q.get_context(), {first_simple_kernel_id});
    vector_with_object_bundles vector_with_kb{kernel_bundle};

    std::vector<sycl::device> zero_device;
    
    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(vector<kernel_bundle>, empty vector<sycl::device>)"),
        [&] { sycl::link(vector_with_kb, zero_device); });

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(kernel_bundle, empty vector<sycl::device>)"),
        [&] { sycl::link(kernel_bundle, zero_device); });

    define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(q);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
