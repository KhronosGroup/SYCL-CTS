/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::link(vector<kernel_bundle<>>, std::vector<device>,
//  property_list) with empty kernel bundle and empty device vector.
//
//  The test verifies that an exception with sycl::errc::invalid was thrown.
//
*******************************************************************************/

#include "../common/assertions.h"
#include "../common/common.h"
#include "kernels.h"
#include "sycl_link.h"

#define TEST_NAME sycl_link_zero_device_and_zero_bundle

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
    vector_with_object_bundles zero_bundle;

    const std::vector<sycl::device> zero_device;

    expect_throws<sycl::errc::invalid>(
        log,
        TestCaseDescription<sycl::bundle_state::executable>(
            "(vector<kernel_bundle>, empty vector<sycl::device>)"),
        [&] { sycl::link(zero_bundle, zero_device); });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
