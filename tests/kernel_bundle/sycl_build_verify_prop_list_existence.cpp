/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides verifications that property list can be passed or omitted for
//  sycl::build overloads.
//
*******************************************************************************/

#include "../common/assertions.h"
#include "../common/common.h"
#include "kernel_bundle.h"
#include "kernels.h"

#define TEST_NAME sycl_build_verify_prop_list_existence

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
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

    using kernel = simple_kernel_descriptor::type;
    const auto first_simple_kernel_id = sycl::get_kernel_id<kernel>();

    auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
        q.get_context(), {first_simple_kernel_id});

    sycl::build(kernel_bundle, {q.get_device()}, sycl::property_list{});
    sycl::build(kernel_bundle, sycl::property_list{});

    sycl::build(kernel_bundle, {q.get_device()});
    sycl::build(kernel_bundle);

    define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(
        q, submit_kernel::yes);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
