/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::has_kernel_bundle<kernel_name, bundle_state::input>(
//  context, devices) for kernel with sycl::aspect::online_compiler
//  (if bundle_state::input) and for sycl::aspect::linker (if
//  bundle_state::object).
//
//  The test verifies that the has_kernel_bundle result is equal to compatible
//  status for pair of kernel and device.
//
*******************************************************************************/

#include "../common/common.h"
#include "has_kernel_bundle_core.h"

#define TEST_NAME has_kernel_bundle_core_all_states_dev_and_k_name

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::has_kernel_bundle;
using namespace check;
using namespace kernels;

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

    core<simple_kernel_descriptor, sycl::bundle_state::input,
         overload::id::ctx_dev_kname>{}(log, ctx, dev,
                                        "simple_kernel_descriptor");
    core<simple_kernel_descriptor, sycl::bundle_state::object,
         overload::id::ctx_dev_kname>{}(log, ctx, dev,
                                        "simple_kernel_descriptor");
    core<simple_kernel_descriptor, sycl::bundle_state::executable,
         overload::id::ctx_dev_kname>{}(log, ctx, dev,
                                        "simple_kernel_descriptor");
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
