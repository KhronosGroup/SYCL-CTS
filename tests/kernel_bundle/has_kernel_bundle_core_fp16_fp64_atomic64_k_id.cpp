/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls sycl::has_kernel_bundle<all bundle_state>(ctx, kernelIds)
//  for kernels with sycl::requires(has(sycl::aspect::fp16))/fp64/atomic64
//  requirements for all available devices.
//
//  The test verifies that the has_kernel_bundle result is equal to compatible
//  status for pair of kernel and device.
//
*******************************************************************************/

#include "../common/common.h"
#include "has_kernel_bundle_core.h"

#define TEST_NAME has_kernel_bundle_core_fp16_fp64_atomic64_k_id

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::has_kernel_bundle;

class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <typename T>
  using check_input_state =
      check::core<T, sycl::bundle_state::input, overload::id::ctx_kid>;
  template <typename T>
  using check_object_state =
      check::core<T, sycl::bundle_state::object, overload::id::ctx_kid>;
  template <typename T>
  using check_executable_state =
      check::core<T, sycl::bundle_state::executable, overload::id::ctx_kid>;

  /** execute the test
   */
  void run(util::logger &log) override {
    sycl::device dev = util::get_cts_object::device();
    sycl::context ctx(dev.get_platform().get_devices());

    for_all_types<check_input_state>(kernels_types_fp16_fp64_at64, log, ctx,
                                     dev);
    for_all_types<check_object_state>(kernels_types_fp16_fp64_at64, log, ctx,
                                      dev);
    for_all_types<check_executable_state>(kernels_types_fp16_fp64_at64, log,
                                          ctx, dev);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
