/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls these sycl::get_kernel_bundle overloads:
//    1) sycl::get_kernel_bundle<BundleState>(context, devices, selector)
//    2) sycl::get_kernel_bundle<BundleState>(context, selector)
//    3) sycl::get_kernel_bundle<BundleState>(context, devices)
//    4) sycl::get_kernel_bundle<BundleState>(context, kernelIds)
//    5) sycl::get_kernel_bundle<BundleState>(context, devices, kernelIds)
//    6) sycl::get_kernel_bundle<BundleState>(context)
//    7) sycl::get_kernel_bundle<KernelName, BundleState>(context)
//    8) sycl::get_kernel_bundle<KernelName, BundleState>(context, devices)
//  For kernels with CPU, GPU and accelerator requirements, likely unsupported
//  sub group size, likely supported sub group size, likely unsupported work
//  group groupsize, likely supported work group size and for two kernels
//  without any requirements for device from chosen platform.
//
//  The test verifies that the kernel bundle which is obtained from each
//  get_kernel_bundle overload contains only kernels that are compatible with
//  the provided device.
//
//  IMPORTANT: This file should not be merged with another *.cpp file because we
//  should have strictly defined kernels in application.
//
*******************************************************************************/

#include "get_kernel_bundle.h"

#define TEST_NAME get_kernel_bundle_with_kernel_attr

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::kernel_bundle;
using namespace sycl_cts::tests::kernel_bundle::get_kernel_bundle;

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

    if (!queue.get_device().has(sycl::aspect::online_linker) ||
        !queue.get_device().has(sycl::aspect::online_compiler)) {
      SKIP(
          "Device does not support online_linker or/and "
          "online_compiler aspect.");
    }

    std::vector<sycl::kernel_id> user_defined_kernel_ids{};

    // fill vector with kernel ids with for pre-defined kernels
    for_all_types<fill_vector_with_user_defined_kernel_ids>(
        kernels_with_attributes, user_defined_kernel_ids, queue.get_device());

    run_test_for_all_overload_types<sycl::bundle_state::executable>(
        log, queue, kernels_with_attributes, user_defined_kernel_ids);

    log.note(
        "Run test for sycl::get_kernel_bundle<KernelName,BundleState>(context) "
        "and sycl::get_kernel_bundle<KernelName, BundleState>(context, "
        "devices) overloads");
    for_all_types<run_tests_for_overloads_that_obtain_kernel_name>(
        kernels_with_attributes, log, queue);

    for_all_types<execute_kernel_and_verify_executions>(kernels_with_attributes,
                                                        log, queue);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
