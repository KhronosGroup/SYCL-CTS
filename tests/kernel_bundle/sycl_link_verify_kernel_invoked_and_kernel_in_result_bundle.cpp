/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls this sycl::link overloads:
//    1) sycl::link(vector<kernel_bundle<>>, std::vector<device>, property_list)
//    2) sycl::link(vector<kernel_bundle<>>, property_list)
//    3) sycl::link(kernel_bundle<>, std::vector<device>, property_list)
//    4) sycl::link(kernel_bundle<>, property_list)
//  For kernels with CPU, GPU and accelerator requirements, kernels that use
//  fp16 and fp64 data types, a kernel that uses atomic_ref (it requires
//  atomic64 aspect, but this kernel don't use sycl::requires) and two kernels
//  without any requirement, on device from the chosen platform.
//
//  In this tests kernel_bundles<sycl::bundle_state::object> will be obtained by
//  three ways:
//    1) Obtain bundles in executable state.
//    2) Obtain bundles in object state and then call sycl::compile to obtain
//      kernel bundle in object state.
//    3) Construct std::vector with bundles that obtain from two previous ways.
//  For different overloads we obtain kernel bundles by different ways:
//  1) For overload that obtain std::vector<kernel_bundle> we provide vector
//    with all bundles that receive by three ways that described above.
//  2) For overload that obtain single kernel bundle, call sycl::join to merge
//    all bundles in one and then provide it to sycl::link.
//
//
//  The test verifies that the obtained kernel bundle in executable state
//  contain all provided kernels (only in case when kernel compatible with
//  current device, otherwise this kernel should not be presented in result
//  bundle).
//  This test also verifies that kernel bundle context is equal to provided
//  context, provided kernels were invoked and devices from result kernel bundle
//  are equal to provided devices with duplicates removed.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "kernels.h"
#include "sycl_link.h"

#define TEST_NAME sycl_link_verify_kernel_invoked_and_kernel_in_result_bundle

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::kernel_bundle;
using namespace sycl_cts::tests::sycl_link;

/** @brief Provide verification for provided bundle that it has kernels without
 *         any requirements, that kernel with CPU, GPU or accelerator
 *         requirements presented in bundle if this kernel compatible with
 *         current device. Also this function verifies that context, provided to
 *         kernel_bundle<bundle_state::input> is equal to obtained bundle
 *         context and that devices from obtained bundle are equal to provided
 *         devices.
 *  @param log sycl_cts::util::logger class object
 *  @param dev_vector std::vector with sycl::device
 *  @param ctx sycl::context that will be comparing with context from obtained
 *         bundle
 *  @param kernel_bundle sycl::kernel bundle that was obtained from
 *         sycl::build
 */
void verify_results(
    util::logger &log, const std::vector<sycl::device> &dev_vector,
    const sycl::context &ctx,
    const sycl::kernel_bundle<sycl::bundle_state::executable> &kernel_bundle) {
  for_all_types<verify_that_kernel_in_bundle>(kernels_for_link_and_build, log,
                                              kernel_bundle, dev_vector);
  if (kernel_bundle.get_context() != ctx) {
    FAIL(log, "Kernel bundle's context does not equal to provided context");
  }
  if (kernel_bundle.get_devices() != dev_vector) {
    FAIL(log, "Devices from kernel bundle not equal to provided devices");
  }
}

/** @brief For more fully coverage this function obtain sycl::kernel_bundle by
 *         two ways: obtain bundle in object state and obtain kernel bundle in
 *         input state and then calls sycl::compile to move it to object state,
 *         then obtain vector with bundles and calls sycl::link for this vectors
 *         (for overloads that receive one kernel bundle calls sycl::join to
 *         merge all bundle into one). Then do some verifications for obtained
 *         bundle in executable state.
 *  @param log sycl_cts::util::logger class object
 *  @param queue sycL::queue class object
 */
void run_verification(util::logger &log, sycl::queue &queue) {
  auto ctx = queue.get_context();

  const sycl::kernel_id first_simple_kernel_id{
      sycl::get_kernel_id<first_simple_kernel>()};
  const sycl::kernel_id second_simple_kernel_id{
      sycl::get_kernel_id<second_simple_kernel>()};

  auto kb_with_first_simple_kernel{
      sycl::get_kernel_bundle<sycl::bundle_state::object>(
          ctx, {first_simple_kernel_id})};
  auto kb_with_second_simple_kernel{
      sycl::get_kernel_bundle<sycl::bundle_state::object>(
          ctx, {second_simple_kernel_id})};

  auto kb_with_first_simple_kernel_from_input{
      sycl::compile(sycl::get_kernel_bundle<sycl::bundle_state::input>(
          ctx, {first_simple_kernel_id}))};
  auto kb_with_second_simple_kernel_from_input{
      sycl::compile(sycl::get_kernel_bundle<sycl::bundle_state::input>(
          ctx, {second_simple_kernel_id}))};

  vector_with_object_bundles kernel_bundles{
      kb_with_first_simple_kernel, kb_with_second_simple_kernel};
  vector_with_object_bundles kernel_bundles_from_input{
      kb_with_first_simple_kernel_from_input,
      kb_with_second_simple_kernel_from_input};
  vector_with_object_bundles mixed_kernel_bundles{
      kb_with_first_simple_kernel_from_input,
      kb_with_second_simple_kernel};

  std::vector<sycl::device> dev_vector{ctx.get_devices()};
  std::vector<sycl::device> current_dev_vector{util::get_cts_object::device()};

  log.note("Verify link(vector<kernel_bundle<>>, vector<device>) overload");
  verify_results(log, current_dev_vector, ctx,
                 sycl::link(kernel_bundles, current_dev_vector));
  verify_results(log, current_dev_vector, ctx,
                 sycl::link(kernel_bundles_from_input, current_dev_vector));
  verify_results(log, current_dev_vector, ctx,
                 sycl::link(mixed_kernel_bundles, current_dev_vector));

  log.note("Verify link(vector<kernel_bundle<>>) overload");
  verify_results(log, dev_vector, ctx, sycl::link(kernel_bundles));
  verify_results(log, dev_vector, ctx, sycl::link(kernel_bundles_from_input));
  verify_results(log, dev_vector, ctx, sycl::link(mixed_kernel_bundles));

  log.note("Verify link(kernel_bundle<>, vector<device>) overload");
  verify_results(
      log, current_dev_vector, ctx,
      sycl::link(sycl::join<sycl::bundle_state::object>(kernel_bundles),
                 current_dev_vector));
  verify_results(log, current_dev_vector, ctx,
                 sycl::link(sycl::join<sycl::bundle_state::object>(
                                kernel_bundles_from_input),
                            current_dev_vector));
  verify_results(
      log, current_dev_vector, ctx,
      sycl::link(sycl::join<sycl::bundle_state::object>(mixed_kernel_bundles),
                 current_dev_vector));

  log.note("Verify link(kernel_bundle<>) overload");
  verify_results(
      log, dev_vector, ctx,
      sycl::link(sycl::join<sycl::bundle_state::object>(kernel_bundles)));
  verify_results(log, dev_vector, ctx,
                 sycl::link(sycl::join<sycl::bundle_state::object>(
                     kernel_bundles_from_input)));
  verify_results(
      log, dev_vector, ctx,
      sycl::link(sycl::join<sycl::bundle_state::object>(mixed_kernel_bundles)));
}

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
    if (!queue.get_device().has(sycl::aspect::online_linker)) {
      log.note("Device does not support online linking of device code");
      return;
    }

    run_verification(log, queue);

    for_all_types<execute_kernel_and_verify_executions>(
        kernels_for_link_and_build, log, queue);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
