/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls these sycl::build overloads:
//    1) sycl::build(kernel_bundle<>, std::vector<device>, property_list)
//    2) sycl::build(kernel_bundle<>, property_list)
//  For kernels with CPU, GPU and accelerator requirements, kernels that use
//  fp16 and fp64 data types, a kernel that uses atomic_ref (it uses atomic_ref
//  explicitly but not marked with sycl::requires attribute) and two kernels
//  without any requirement, on device from the chosen platform.
//
//  This test verifies following things:
//    1) That the obtained kernel bundle in executable state contain all
//  provided kernels (only in case when kernel compatible with current device,
//  otherwise this kernel should not be presented in result bundle).
//    2) That kernel bundle context is equal to provided context
//    3) That provided kernels were invoked
//    4) That devices from resulted kernel bundle are equal to provided devices.
//    5) That kernel bundle, that obtained from syc::build is equal (presented
//  the same kernels, context and devices) to kernel bundle from sycl::compile
//  and sycl::link sequentially.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "kernels.h"
#include "sycl_build.h"

#define TEST_NAME sycl_build_verify_kernel_invoked_and_kernel_in_result_bundle

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::kernel_bundle;
using namespace sycl_cts::tests::sycl_build;

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
  for_all_types<verify_that_kernel_in_bundle>(
      kernels_for_link_and_build, log, kernel_bundle, ctx.get_devices().size());
  if (kernel_bundle.get_context() != ctx) {
    FAIL(log, "Kernel bundle's context does not equal to provided context");
  }
  if (kernel_bundle.get_devices() != dev_vector) {
    FAIL(log, "Devices from kernel bundle not equal to provided devices");
  }
}

/** @brief Verify that provided sycl::kernel_bundle presented kernel if it
 *         device compatible with kernel requirements
 *  @tparam KernelDescriptorT Kernel descriptor
 *  @param log sycl_cts::util::logger class object
 *  @param kernel_bundle kernel bundle that was obtained from different
 *         overloads of sycl::get_kernel_bundle function
 *  @param kernel_name String with tested kernel representation
 */
template <typename KernelDescriptorT>
struct verify_that_bundles_are_same {
  void operator()(util::logger &log,
                  const sycl::kernel_bundle<sycl::bundle_state::executable>
                      &kernel_bundle_from_build,
                  const sycl::kernel_bundle<sycl::bundle_state::executable>
                      &kernel_bundle_from_comp_and_link,
                  const std::string &kernel_name) {
    using kernel = typename KernelDescriptorT::type;
    auto k_id{sycl::get_kernel_id<kernel>()};

    if (kernel_bundle_from_build.has_kernel(k_id) !=
        kernel_bundle_from_comp_and_link.has_kernel(k_id)) {
      FAIL(log, "Different results for containing " + kernel_name +
                    " between kernel bundles that obtained using sycl::build "
                    "and using sycl::link(sycl::compile())");
    }
    if (kernel_bundle_from_build.get_context() !=
        kernel_bundle_from_comp_and_link.get_context()) {
      FAIL(log,
           "Different contexts between kernel bundles that obtained using "
           "sycl::build and using sycl::link(sycl::compile())");
    }
    if (kernel_bundle_from_build.get_devices() !=
            kernel_bundle_from_comp_and_link.get_devices() ||
        kernel_bundle_from_build.get_devices().size() !=
            kernel_bundle_from_comp_and_link.get_devices().size()) {
      FAIL(log,
           "Different devices between kernel bundles that obtained using "
           "sycl::build and using sycl::link(sycl::compile())");
    }
  }
};

/** @brief Obtaining sycl::kernel_bundle by two ways: by using sycl::build and
 *         by using sycl::link and sycl::compile sequentially. Do some
 *         verifications for sycl::kernel_bundle, that obtained by sycl::build
 *         (this verifications described in "verify_results" function), then
 *         verify that kernels bundles, that by using sycl::build and by using
 *         sycl::link and sycl::compile sequentially is same (this verifications
 *         described in "verify_that_bundles_are_same" functions)
 *  @param log sycl_cts::util::logger class object
 *  @param q sycL::queue class object
 */
void run_verification(util::logger &log, sycl::queue &q) {
  auto ctx = q.get_context();
  std::vector<sycl::device> dev_vector{ctx.get_devices()[0]};

  const auto cpu_kernel_id = sycl::get_kernel_id<cpu_kernel>();
  const auto gpu_kernel_id = sycl::get_kernel_id<gpu_kernel>();
  const auto accelerator_kernel_id = sycl::get_kernel_id<accelerator_kernel>();
  const auto first_simple_kernel_id =
      sycl::get_kernel_id<first_simple_kernel>();
  const auto second_simple_kernel_id =
      sycl::get_kernel_id<second_simple_kernel>();
  const auto fp16_kernel_id = sycl::get_kernel_id<fp16_kernel>();
  const auto fp64_kernel_id = sycl::get_kernel_id<fp64_kernel>();
  const auto atomic64_kernel_id = sycl::get_kernel_id<atomic64_kernel>();

  auto kb_with_cpu_kernel{
      sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx, {cpu_kernel_id})};
  auto kb_with_gpu_kernel{
      sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx, {gpu_kernel_id})};
  auto kb_with_accelerator_kernel{
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          ctx, {accelerator_kernel_id})};
  auto kb_with_fp16_kernel{sycl::get_kernel_bundle<sycl::bundle_state::input>(
      ctx, {fp16_kernel_id})};
  auto kb_with_fp64_kernel{sycl::get_kernel_bundle<sycl::bundle_state::input>(
      ctx, {fp64_kernel_id})};
  auto kb_with_atomic64_kernel{
      sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx,
                                                         {atomic64_kernel_id})};
  auto kb_with_first_simple_kernel{
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          ctx, {first_simple_kernel_id})};
  auto kb_with_second_simple_kernel{
      sycl::get_kernel_bundle<sycl::bundle_state::input>(
          ctx, {second_simple_kernel_id})};

  std::vector<sycl::kernel_bundle<sycl::bundle_state::input>> kernel_bundles{
      kb_with_cpu_kernel,          kb_with_gpu_kernel,
      kb_with_accelerator_kernel,  kb_with_fp16_kernel,
      kb_with_fp64_kernel,         kb_with_atomic64_kernel,
      kb_with_first_simple_kernel, kb_with_second_simple_kernel};

  const sycl::kernel_bundle<sycl::bundle_state::input> all_kernel_bundles{
      sycl::join(kernel_bundles)};

  {
    log.note("Verify build(kernel_bundle<>, vector<device>) overload");
    const sycl::kernel_bundle<sycl::bundle_state::executable>
        kernel_bundle_from_build{sycl::build(all_kernel_bundles, dev_vector)};
    const sycl::kernel_bundle<sycl::bundle_state::executable>
        kernel_bundle_from_compile_link{
            sycl::link(sycl::compile(all_kernel_bundles), dev_vector)};

    verify_results(log, dev_vector, ctx, kernel_bundle_from_build);
    for_all_types<verify_that_bundles_are_same>(
        kernels_for_link_and_build, log, kernel_bundle_from_build,
        kernel_bundle_from_compile_link);
  }

  {
    log.note("Verify build(kernel_bundle<>) overload");
    const sycl::kernel_bundle<sycl::bundle_state::executable>
        kernel_bundle_from_build{sycl::build(all_kernel_bundles)};
    const sycl::kernel_bundle<sycl::bundle_state::executable>
        kernel_bundle_from_compile_link{
            sycl::link(sycl::compile(all_kernel_bundles))};

    verify_results(log, dev_vector, ctx, kernel_bundle_from_build);
    for_all_types<verify_that_bundles_are_same>(
        kernels_for_link_and_build, log, kernel_bundle_from_build,
        kernel_bundle_from_compile_link);
  }
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
    auto q = util::get_cts_object::queue();
    if (!q.get_device().has(sycl::aspect::online_linker) ||
        !q.get_device().has(sycl::aspect::online_compiler)) {
      log.skip(
          "Device does not support online linking or online compiling of "
          "device code");
      return;
    }

    run_verification(log, q);

    for_all_types<execute_kernel_and_verify_executions>(
        kernels_for_link_and_build, log, q);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
