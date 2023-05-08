/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This test calls these sycl::get_kernel_bundle overloads:
//    1) sycl::get_kernel_bundle<BundleState>(context, kernelIds)
//    2) sycl::get_kernel_bundle<BundleState>(context, devices, kernelIds)
//    3) get_kernel_bundle<KernelName, BundleState>(const context& ctxt)
//    4) get_kernel_bundle<KernelName, BundleState>(const context& ctxt,
                                       const std::vector<device>& devs)
//  For kernels that are incompatible with current device.
//
//  The test verifies that the exceptions with sycl::errc::invalid are thrown.
//
*******************************************************************************/

#include "../../util/sycl_exceptions.h"
#include "../common/assertions.h"
#include "../common/common.h"
#include "../common/get_cts_string.h"
#include "get_kernel_bundle.h"
#include "kernel_bundle.h"
#include "kernels.h"
#include <catch2/matchers/catch_matchers.hpp>

#define TEST_NAME get_kernel_bundle_with_incompatible_kernels

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace kernels;
using namespace sycl_cts::tests::kernel_bundle;
using namespace sycl_cts::tests::kernel_bundle::get_kernel_bundle;

using kernel = simple_kernel_descriptor::type;

/** @brief Calls get_kernel_bundle overloads when provided vector with
 *         kernel_ids contains kernels that incompatible with current device
 *  @tparam BundleState Bundle state
 *  @param log sycl_cts::util::logger class object
 *  @param context Context that will be used in sycl::get_kernel_bundle
 *  @param devices std::vector sycl::device with that will be used in
 *         sycl::get_kernel_bundle
 */
template <sycl::bundle_state BundleState>
void run_test_for_overload_types(
    util::logger &log, const sycl::context &context,
    const std::vector<sycl::device> &devices,
    const std::vector<sycl::kernel_id> &incompatible_kernel_ids) {
  std::string state_string =
      sycl_cts::get_cts_string::for_bundle_state<BundleState>();
  {
    INFO("Check overload (context, devices, kernel_ids) for BundleState " +
         state_string);
    CHECK_THROWS_MATCHES(sycl::get_kernel_bundle<BundleState>(
                             context, devices, incompatible_kernel_ids),
                         sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  {
    INFO("Check overload (context, kernel_ids) for BundleState " +
         state_string);
    CHECK_THROWS_MATCHES(
        sycl::get_kernel_bundle<BundleState>(context, incompatible_kernel_ids),
        sycl::exception, sycl_cts::util::equals_exception(sycl::errc::invalid));
  }
}

template <typename KernelName, sycl::bundle_state BundleState>
void run_test_overload_types_with_kernel_names(
    const sycl::context &context, const std::vector<sycl::device> &devices) {
  std::string state_string =
      sycl_cts::get_cts_string::for_bundle_state<BundleState>();
  {
    INFO("Check overload <KernelName>(context, devices) for BundleState " +
         state_string);
    auto action = [&] {
      sycl::get_kernel_bundle<KernelName, BundleState>(context, devices);
    };
    CHECK_THROWS_MATCHES(action(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  {
    INFO("Check overload <KernelName>(context) for BundleState " +
         state_string);
    auto action = [&] {
      sycl::get_kernel_bundle<KernelName, BundleState>(context);
    };
    CHECK_THROWS_MATCHES(action(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }
}

template <typename KernelDescriptorT>
struct run_with_kernel_names {
  void operator()(const sycl::context &context, const sycl::device &device,
                  const std::string &kernel_name) {
    using kernelName = typename KernelDescriptorT::type;
    INFO("check for " + kernel_name);
    if (!sycl::is_compatible<kernelName>(device)) {
      if (!device.has(sycl::aspect::online_linker)) {
        WARN(
            "Tests for input bundle state skipped due to device does "
            "not support online_linker.");
      } else {
        run_test_overload_types_with_kernel_names<kernelName,
                                                  sycl::bundle_state::input>(
            context, {device});
      }

      if (!device.has(sycl::aspect::online_compiler)) {
        WARN(
            "Tests for object bundle states skipped due to device does "
            "not support online_compiler aspect.");
      } else {
        run_test_overload_types_with_kernel_names<kernelName,
                                                  sycl::bundle_state::object>(
            context, {device});
      }

      run_test_overload_types_with_kernel_names<kernelName,
                                                sycl::bundle_state::executable>(
          context, {device});
    }
  }
};

template <typename KernelDescriptorT>
struct fill_vector_with_all_user_defined_kernel_ids {
  void operator()(std::vector<sycl::kernel_id> &user_defined_kernel_ids,
                  const std::string &kernel_name) {
    using kernel = typename KernelDescriptorT::type;
    auto k_id{sycl::get_kernel_id<kernel>()};

    user_defined_kernel_ids.push_back(k_id);
  }
};

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
    auto queue{util::get_cts_object::queue()};
    sycl::context context{queue.get_context()};
    sycl::device device{queue.get_device()};

    std::vector<sycl::kernel_id> incompatible_kernel_ids{};

    // fill vector with kernel ids with for pre-defined kernels
    for_all_types<fill_vector_with_all_user_defined_kernel_ids>(
        kernels_with_attributes, incompatible_kernel_ids);

    if (!device.has(sycl::aspect::online_linker)) {
      WARN(
          "Tests for input bundle state skipped due to device does "
          "not support online_linker.");
    } else {
      run_test_for_overload_types<sycl::bundle_state::input>(
          log, context, {device}, incompatible_kernel_ids);
    }

    if (!device.has(sycl::aspect::online_compiler)) {
      WARN(
          "Tests for object bundle states skipped due to device does "
          "not support online_compiler aspect.");
    } else {
      run_test_for_overload_types<sycl::bundle_state::object>(
          log, context, {device}, incompatible_kernel_ids);
    }

    run_test_for_overload_types<sycl::bundle_state::executable>(
        log, context, {device}, incompatible_kernel_ids);

    for_all_types<run_with_kernel_names>(kernels_with_attributes, context,
                                         device);

    for_all_types<execute_kernel_and_verify_executions>(kernels_with_attributes,
                                                        log, queue);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
