/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tools for sycl::compile tests
//
*******************************************************************************/

#include "../../util/aspect_set.h"
#include "../../util/device_set.h"
#include "../../util/kernel_restrictions.h"
#include "../common/assertions.h"
#include "../common/common.h"
#include "kernel_bundle.h"
#include "kernels.h"

#include <algorithm>

#ifndef __SYCLCTS_TESTS_SYCL_COMPILE_H
#define __SYCLCTS_TESTS_SYCL_COMPILE_H

namespace sycl_cts {
namespace tests {
namespace sycl_compile {

class TestCaseDescription
    : public kernel_bundle::TestCaseDescriptionBase<sycl::bundle_state::input> {
 public:
  TestCaseDescription(std::string_view functionOverload)
      : TestCaseDescriptionBase<sycl::bundle_state::input>("sycl::compile",
                                                           functionOverload){};
};

/** @brief Used to select sycl::compile overload
 */
enum class CompileOverload { bundle_and_devs, bundle_only };

using input_bundle_t = sycl::kernel_bundle<sycl::bundle_state::input>;

/** @brief Call sycl::compile overload selected by Overload tparam
 *  @tparam Overload Selected overload of sycl::compile
 */
template <CompileOverload Overload>
static auto compile_bundle(input_bundle_t &inKb,
                           const std::vector<sycl::device> &devices) {
  if constexpr (Overload == CompileOverload::bundle_and_devs) {
    return sycl::compile(inKb, devices);
  } else if constexpr (Overload == CompileOverload::bundle_only) {
    return sycl::compile(inKb);
  } else {
    static_assert(Overload != Overload, "Wrong Overload passed");
  }
}

/** @brief Check that sycl::compile result bundle contains kernel given
 *  @tparam DescriptorT Selected kernel descriptor
 *  @tparam Overload Selected overload of sycl::compile
 */
template <typename DescriptorT, CompileOverload Overload>
static void check_bundle_kernels(util::logger &log, const std::string &kName) {
  using kernel_t = typename DescriptorT::type;

  auto queue = util::get_cts_object::queue();
  auto ctx = queue.get_context();
  auto dev = queue.get_device();

  auto k_id = sycl::get_kernel_id<kernel_t>();
  // Test can be skipped if all devices do not support online compilation
  // since this is not sycl::compile's fault
  if (!sycl::has_kernel_bundle<sycl::bundle_state::input>(ctx, {dev}, {k_id})) {
    log.skip("No kernel bundle with input state with kernel: " + kName +
             " (skipped).");
    return;
  }

  auto input_kb = sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx, {dev}, {k_id});
  auto input_ids = input_kb.get_kernel_ids();

  auto obj_kb = compile_bundle<Overload>(input_kb, {dev});
  kernel_bundle::define_kernel<DescriptorT, sycl::bundle_state::input>(queue);

  auto restrictions{kernel_bundle::get_restrictions<DescriptorT, sycl::bundle_state::input>()};
  bool dev_is_compat = restrictions.is_compatible(dev);

  if (obj_kb.has_kernel(k_id) != dev_is_compat) {
    FAIL(log, "Device does not support kernel " + kName +
                  " but compiled bundle"
                  " contains it");
  }

  // Check that result object bundle has the same kernels as input bundle
  bool same_kernels_in_obj_bundle = true;
  for (const auto &in_id : input_ids) {
    same_kernels_in_obj_bundle &= obj_kb.has_kernel(in_id);
  }

  if (!same_kernels_in_obj_bundle) {
    FAIL(log,
         "Result bundle does not contain all kernels from input bundle "
         "(kernel: " +
             kName + ")");
  }

  // Check that input and result kernel bundles have the same context
  if (input_kb.get_context() != obj_kb.get_context()) {
    FAIL(log,
         "Input bundle and result bundle have different contexts "
         "(kernel: " +
             kName + ")");
  }
}

/** @brief Check that sycl::compile result bundle has the same associated
 *         devices as presented in input device vector
 *  @tparam Overload Selected overload of sycl::compile
 */
template <CompileOverload Overload>
static void check_associated_devices(util::logger &log) {
  auto queue = util::get_cts_object::queue();
  auto ctx = queue.get_context();
  std::vector<sycl::device> devices{queue.get_device()};
  // Force duplicates
  devices.push_back(devices[0]);

  // Test can be skipped if all devices do not support online compilation
  // since this is not sycl::compile's fault
  if (!sycl::has_kernel_bundle<sycl::bundle_state::input>(ctx)) {
    log.skip("No kernel bundle with input state for test (skipped).");
    return;
  }

  auto input_kb = sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx, devices);
  auto obj_kb = compile_bundle<Overload>(input_kb, devices);

  // Check that result kernel bundle contains all devices from passed vector
  bool same_devs = true;
  auto kb_devs = obj_kb.get_devices();
  for (const auto &dev : devices) {
    same_devs &=
        (std::find(kb_devs.begin(), kb_devs.end(), dev) != kb_devs.end());
  }
  if (!same_devs) {
    FAIL(log,
         "Set of associated to obj_kb devices is not equal to list of "
         "devices passed.");
  }

  // Check that result kernel bundle does not have duplicates
  bool unique_devs = true;
  for (const auto &dev : kb_devs) {
    unique_devs &= (std::count(kb_devs.begin(), kb_devs.end(), dev) == 1);
  }
  if (!unique_devs) {
    FAIL(log, "Set of associated to obj_kb devices has duplicates.");
  }

  using Descr = kernels::simple_kernel_descriptor;
  kernel_bundle::define_kernel<Descr, sycl::bundle_state::input>(queue);
}

}  // namespace sycl_compile
}  // namespace tests
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_SYCL_COMPILE_H
