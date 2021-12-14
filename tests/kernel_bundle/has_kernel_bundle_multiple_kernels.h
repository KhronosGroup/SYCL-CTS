/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides verification logic for case of multiple kernel ids call
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_MULTIPLE_KERNELS_H
#define __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_MULTIPLE_KERNELS_H

#include "../common/common.h"
#include "has_kernel_bundle.h"
#include "kernels.h"
#include <string>
#include <vector>

namespace sycl_cts::testshas_kernel_bundle::check {
namespace details {
/** @brief Check that calling has_kernel_bundle overload for a two kernels
 *         is same as:
 *           - call has_kernel_bundle overload for each of kernels and
 *           - apply 'AND' operation over the results of the separate calls
 *  @details According to the SYCL-2020 rev.3 #4.11.8 "Querying if a kernel
 *           bundle exists" we need to ensure that
 *           - if we call specific sycl::has_kernel_bundle overload for multiple
 *             kernel ids
 *           - the result "true" is expected only if every kernel that used
 *             is compatible with device
 */
template <sycl::bundle_state BundleState, typename SyclOverloadT>
void verify_specific_kernels(sycl_cts::util::logger &log, sycl::context ctx,
                             const sycl::device &device,
                             SyclOverloadT call_has_kernel_bundle) {
  using cpu_desc = typename kernels::kernel_cpu_descriptor;
  using gpu_desc = typename kernels::kernel_gpu_descriptor;
  using acc_desc = typename kernels::kernel_accelerator_descriptor;
  using cpu_diff_desc = typename kernels::kernel_cpu_descriptor_second;
  using gpu_diff_desc = typename kernels::kernel_gpu_descriptor_second;
  using acc_diff_desc = typename kernels::kernel_accelerator_descriptor_second;

  // Try to submit kernels we want to use to make sycl::get_kernel_id usage
  // valid
  {
    sycl::queue queue(ctx, device);
    kernel_bundle::define_kernel<cpu_desc, BundleState>(queue);
    kernel_bundle::define_kernel<gpu_desc, BundleState>(queue);
    kernel_bundle::define_kernel<acc_desc, BundleState>(queue);
    kernel_bundle::define_kernel<cpu_diff_desc, BundleState>(queue);
    kernel_bundle::define_kernel<gpu_diff_desc, BundleState>(queue);
    kernel_bundle::define_kernel<acc_diff_desc, BundleState>(queue);
  }

  // Retrieve kernel ids for test
  const auto cpu = sycl::get_kernel_id<cpu_desc::type>();
  const auto gpu = sycl::get_kernel_id<gpu_desc::type>();
  const auto acc = sycl::get_kernel_id<acc_desc::type>();
  const auto cpu_diff = sycl::get_kernel_id<cpu_diff_desc::type>();
  const auto gpu_diff = sycl::get_kernel_id<gpu_diff_desc::type>();
  const auto acc_diff = sycl::get_kernel_id<acc_diff_desc::type>();

  // We expect device to be either cpu, gpu or accelerator for this test
  const bool only_cpu = call_has_kernel_bundle({cpu});
  const bool only_gpu = call_has_kernel_bundle({gpu});
  const bool only_acc = call_has_kernel_bundle({acc});
  const bool only_cpu_diff = call_has_kernel_bundle({cpu_diff});
  const bool only_gpu_diff = call_has_kernel_bundle({gpu_diff});
  const bool only_acc_diff = call_has_kernel_bundle({acc_diff});

  // Provide verbose logs
  log.debug([&] {
    const auto device_name = device.get_info<sycl::info::device::name>();
    const auto platform = device.get_platform();
    const auto platform_name = platform.get_info<sycl::info::platform::name>();

    std::string message("Running for");
    message += " platform '" + platform_name;
    message += "', device '" + device_name;
    message += "'\n  is CPU: " + std::to_string(device.is_cpu());
    message += "\n  is GPU: " + std::to_string(device.is_gpu());
    message += "\n  is ACC: " + std::to_string(device.is_accelerator());
    message +=
        "\n  sycl::has_kernel_bundle(CPU kernel): " + std::to_string(only_cpu);
    message +=
        "\n  sycl::has_kernel_bundle(GPU kernel): " + std::to_string(only_gpu);
    message +=
        "\n  sycl::has_kernel_bundle(ACC kernel): " + std::to_string(only_acc);
    return message;
  });

  auto on_failure = [&](bool retrieved, bool expected,
                        const std::string description) {
    const auto device_name = device.get_info<sycl::info::device::name>();
    std::string message;

    message += "Check for " + description + " failed. Retrieved: ";
    message += std::to_string(retrieved) + ", expected: ";
    message += std::to_string(expected) +
               ".\nResults for single "
               "kernels:";
    message += "CPU " + std::to_string(only_cpu);
    message += ", GPU " + std::to_string(only_gpu);
    message += ", accelerator: " + std::to_string(only_acc);
    message += ".\nDevice name: " + device_name;
    message += ".\nDevice is CPU: " + std::to_string(device.is_cpu());
    message += ", is GPU: " + std::to_string(device.is_gpu());
    message += ", is accelerator: " + std::to_string(device.is_accelerator());
    message += "\n";
    FAIL(log, message);
  };

  // We need to check that call({a, b}) = call(a) && call(b)
  // So we need pairs of [0,0], [0,1] and [1,1]
  {
    const bool expected = only_cpu && only_gpu;
    const bool result = call_has_kernel_bundle({cpu, gpu});

    if (result != expected) {
      on_failure(result, expected, "CPU and GPU aspects");
    }
  }
  {
    const bool expected = only_cpu && only_acc;
    const bool result = call_has_kernel_bundle({cpu, acc});

    if (result != expected) {
      on_failure(result, expected, "CPU and accessor aspects");
    }
  }
  {
    const bool expected = only_cpu && only_cpu_diff;
    const bool result = call_has_kernel_bundle({cpu, cpu_diff});

    if (result != expected) {
      on_failure(result, expected, "two kernels with CPU aspects");
    }
  }
  {
    const bool expected = only_gpu && only_gpu_diff;
    const bool result = call_has_kernel_bundle({gpu, gpu_diff});

    if (result != expected) {
      on_failure(result, expected, "two kernels with GPU aspects");
    }
  }
  {
    const bool expected = only_acc && only_acc_diff;
    const bool result = call_has_kernel_bundle({acc, acc_diff});

    if (result != expected) {
      on_failure(result, expected, "two kernels with CPU aspects");
    }
  }
}

/** @brief Check that calling has_kernel_bundle overload for a two kernels
 *         is same as:
 *           - call has_kernel_bundle overload for each of kernels and
 *           - apply 'OR' operation over the results of the separate calls
 *  @details According to the SYCL-2020 rev.3 #4.11.8 "Querying if a kernel
 *           bundle exists" we need to ensure that
 *           - if we call specific sycl::has_kernel_bundle overload for multiple
 *             kernel ids
 *           - the result "true" is expected only if any kernel within
 *             application is compatible with device and is representable in the
 *             requested state.
 */
template <sycl::bundle_state BundleState, typename SyclOverloadT>
void verify_all_kernels_in_application(sycl_cts::util::logger &log,
                                       sycl::context ctx,
                                       const sycl::device &device,
                                       SyclOverloadT call_has_kernel_bundle) {
  const std::vector<sycl::device> devices{device};

  using cpu_desc = typename kernels::kernel_cpu_descriptor;
  using gpu_desc = typename kernels::kernel_gpu_descriptor;
  using acc_desc = typename kernels::kernel_accelerator_descriptor;

  // Submit these three kernels to make sure that the application contains at
  // least some kernels. However, the test below operates on all the kernels in
  // the application, which might include more than these three.
  {
    sycl::queue queue(ctx, device);
    kernel_bundle::define_kernel<cpu_desc, BundleState>(queue);
    kernel_bundle::define_kernel<gpu_desc, BundleState>(queue);
    kernel_bundle::define_kernel<acc_desc, BundleState>(queue);
  }

  // Retrieve kernel ids for test
  auto ids = sycl::get_kernel_ids();

  // Provide verbose logs
  log.debug([&] {
    const auto device_name = device.get_info<sycl::info::device::name>();
    const auto platform = device.get_platform();
    const auto platform_name = platform.get_info<sycl::info::platform::name>();

    std::string message("Running for");
    message += " platform '" + platform_name;
    message += "', device '" + device_name;
    message += "'\n  is CPU: " + std::to_string(device.is_cpu());
    message += "\n  is GPU: " + std::to_string(device.is_gpu());
    message += "\n  is ACC: " + std::to_string(device.is_accelerator());
    return message;
  });

  // Assert on test pre-condition
  {
    const auto ctx_devices = ctx.get_devices();
    if (ctx_devices != devices) {
      FAIL(log, "Pre-condition failed, test issue");
    }
  }

  // Retrieve results of a single calls
  bool expected = false;
  for (const auto &id : ids) {
    expected |= sycl::has_kernel_bundle<BundleState>(ctx, devices, {id});
  }

  // Run the test
  const bool retrieved = call_has_kernel_bundle();
  if (retrieved != expected) {
    const auto device_name = device.get_info<sycl::info::device::name>();
    std::string message;

    message += "Check failed. Retrieved: ";
    message += std::to_string(retrieved) + ", expected: ";
    message += std::to_string(expected);
    message += ".\nDevice name: " + device_name;
    message += ".\nDevice is CPU: " + std::to_string(device.is_cpu());
    message += ", is GPU: " + std::to_string(device.is_gpu());
    message += ", is accelerator: " + std::to_string(device.is_accelerator());
    message += "\n";
    FAIL(log, message);
  }
}

}  // namespace details

/** @brief Specialization for case of has_kernel_bundle(context, ids)
 *         overload
 */
template <sycl::bundle_state BundleState>
struct multiple_kernels<BundleState, overload::id::ctx_kid> {
  static void run(sycl_cts::util::logger &log, const sycl::device &device) {
    const std::vector<sycl::device> devices{device};
    sycl::context ctx(devices);

    auto call = [&](const std::vector<sycl::kernel_id> &ids) {
      return sycl::has_kernel_bundle<BundleState>(ctx, ids);
    };
    details::verify_specific_kernels<BundleState>(log, ctx, device, call);
  }
};

/** @brief Specialization for case of has_kernel_bundle(context, devices, ids)
 *         overload
 */
template <sycl::bundle_state BundleState>
struct multiple_kernels<BundleState, overload::id::ctx_dev_kid> {
  static void run(sycl_cts::util::logger &log, const sycl::device &device) {
    const std::vector<sycl::device> devices{device};
    sycl::context ctx(devices);

    auto call = [&](const std::vector<sycl::kernel_id> &ids) {
      return sycl::has_kernel_bundle<BundleState>(ctx, devices, ids);
    };
    details::verify_specific_kernels<BundleState>(log, ctx, device, call);
  }
};

/** @brief Specialization for case of has_kernel_bundle(context, devices)
 *         overload
 */
template <sycl::bundle_state BundleState>
struct multiple_kernels<BundleState, overload::id::ctx_dev> {
  static void run(sycl_cts::util::logger &log, const sycl::device &device) {
    const std::vector<sycl::device> devices{device};
    sycl::context ctx(devices);

    auto call = [&]() {
      return sycl::has_kernel_bundle<BundleState>(ctx, devices);
    };
    details::verify_all_kernels_in_application<BundleState>(log, ctx, device,
                                                            call);
  }
};

/** @brief Specialization for case of has_kernel_bundle(context) overload
 */
template <sycl::bundle_state BundleState>
struct multiple_kernels<BundleState, overload::id::ctx_only> {
  static void run(sycl_cts::util::logger &log, const sycl::device &device) {
    const std::vector<sycl::device> devices{device};
    sycl::context ctx(devices);

    auto call = [&]() { return sycl::has_kernel_bundle<BundleState>(ctx); };
    details::verify_all_kernels_in_application<BundleState>(log, ctx, device,
                                                            call);
  }
};

}  // namespace sycl_cts::testshas_kernel_bundle::check

#endif  // __SYCLCTS_TESTS_HAS_KERNEL_BUNDLE_H
