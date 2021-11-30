/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides helper types and tool for tests on kernel bundle
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_KERNELS_H
#define __SYCLCTS_TESTS_KERNELS_H

#include "../../util/kernel_restrictions.h"
#include "../common/common.h"
#include <stdint.h>

namespace kernels {
using namespace sycl_cts;

struct kernel_base {
  // We are using unsigned long long to make it universal for all cases
  // including atomic64
  using element_type = unsigned long long;
  using accessor_t = sycl::accessor<element_type, 1,
                                    sycl::access_mode::read_write,
                                    sycl::target::global_buffer>;
  static constexpr element_type INIT_VAL = 1;
  static constexpr element_type EXPECTED_VAL = 2;
  static constexpr element_type DIFF_VAL = EXPECTED_VAL - INIT_VAL;
  accessor_t m_acc;

  kernel_base(accessor_t acc) : m_acc(acc) {}

  void trigger_invocation_flag(sycl::item<1> item) const {
    if (item.get_linear_id() == 0) {
      m_acc[0] = EXPECTED_VAL;
    }
  }
};

struct kernel_cpu : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::cpu))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_cpu_descriptor {
  using type = kernel_cpu;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::cpu});
    return restrictions;
  }
};

// that kernel and struct using in multiple kernels in one test
struct kernel_cpu_second : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::cpu))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_cpu_descriptor_second {
  using type = kernel_cpu_second;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::cpu});
    return restrictions;
  }
};

struct kernel_gpu : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::gpu))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_gpu_descriptor {
  using type = kernel_gpu;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::gpu});
    return restrictions;
  }
};

// that kernel and struct using in multiple kernels in one test
struct kernel_gpu_second : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::gpu))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_gpu_descriptor_second {
  using type = kernel_gpu_second;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::gpu});
    return restrictions;
  }
};

struct kernel : kernel_base {
  void operator()(sycl::item<1> id) const { trigger_invocation_flag(id); }
};

struct simple_kernel_descriptor {
  using type = kernel;
  static auto get_restrictions() { return util::kernel_restrictions(); }
};

struct kernel_second : kernel_base {
  void operator()(sycl::item<1> id) const { trigger_invocation_flag(id); }
};

struct simple_kernel_descriptor_second {
  using type = kernel_second;
  static auto get_restrictions() { return util::kernel_restrictions(); }
};

struct kernel_accelerator : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::accelerator))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_accelerator_descriptor {
  using type = kernel_accelerator;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::accelerator});
    return restrictions;
  }
};

// that kernel and struct using in multiple kernels in one test
struct kernel_accelerator_second : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::accelerator))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_accelerator_descriptor_second {
  using type = kernel_accelerator_second;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::accelerator});
    return restrictions;
  }
};

struct kernel_custom : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::custom))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_custom_descriptor {
  using type = kernel_custom;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::custom});
    return restrictions;
  }
};

struct kernel_fp16 : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::fp16))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_fp16_descriptor {
  using type = kernel_fp16;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::fp16});
    return restrictions;
  }
};

struct kernel_fp64 : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::fp64))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_fp64_descriptor {
  using type = kernel_fp64;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::fp64});
    return restrictions;
  }
};

struct kernel_atomic64 : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::atomic64))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_atomic64_descriptor {
  using type = kernel_atomic64;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::atomic64});
    return restrictions;
  }
};

// fp16, fp64, atomic64 kernels without sycl::requires attribute but with
// explicit operations

struct kernel_fp16_no_attr : kernel_base {
  void operator()(sycl::item<1> id) const {
    if (id.get_linear_id() == 0) {
      const auto fp = static_cast<sycl::half>(m_acc[0] + DIFF_VAL);
      m_acc[0] = fp;
    }
  }
};

struct kernel_fp16_no_attr_descriptor {
  using type = kernel_fp16_no_attr;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::fp16});
    return restrictions;
  }
};

struct kernel_fp64_no_attr : kernel_base {
  void operator()(sycl::item<1> id) const {
    if (id.get_linear_id() == 0) {
      const auto fp = static_cast<double>(m_acc[0] + DIFF_VAL);
      m_acc[0] = fp;
    }
  }
};

struct kernel_fp64_no_attr_descriptor {
  using type = kernel_fp64_no_attr;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::fp64});
    return restrictions;
  }
};

struct kernel_atomic64_no_attr : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::atomic64))]] {
    if (id.get_linear_id() == 0) {
      using ref_t = sycl::atomic_ref<unsigned long long,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::work_group,
                                    sycl::access::address_space::global_space>;
      ref_t longAtomic(m_acc[0]);
      longAtomic.fetch_add(DIFF_VAL);
    }
  }
};

struct kernel_atomic64_no_attr_descriptor {
  using type = kernel_atomic64_no_attr;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::atomic64});
    return restrictions;
  }
};

struct kernel_image : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::image))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_image_descriptor {
  using type = kernel_image;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::image});
    return restrictions;
  }
};

struct kernel_online_compiler : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::online_compiler))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_online_compiler_descriptor {
  using type = kernel_online_compiler;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::online_compiler});
    return restrictions;
  }
};

struct kernel_online_linker : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::online_linker))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_online_linker_descriptor {
  using type = kernel_online_linker;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::online_linker});
    return restrictions;
  }
};

struct kernel_queue_profiling : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::queue_profiling))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_queue_profiling_descriptor {
  using type = kernel_queue_profiling;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::queue_profiling});
    return restrictions;
  }
};

struct kernel_usm_device_allocations : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::usm_device_allocations))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_usm_device_allocations_descriptor {
  using type = kernel_usm_device_allocations;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::usm_device_allocations});
    return restrictions;
  }
};

struct kernel_usm_host_allocations : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::usm_host_allocations))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_usm_host_allocations_descriptor {
  using type = kernel_usm_host_allocations;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::usm_host_allocations});
    return restrictions;
  }
};

struct kernel_usm_atomic_host_allocations : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::usm_atomic_host_allocations))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_usm_atomic_host_allocations_descriptor {
  using type = kernel_usm_atomic_host_allocations;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::usm_atomic_host_allocations});
    return restrictions;
  }
};

struct kernel_usm_shared_allocations : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::usm_shared_allocations))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_usm_shared_allocations_descriptor {
  using type = kernel_usm_shared_allocations;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::usm_shared_allocations});
    return restrictions;
  }
};

struct kernel_usm_atomic_shared_allocations : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::usm_atomic_shared_allocations))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_usm_atomic_shared_allocations_descriptor {
  using type = kernel_usm_atomic_shared_allocations;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::usm_atomic_shared_allocations});
    return restrictions;
  }
};

struct kernel_usm_system_allocations : kernel_base {
  void operator()(sycl::item<1> id) const
      [[sycl::requires(has(sycl::aspect::usm_system_allocations))]] {
    trigger_invocation_flag(id);
  }
};

struct kernel_usm_system_allocations_descriptor {
  using type = kernel_usm_system_allocations;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_aspects({sycl::aspect::usm_system_allocations});
    return restrictions;
  }
};

struct kernel_likely_supported_work_group_size : kernel_base {
  [[sycl::reqd_work_group_size(1)]] void operator()(sycl::item<1> id) const {
    trigger_invocation_flag(id);
  }
};

struct kernel_likely_supported_work_group_size_descriptor {
  using type = kernel_likely_supported_work_group_size;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_work_group_size(sycl::id<1>{1});
    return restrictions;
  }
};

struct kernel_likely_unsupported_work_group_size : kernel_base {
  [[sycl::reqd_work_group_size(SIZE_MAX)]] void operator()(
      sycl::item<1> id) const {
    trigger_invocation_flag(id);
  }
};

struct kernel_likely_unsupported_work_group_size_descriptor {
  using type = kernel_likely_unsupported_work_group_size;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_work_group_size(sycl::id<1>{SIZE_MAX});
    return restrictions;
  }
};

struct kernel_likely_unsupported_sub_group_size : kernel_base {
  [[sycl::reqd_sub_group_size(3)]] void operator()(sycl::item<1> id) const {
    trigger_invocation_flag(id);
  }
};

struct kernel_likely_unsupported_sub_group_size_descriptor {
  using type = kernel_likely_unsupported_sub_group_size;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_sub_group_size(3);
    return restrictions;
  }
};

struct kernel_likely_supported_sub_group_size : kernel_base {
  [[sycl::reqd_sub_group_size(32)]] void operator()(sycl::item<1> id) const {
    trigger_invocation_flag(id);
  }
};

struct kernel_likely_supported_sub_group_size_descriptor {
  using type = kernel_likely_supported_sub_group_size;
  static auto get_restrictions() {
    auto restrictions = util::kernel_restrictions();
    restrictions.set_sub_group_size(32);
    return restrictions;
  }
};

}  // namespace kernels

#endif  // __SYCLCTS_TESTS_KERNELS_H