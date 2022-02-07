/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides tests for sycl::compile to check that all fp16, fp64, atomic64,
//  cpu, gpu, accelerator kernels are represented in result bundle if device
//  has aspect given. Overload (bundle, devices, pl)
//
*******************************************************************************/

#include "../common/common.h"
#include "sycl_compile.h"
#include "kernels.h"

#define TEST_NAME sycl_compile_special_kernels_check_bundle_and_devs

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace sycl_cts::tests::sycl_compile;

using descr_fp16 = kernels::kernel_fp16_no_attr_descriptor;
using descr_fp64 = kernels::kernel_fp64_no_attr_descriptor;
using descr_atomic64 = kernels::kernel_atomic64_no_attr_descriptor;
using descr_cpu = kernels::kernel_cpu_descriptor;
using descr_gpu = kernels::kernel_gpu_descriptor;
using descr_acc = kernels::kernel_accelerator_descriptor;

/** test sycl::compile to check all fp16, fp64, atomic64, cpu, gpu, accelerator
 *  kernels existence in result bundle
 */
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
    constexpr auto overload = CompileOverload::bundle_and_devs;
    check_bundle_kernels<descr_fp16, overload>(log, "kernel_fp16");
    check_bundle_kernels<descr_fp64, overload>(log, "kernel_fp64");
    check_bundle_kernels<descr_atomic64, overload>(log, "kernel_atomic64");
    check_bundle_kernels<descr_cpu, overload>(log, "kernel_cpu");
    check_bundle_kernels<descr_gpu, overload>(log, "kernel_gpu");
    check_bundle_kernels<descr_acc, overload>(log, "kernel_acc");
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
