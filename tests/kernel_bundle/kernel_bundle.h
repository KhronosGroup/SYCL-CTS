/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for tests on kernel bundle
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_KERNEL_BUNDLE_H
#define __SYCLCTS_TESTS_KERNEL_BUNDLE_H


#include "../common/type_coverage.h"
#include "kernels.h"

namespace sycl_cts {
namespace tests {
namespace kernel_bundle {

inline auto kernels_for_link_and_build = named_type_pack<
    kernels::kernel_cpu_descriptor, kernels::kernel_gpu_descriptor,
    kernels::kernel_accelerator_descriptor, kernels::simple_kernel_descriptor,
    kernels::simple_kernel_descriptor_second,
    kernels::kernel_fp16_no_attr_descriptor,
    kernels::kernel_fp64_no_attr_descriptor,
    kernels::kernel_atomic64_no_attr_descriptor>{
    "kernel_cpu_descriptor",           "kernel_gpu_descriptor",
    "kernel_accelerator_descriptor",   "simple_kernel_descriptor",
    "simple_kernel_descriptor_second", "kernel_fp16_no_attr_descriptor",
    "kernel_fp64_no_attr_descriptor",  "kernel_atomic64_no_attr_descriptor"};

using cpu_kernel = kernels::kernel_cpu_descriptor::type;
using gpu_kernel = kernels::kernel_gpu_descriptor::type;
using accelerator_kernel = kernels::kernel_accelerator_descriptor::type;
using first_simple_kernel = kernels::simple_kernel_descriptor::type;
using second_simple_kernel = kernels::simple_kernel_descriptor_second::type;
using fp16_kernel = kernels::kernel_fp16_no_attr_descriptor::type;
using fp64_kernel = kernels::kernel_fp64_no_attr_descriptor::type;
using atomic64_kernel = kernels::kernel_atomic64_no_attr_descriptor::type;


}  // namespace kernel_bundle
}  // namespace tests
}  // namespace sycl_cts

#endif  // __SYCLCTS_TESTS_KERNEL_BUNDLE_H
