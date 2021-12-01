/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for sycl::build tests
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SYCL_BUILD_H
#define __SYCLCTS_TESTS_SYCL_BUILD_H

#include "../common/common.h"
#include "kernel_bundle.h"
#include "kernels.h"
#include <vector>

namespace tests_for_sycl_build {

using cpu_kernel = kernels::kernel_cpu_descriptor::type;
using gpu_kernel = kernels::kernel_gpu_descriptor::type;
using accelerator_kernel = kernels::kernel_accelerator_descriptor::type;
using first_simple_kernel = kernels::simple_kernel_descriptor::type;
using second_simple_kernel = kernels::simple_kernel_descriptor_second::type;

template <sycl::bundle_state BundleState>
class TestCaseDescription
    : public sycl_cts::tests::kernel_bundle::TestCaseDescriptionBase<
          BundleState> {
 public:
  constexpr TestCaseDescription(std::string_view functionOverload)
      : sycl_cts::tests::kernel_bundle::TestCaseDescriptionBase<BundleState>(
            "sycl::build", functionOverload) {
    this->m_print_bundle_state = false;
  };
};

}  // namespace tests_for_sycl_build

#endif  // __SYCLCTS_TESTS_SYCL_BUILD_H
