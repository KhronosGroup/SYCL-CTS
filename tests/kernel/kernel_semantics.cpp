/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/semantics_reference.h"
#include "../kernel_bundle/kernel_bundle.h"

struct storage {
  sycl::context context;

  explicit storage(const sycl::kernel& kernel)
      : context(kernel.get_context()) {}

  bool check(const sycl::kernel& kernel) const {
    return kernel.get_context() == context;
  }
};

TEST_CASE("kernel common reference semantics", "[kernel]") {
  sycl::context context_0 = sycl_cts::util::get_cts_object::context();
  sycl::context context_1 = sycl_cts::util::get_cts_object::context();
  sycl::queue queue_0 = sycl_cts::util::get_cts_object::queue();
  sycl::queue queue_1 = sycl_cts::util::get_cts_object::queue();

  using k_name = class kernel_comm_ref_sem;
  using k_name_other = class kernel_other_comm_ref_sem;

  sycl::kernel kernel_0 =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(context_0)
          .get_kernel(sycl::get_kernel_id<k_name>());
  sycl::kernel kernel_1 =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(context_1)
          .get_kernel(sycl::get_kernel_id<k_name_other>());
  common_reference_semantics::check_host<storage>(kernel_0, kernel_1, "kernel");
  sycl_cts::tests::kernel_bundle::define_kernel<k_name>(queue_0);
  sycl_cts::tests::kernel_bundle::define_kernel<k_name_other>(queue_1);
}
