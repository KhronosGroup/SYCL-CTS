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
#include "get_kernel_id.h"
#include "kernel_bundle.h"

// kernel_id no members
struct storage {
  std::string name;

  explicit storage(const sycl::kernel_id& kernel_id)
      : name(kernel_id.get_name()) {}

  bool check(const sycl::kernel_id& kernel_id) const {
    return kernel_id.get_name() == name;
  }
};

TEST_CASE("kernel_id common reference semantics", "[kernel_id]") {
  sycl::context context = sycl_cts::util::get_cts_object::context();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();

  using k_name = class kernel_id_comm_ref_sem;
  using k_name_other = class kernel_id_other_comm_ref_sem;

  sycl::kernel_id kernel_id_0 = sycl::get_kernel_id<k_name>();
  sycl::kernel_id kernel_id_1 = sycl::get_kernel_id<k_name_other>();
  common_reference_semantics::check_host<storage>(kernel_id_0, kernel_id_1,
                                                  "kernel_id");
  sycl_cts::tests::kernel_bundle::define_kernel<k_name>(queue);
  sycl_cts::tests::kernel_bundle::define_kernel<k_name_other>(queue);
}

TEST_CASE("kernel_id special reference semantics", "[kernel_id]") {
  // Check that two sycl::kernel_ids referring to the same kernel name are equal
  sycl_cts::util::get_cts_object::queue().submit([&](sycl::handler& cgh) {
    using k_name = class two_kernel_ids_same_kernel;
    sycl::kernel_id k_id = sycl::get_kernel_id<k_name>();
    sycl::kernel_id k_id_same = sycl::get_kernel_id<k_name>();
    CHECK(k_id == k_id_same);
    // Dummy kernel
    cgh.single_task<k_name>([] {});
  });
}
