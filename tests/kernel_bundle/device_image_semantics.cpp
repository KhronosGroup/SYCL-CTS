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
#include "kernel_bundle.h"

struct dummy_kernel;

struct storage {
  bool has_kernel;

  template <sycl::bundle_state BundleState>
  explicit storage(const sycl::device_image<BundleState>& device_image)
      : has_kernel(
            device_image.has_kernel(sycl::get_kernel_id<dummy_kernel>())) {}

  template <sycl::bundle_state BundleState>
  bool check(const sycl::device_image<BundleState>& device_image) const {
    return device_image.has_kernel(sycl::get_kernel_id<dummy_kernel>()) ==
           has_kernel;
  }
};

TEST_CASE("device_image common reference semantics", "[device_image]") {
  sycl::context context = sycl_cts::util::get_cts_object::context();
  sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
      sycl::get_kernel_bundle<dummy_kernel, sycl::bundle_state::executable>(
          context, {device});
  // only accessible as const reference
  const sycl::device_image<sycl::bundle_state::executable>& device_image =
      kernel_bundle.begin()[0];
  common_reference_semantics::check_host<storage>(
      device_image, "device_image<bundle_state::executable>");

  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  sycl_cts::tests::kernel_bundle::define_kernel<dummy_kernel>(queue);
}
