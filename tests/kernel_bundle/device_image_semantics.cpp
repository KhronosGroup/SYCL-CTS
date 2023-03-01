/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
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
