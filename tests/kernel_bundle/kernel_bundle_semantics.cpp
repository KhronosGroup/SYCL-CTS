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
  bool is_empty;
  std::size_t device_count;
  std::size_t kernel_id_count;

  template <sycl::bundle_state BundleState>
  explicit storage(const sycl::kernel_bundle<BundleState>& kernel_bundle)
      : is_empty(kernel_bundle.empty()),
        device_count(kernel_bundle.get_devices().size()),
        kernel_id_count(kernel_bundle.get_kernel_ids().size()) {}

  template <sycl::bundle_state BundleState>
  bool check(const sycl::kernel_bundle<BundleState>& kernel_bundle) const {
    return kernel_bundle.empty() == is_empty &&
           kernel_bundle.get_devices().size() == device_count &&
           kernel_bundle.get_kernel_ids().size() == kernel_id_count;
  }
};

TEST_CASE("kernel_bundle common reference semantics", "[kernel_bundle]") {
  sycl::context context = sycl_cts::util::get_cts_object::context();
  sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::kernel_bundle<sycl::bundle_state::executable> kernel_bundle =
      sycl::get_kernel_bundle<dummy_kernel, sycl::bundle_state::executable>(
          context, {device});
  common_reference_semantics::check_host<storage>(kernel_bundle,
                                                  "kernel_bundle");

  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  sycl_cts::tests::kernel_bundle::define_kernel<dummy_kernel>(queue);
}
