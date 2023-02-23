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
#include "../kernel_bundle/kernel_bundle.h"

struct storage {
  std::uint32_t num_args;

  explicit storage(const sycl::kernel& kernel)
      : num_args(kernel.get_info<sycl::info::kernel::num_args>()) {}

  bool check(const sycl::kernel& kernel) const {
    return kernel.get_info<sycl::info::kernel::num_args>() == num_args;
  }
};

TEST_CASE("kernel common reference semantics", "[kernel]") {
  sycl::context context = sycl_cts::util::get_cts_object::context();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();

  using k_name = class kernel_comm_ref_sem;
  using k_name_other = class kernel_other_comm_ref_sem;

  sycl::kernel kernel_0 =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(context)
          .template get_kernel(sycl::get_kernel_id<k_name>());
  sycl::kernel kernel_1 =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(context)
          .template get_kernel(sycl::get_kernel_id<k_name_other>());
  common_reference_semantics::check_host<storage>(kernel_0, kernel_1, "kernel");
  sycl_cts::tests::kernel_bundle::define_kernel<k_name>(queue);
  sycl_cts::tests::kernel_bundle::define_kernel<k_name_other>(queue);
}
