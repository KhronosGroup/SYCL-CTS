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
//  Provides tests for kernel_bundle api
//
*******************************************************************************/
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_test_macros.hpp"

#include "kernel_bundle.h"

TEST_CASE(
    "Check that is_default_constructible_v<sycl::kernel_bundle> is false"
    "kernels",
    "[kernel_bundle]") {
  CHECK(!std::is_default_constructible_v<
        sycl::kernel_bundle<sycl::bundle_state::input>>);
  CHECK(!std::is_default_constructible_v<
        sycl::kernel_bundle<sycl::bundle_state::object>>);
  CHECK(!std::is_default_constructible_v<
        sycl::kernel_bundle<sycl::bundle_state::executable>>);
}

TEST_CASE(
    "Check that kernel_bundle::get_backend() return type is sycl::backend"
    "kernels",
    "[kernel_bundle]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      queue.get_context());
  auto backend = kernel_bundle.get_backend();
  CHECK(std::is_same_v<decltype(backend), sycl::backend>);
}

TEST_CASE(
    "Check that kernel_bundle::get_devices() return type is "
    "std::vector<sycl::device> and vector contains selected device"
    "kernels",
    "[kernel_bundle]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto device = queue.get_device();
  std::vector<sycl::device> dev_vector{device};
  auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      queue.get_context(), dev_vector);
  auto kernel_bundle_dev_vector = kernel_bundle.get_devices();
  CHECK(std::is_same_v<decltype(kernel_bundle_dev_vector),
                       std::vector<sycl::device>>);
  REQUIRE(1 == kernel_bundle_dev_vector.size());
  CHECK(device == kernel_bundle_dev_vector[0]);
}

TEST_CASE(
    "Check kernel_bundle::has_kernel(const kernel_id&, const device&)"
    "kernels",
    "[kernel_bundle]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      queue.get_context());
  auto kernel_ids = kernel_bundle.get_kernel_ids();
  auto device = queue.get_device();
  for (auto& kernel_id : kernel_ids) {
    auto comp_res = sycl::is_compatible({kernel_id}, device);
    CHECK(comp_res == kernel_bundle.has_kernel(kernel_id, device));
  }
}

TEST_CASE(
    "Check kernel_bundle::has_kernel<typename KernelName>()"
    "kernels",
    "[kernel_bundle]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto device = queue.get_device();
  using kernel_name = class simple_kernel_has_kernel;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name>(queue);
  auto kernel_bundle =
      sycl::get_kernel_bundle<kernel_name, sycl::bundle_state::executable>(
          queue.get_context());
  CHECK(kernel_bundle.has_kernel<kernel_name>());
}

TEST_CASE(
    "Check kernel_bundle::has_kernel<typename KernelName>(const device&)"
    "kernels",
    "[kernel_bundle]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto device = queue.get_device();
  using kernel_name = class simple_kernel_has_kernel_device;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name>(queue);
  auto kernel_bundle =
      sycl::get_kernel_bundle<kernel_name, sycl::bundle_state::executable>(
          queue.get_context());
  CHECK(kernel_bundle.has_kernel<kernel_name>(device));
}

TEST_CASE(
    "Check kernel_bundle::get_kernel_ids()"
    "kernels",
    "[kernel_bundle]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  using kernel_name_1 = class simple_kernel1;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name_1>(queue);
  using kernel_name_2 = class simple_kernel2;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name_2>(queue);
  using kernel_name_3 = class simple_kernel3;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name_3>(queue);
  using kernel_name_4 = class simple_kernel4;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name_4>(queue);
  auto kernel_bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      queue.get_context());
  auto kernel_ids = kernel_bundle.get_kernel_ids();
  CHECK(std::is_same_v<decltype(kernel_ids), std::vector<sycl::kernel_id>>);
  CHECK(kernel_ids.size() >= 4);
  for (auto it = kernel_ids.begin(), nextit = kernel_ids.begin();
       it != kernel_ids.end(); ++it) {
    auto& kernel_id = *it;
    ++nextit;
    CHECK(std::find(nextit, kernel_ids.end(), kernel_id) == kernel_ids.end());
  }
}

// FIXME: re-enable when sycl::kernel_bundle::get_kernel<KernelName>() is
// implemented
DISABLED_FOR_TEST_CASE(DPCPP)
("Check kernel_bundle::get_kernel<KernelName>()"
 "kernels",
 "[kernel_bundle]")({
  auto queue = sycl_cts::util::get_cts_object::queue();
  using kernel_name = class simple_kernel_get_kernel;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name>(queue);
  auto kernel_bundle =
      sycl::get_kernel_bundle<kernel_name, sycl::bundle_state::executable>(
          queue.get_context());
  auto kernel = kernel_bundle.get_kernel<kernel_name>();
  CHECK(std::is_same_v<decltype(kernel), sycl::kernel>);
});
