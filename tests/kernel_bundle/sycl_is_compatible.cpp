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
//  Provides tests for is_compatible()
//
*******************************************************************************/

#include "kernel_bundle.h"
#include "kernels.h"

TEST_CASE(
    "Check is_compatible(const std::vector<kernel_id>, const device) for zero "
    "kernels",
    "[kernel_bundle]") {
  const sycl::device device = sycl_cts::util::get_cts_object::device();
  const std::vector<sycl::kernel_id> kernelIds;

  CHECK(sycl::is_compatible(kernelIds, device));
}

TEST_CASE(
    "Check is_compatible(const std::vector<kernel_id>, const device) for "
    "built-in kernel ids for the selected device",
    "[kernel_bundle]") {
  const sycl::device device = sycl_cts::util::get_cts_object::device();
  const std::vector<sycl::kernel_id> builtinKernelIds =
      device.get_info<sycl::info::device::built_in_kernel_ids>();

  if (builtinKernelIds.empty())
    SKIP("No built-in kernels available for selected device.");

  CHECK(sycl::is_compatible(builtinKernelIds, device));
}

TEST_CASE(
    "Check is_compatible(const std::vector<kernel_id>, const device) for "
    "built-in kernel ids from every root device other than the selected device",
    "[kernel_bundle]") {
  const sycl::device device = sycl_cts::util::get_cts_object::device();
  auto platforms = sycl::platform::get_platforms();
  std::vector<sycl::kernel_id> builtinKernelIds;

  for (auto p : platforms) {
    auto devices = p.get_devices();
    for (auto d : devices) {
      if (d != device) {
        std::vector<sycl::kernel_id> newKernelIds =
            d.get_info<sycl::info::device::built_in_kernel_ids>();
        builtinKernelIds.insert(builtinKernelIds.end(), newKernelIds.begin(),
                                newKernelIds.end());
      }
    }
  }
  if (builtinKernelIds.empty())
    SKIP("No built-in kernels available for non-tested devices.");
  CHECK(!sycl::is_compatible(builtinKernelIds, device));
}

template <typename KernelDescriptorT>
void check_with_optional_features(sycl::device device, sycl::queue queue,
                                  bool result) {
  using kernel_name = typename KernelDescriptorT::type;
  sycl_cts::tests::kernel_bundle::define_kernel<KernelDescriptorT,
                                                sycl::bundle_state::executable>(
      queue);

  CHECK(sycl::is_compatible<kernel_name>(device) == result);

  std::vector<sycl::kernel_id> builtinKernelIds(
      1, sycl::get_kernel_id<kernel_name>());
  CHECK(sycl::is_compatible(builtinKernelIds, device) == result);
}

TEST_CASE("Check is_compatible for kernels with no kernel attributes",
          "[kernel_bundle]") {
  const sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();

  SECTION("for kernel with no optional features") {
    using kernel_name = class simple_kernel;
    sycl_cts::tests::kernel_bundle::define_kernel<kernel_name>(queue);

    CHECK(sycl::is_compatible<kernel_name>(device));

    std::vector<sycl::kernel_id> builtinKernelIds(
        1, sycl::get_kernel_id<kernel_name>());
    CHECK(sycl::is_compatible(builtinKernelIds, device));
  }

  SECTION("for a kernel that uses `sycl::half`") {
    check_with_optional_features<kernels::kernel_fp16_no_attr_descriptor>(
        device, queue, device.has(sycl::aspect::fp16));
  }

  SECTION("for a kernel that uses `double`") {
    check_with_optional_features<kernels::kernel_fp64_no_attr_descriptor>(
        device, queue, device.has(sycl::aspect::fp64));
  }

  SECTION("for a kernel that uses 64-bit atomic operations") {
    check_with_optional_features<kernels::kernel_atomic64_no_attr_descriptor>(
        device, queue, device.has(sycl::aspect::atomic64));
  }
}

template <size_t SIZE>
void check_max_work_group_size(sycl::device device, sycl::queue queue,
                               size_t max_work_group_size) {
  using kernel_name = kernels::kernel_work_group_size_descriptor<SIZE>;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name,
                                                sycl::bundle_state::executable>(
      queue);
  bool expected_result = max_work_group_size > SIZE;

  CHECK(sycl::is_compatible<typename kernel_name::type>(device) ==
        expected_result);

  std::vector<sycl::kernel_id> builtinKernelIds(
      1, sycl::get_kernel_id<typename kernel_name::type>());
  CHECK(sycl::is_compatible(builtinKernelIds, device) == expected_result);
}

TEST_CASE(
    "Check is_compatible for kernels with `[[sycl::reqd_work_group_size()]]` "
    "attributes",
    "[kernel_bundle]") {
  const sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  size_t max_work_group_size =
      device.get_info<sycl::info::device::max_work_item_sizes<1>>();

  SECTION("for a kernel with `[[sycl::reqd_work_group_size(8)]]`") {
    check_max_work_group_size<8>(device, queue, max_work_group_size);
  }

  SECTION("for a kernel with `[[sycl::reqd_work_group_size(16)]]`") {
    check_max_work_group_size<16>(device, queue, max_work_group_size);
  }

  SECTION("for a kernel with `[[sycl::reqd_work_group_size(4294967295)]]`") {
    check_max_work_group_size<4294967295>(device, queue, max_work_group_size);
  }
}

template <size_t SIZE>
void check_sub_group_size(sycl::device device, sycl::queue queue,
                          std::vector<size_t> sub_group_sizes) {
  using kernel_name = kernels::kernel_sub_group_size_descriptor<SIZE>;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name,
                                                sycl::bundle_state::executable>(
      queue);
  bool expected_result =
      std::any_of(sub_group_sizes.cbegin(), sub_group_sizes.cend(),
                  [](size_t i) { return i == SIZE; });

  CHECK(sycl::is_compatible<typename kernel_name::type>(device) ==
        expected_result);

  std::vector<sycl::kernel_id> builtinKernelIds(
      1, sycl::get_kernel_id<typename kernel_name::type>());
  CHECK(sycl::is_compatible(builtinKernelIds, device) == expected_result);
}

TEST_CASE(
    "Check is_compatible for kernels with `[[sycl::reqd_sub_group_size()]]` "
    "attributes",
    "[kernel_bundle]") {
  const sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  auto sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>();

  SECTION("for a kernel with `[[sycl::reqd_sub_group_size(8)]]`") {
    check_sub_group_size<8>(device, queue, sub_group_sizes);
  }

  SECTION("for a kernel with `[[sycl::reqd_sub_group_size(16)]]`") {
    check_sub_group_size<16>(device, queue, sub_group_sizes);
  }

  SECTION("for a kernel with `[[sycl::reqd_sub_group_size(4099)]]`") {
    check_sub_group_size<4099>(device, queue, sub_group_sizes);
  }
}

TEST_CASE(
    "Check is_compatible for kernels with no optional features used and with "
    "`[[sycl::device_has()]]` attribute",
    "[kernel_bundle]") {
  const sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();

  SECTION("for a kernel that uses `sycl::half`") {
    check_with_optional_features<kernels::kernel_fp16_descriptor>(
        device, queue, device.has(sycl::aspect::fp16));
  }

  SECTION("for a kernel that uses `double`") {
    check_with_optional_features<kernels::kernel_fp64_descriptor>(
        device, queue, device.has(sycl::aspect::fp64));
  }

  SECTION("for a kernel that uses 64-bit atomic operations") {
    check_with_optional_features<kernels::kernel_atomic64_descriptor>(
        device, queue, device.has(sycl::aspect::atomic64));
  }
}

TEST_CASE(
    "Check is_compatible(const std::vector<kernel_id>, const device) for "
    "multiple ids",
    "[kernel_bundle]") {
  const sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  std::vector<sycl::kernel_id> builtinKernelIds =
      device.get_info<sycl::info::device::built_in_kernel_ids>();

  using kernel_name = kernels::kernel_work_group_size_descriptor<4294967295>;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name,
                                                sycl::bundle_state::executable>(
      queue);
  builtinKernelIds.push_back(sycl::get_kernel_id<kernel_name::type>());

  std::vector<sycl::kernel_id> builtinKernelIdsOne(
      1, sycl::get_kernel_id<kernel_name::type>());

  CHECK(sycl::is_compatible(builtinKernelIds, device) ==
        (size_t(device.get_info<sycl::info::device::max_work_item_sizes<1>>()) >
         4294967295));
}
