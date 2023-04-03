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
#include "../common/disabled_for_test_case.h"
#include "../common/semantics_reference.h"
#include "../image/default_image.h"

struct storage_sampled {
  std::size_t size;

  template <typename DataT, int Dimensions, sycl::image_target AccessTarget>
  explicit storage_sampled(
      const sycl::sampled_image_accessor<DataT, Dimensions, AccessTarget>&
          sampled_image_accessor)
      : size(sampled_image_accessor.size()) {}

  template <typename DataT, int Dimensions, sycl::image_target AccessTarget>
  bool check(
      const sycl::sampled_image_accessor<DataT, Dimensions, AccessTarget>&
          sampled_image_accessor) const {
    return sampled_image_accessor.size() == size;
  }
};

TEST_CASE("sampled_image_accessor common reference semantics (host_task)",
          "[sampled_image_accessor]") {
  auto sampled_image_0 = default_sampled_image::get();
  auto sampled_image_1 = default_sampled_image::get();
  sycl_cts::util::get_cts_object::queue().submit([&](sycl::handler& cgh) {
    auto sampled_image_accessor_0 =
        default_sampled_image::get_acc<sycl::image_target::host_task>(
            sampled_image_0, cgh);
    auto sampled_image_accessor_1 =
        default_sampled_image::get_acc<sycl::image_target::host_task>(
            sampled_image_1, cgh);
    cgh.host_task([=] {
      auto sampled_image_accessor_0_copy = sampled_image_accessor_0;
      common_reference_semantics::check_host<storage_sampled>(
          sampled_image_accessor_0_copy, sampled_image_accessor_1,
          "sampled_image_accessor<int4, 1, image_target::host_task>");
    });
  }).wait_and_throw();
}

TEST_CASE("sampled_image_accessor common reference semantics (kernel)",
          "[sampled_image_accessor]") {
  auto sampled_image = default_sampled_image::get();
  using type = default_sampled_image::acc_type<sycl::image_target::device>;
  using data_type = default_sampled_image::data_type;
  common_reference_semantics::check_kernel<storage_sampled, type>(
      [&](sycl::handler& cgh) {
        return sampled_image.get_access<data_type>(cgh);
      },
      "sampled_image_accessor<int4, 1, image_target::device>");
}

struct storage_unsampled {
  std::size_t size;

  template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
            sycl::image_target AccessTarget>
  explicit storage_unsampled(const sycl::unsampled_image_accessor<
                             DataT, Dimensions, AccessMode, AccessTarget>&
                                 unsampled_image_accessor)
      : size(unsampled_image_accessor.size()) {}

  template <typename DataT, int Dimensions, sycl::access_mode AccessMode,
            sycl::image_target AccessTarget>
  bool check(const sycl::unsampled_image_accessor<DataT, Dimensions, AccessMode,
                                                  AccessTarget>&
                 unsampled_image_accessor) const {
    return unsampled_image_accessor.size() == size;
  }
};

TEST_CASE("unsampled_image_accessor common reference semantics (host_task)",
          "[unsampled_image_accessor]") {
  auto unsampled_image_0 = default_unsampled_image::get();
  auto unsampled_image_1 = default_unsampled_image::get();
  sycl_cts::util::get_cts_object::queue().submit([&](sycl::handler& cgh) {
    auto unsampled_image_accessor_0 =
        default_unsampled_image::get_acc<sycl::access_mode::read,
                                         sycl::image_target::host_task>(
            unsampled_image_0, cgh);

    auto unsampled_image_accessor_1 =
        default_unsampled_image::get_acc<sycl::access_mode::read,
                                         sycl::image_target::host_task>(
            unsampled_image_1, cgh);
    cgh.host_task([=] {
      auto unsampled_image_accessor_0_copy = unsampled_image_accessor_0;
      common_reference_semantics::check_host<storage_unsampled>(
          unsampled_image_accessor_0_copy, unsampled_image_accessor_1,
          "unsampled_image_accessor<int4, 1, access_mode::read, "
          "image_target::host_task>");
    });
  }).wait_and_throw();
}

TEST_CASE("unsampled_image_accessor common reference semantics (kernel)",
          "[unsampled_image_accessor]") {
  auto unsampled_image = default_unsampled_image::get();
  using type = default_unsampled_image::acc_type<sycl::access_mode::read,
                                                 sycl::image_target::device>;
  using data_type = default_unsampled_image::data_type;
  common_reference_semantics::check_kernel<storage_unsampled, type>(
      [&](sycl::handler& cgh) {
        return unsampled_image.get_access<data_type, sycl::access_mode::read>(
            cgh);
      },
      "unsampled_image_accessor<int4, 1, access_mode::read, "
      "image_target::device>");
}
