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
#include "../image/default_image.h"

struct storage_sampled {
  std::size_t size;

  template <typename DataT, int Dimensions>
  explicit storage_sampled(
      const sycl::host_sampled_image_accessor<DataT, Dimensions>&
          host_sampled_image_accessor)
      : size(host_sampled_image_accessor.size()) {}

  template <typename DataT, int Dimensions>
  bool check(const sycl::host_sampled_image_accessor<DataT, Dimensions>&
                 host_sampled_image_accessor) const {
    return size == host_sampled_image_accessor.size();
  }
};

TEST_CASE("host_sampled_image_accessor common reference semantics",
          "[host_sampled_image_accessor]") {
  using data_type = default_sampled_image::data_type;
  auto sampled_image_0 = default_sampled_image::get();
  sycl::host_sampled_image_accessor<data_type, 1> host_sampled_image_accessor_0{
      sampled_image_0};

  auto sampled_image_1 = default_sampled_image::get();
  sycl::host_sampled_image_accessor<data_type, 1> host_sampled_image_accessor_1{
      sampled_image_1};

  common_reference_semantics::check_host<storage_sampled>(
      host_sampled_image_accessor_0, host_sampled_image_accessor_1,
      "host_sampled_image_accessor<int4, 1>");
}

// host_sampled_image_accessor common reference semantics
// Cannot be tested, since sampled_image is read-only.

struct storage_unsampled {
  std::size_t size;

  template <typename DataT, int Dimensions, sycl::access_mode AccessMode>
  explicit storage_unsampled(
      const sycl::host_unsampled_image_accessor<DataT, Dimensions, AccessMode>&
          host_unsampled_image_accessor)
      : size(host_unsampled_image_accessor.size()) {}

  template <typename DataT, int Dimensions, sycl::access_mode AccessMode>
  bool check(
      const sycl::host_unsampled_image_accessor<DataT, Dimensions, AccessMode>&
          host_unsampled_image_accessor) const {
    return size == host_unsampled_image_accessor.size();
  }
};

TEST_CASE("host_unsampled_image_accessor common reference semantics",
          "[host_unsampled_image_accessor]") {
  using data_type = default_unsampled_image::data_type;
  auto unsampled_image_0 = default_unsampled_image::get();
  sycl::host_unsampled_image_accessor<data_type, 1>
      host_unsampled_image_accessor_0{unsampled_image_0};

  auto unsampled_image_1 = default_unsampled_image::get();
  sycl::host_unsampled_image_accessor<data_type, 1>
      host_unsampled_image_accessor_1{unsampled_image_1};

  common_reference_semantics::check_host<storage_unsampled>(
      host_unsampled_image_accessor_0, host_unsampled_image_accessor_1,
      "host_unsampled_image_accessor<int4>");
}

TEST_CASE("host_unsampled_image_accessor common reference semantics, mutation",
          "[host_unsampled_image_accessor]") {
  using data_type = default_unsampled_image::data_type;
  constexpr data_type val{1};
  constexpr data_type new_val{2};
  auto unsampled_image = default_unsampled_image::get();
  sycl::host_unsampled_image_accessor<data_type, 1> t0{unsampled_image};

  SECTION("mutation to copy") {
    t0.write(0, val);
    sycl::host_unsampled_image_accessor<data_type, 1> t1(t0);
    t1.write(0, new_val);
    CHECK(new_val == t0.read(0));
  }

  SECTION("mutation to original") {
    t0.write(0, val);
    sycl::host_unsampled_image_accessor<data_type, 1> t1(t0);
    t0.write(0, new_val);
    CHECK(new_val == t1.read(0));
  }

  SECTION("mutation to original, const copy") {
    t0.write(0, val);
    const sycl::host_unsampled_image_accessor<data_type, 1> t1(t0);
    t0.write(0, new_val);
    CHECK(new_val == t1.read(0));
  }
}
