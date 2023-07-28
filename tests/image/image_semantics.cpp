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
#include "default_image.h"

#if !SYCL_CTS_COMPILING_WITH_HIPSYCL

template <int Dimensions>
struct storage_sampled {
  sycl::range<Dimensions> range;
  std::size_t byte_size;
  std::size_t size;

  explicit storage_sampled(const sycl::sampled_image<Dimensions>& sampled_image)
      : range(sampled_image.get_range()),
        byte_size(sampled_image.byte_size()),
        size(sampled_image.size()) {}

  bool check(const sycl::sampled_image<Dimensions>& sampled_image) const {
    return sampled_image.get_range() == range &&
           sampled_image.byte_size() == byte_size &&
           sampled_image.size() == size;
  }
};

template <int Dimensions>
struct storage_unsampled {
  sycl::range<Dimensions> range;
  std::size_t byte_size;
  std::size_t size;

  explicit storage_unsampled(
      const sycl::unsampled_image<Dimensions>& unsampled_image)
      : range(unsampled_image.get_range()),
        byte_size(unsampled_image.byte_size()),
        size(unsampled_image.size()) {}

  bool check(const sycl::unsampled_image<Dimensions>& unsampled_image) const {
    return unsampled_image.get_range() == range &&
           unsampled_image.byte_size() == byte_size &&
           unsampled_image.size() == size;
  }
};

#endif
DISABLED_FOR_TEST_CASE(hipSYCL)
("sampled_image common reference semantics", "[sampled_image]")({
  auto sampled_image_0 = default_sampled_image::get();
  auto sampled_image_1 = default_sampled_image::get();

  common_reference_semantics::check_host<storage_sampled<1>>(
      sampled_image_0, sampled_image_1, "sampled_image");
});

// sampled_image common reference semantics, mutation
// Cannot be tested, since sampled_image is read-only.

DISABLED_FOR_TEST_CASE(hipSYCL)
("unsampled_image common reference semantics", "[unsampled_image]")({
  auto unsampled_image_0 = default_unsampled_image::get();
  auto unsampled_image_1 = default_unsampled_image::get();

  common_reference_semantics::check_host<storage_unsampled<1>>(
      unsampled_image_0, unsampled_image_1, "unsampled_image");
});

DISABLED_FOR_TEST_CASE(hipSYCL)
("unsampled_image common reference semantics, mutation", "[unsampled_image]")({
  using data_type = default_unsampled_image::data_type;
  constexpr data_type val{1};
  constexpr data_type new_val{1};
  auto t0 = default_unsampled_image::get();

  SECTION("mutation to copy") {
    t0.get_host_access<data_type>().write(0, val);
    sycl::unsampled_image<1> t1(t0);
    t1.get_host_access<data_type>().write(0, new_val);
    data_type read_val = t0.get_host_access<data_type>().read(0);
    CHECK(value_operations::are_equal(new_val, read_val));
  }

  SECTION("mutation to original") {
    t0.get_host_access<data_type>().write(0, val);
    sycl::unsampled_image<1> t1(t0);
    t0.get_host_access<data_type>().write(0, new_val);
    data_type read_val = t1.get_host_access<data_type>().read(0);
    CHECK(value_operations::are_equal(new_val, read_val));
  }
});
