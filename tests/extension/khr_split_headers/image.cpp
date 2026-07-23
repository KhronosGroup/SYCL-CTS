/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2026 The Khronos Group Inc.
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

#include "util.h"
#include <catch2/catch_test_macros.hpp>
#include <sycl/khr/split_headers/image.hpp>

namespace khr_split_headers::tests {

TEST_CASE("the image header defines the SYCL_KHR_SPLIT_HEADERS macro",
          "[khr_split_headers][image]") {
#ifdef SYCL_KHR_SPLIT_HEADERS
  constexpr bool macro_is_defined = true;
#else
  constexpr bool macro_is_defined = false;
#endif
  STATIC_REQUIRE(macro_is_defined);
}

TEST_CASE("the image header defines the unsampled_image class",
          "[khr_split_headers][image]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::unsampled_image<1>>);
}

TEST_CASE("the image header defines the sampled_image class",
          "[khr_split_headers][image]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::sampled_image<1>>);
}

TEST_CASE("the image header defines the image_allocator class",
          "[khr_split_headers][image]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::image_allocator>);
}

TEST_CASE("the image header defines the unsampled_image_accessor class",
          "[khr_split_headers][image]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<sycl::unsampled_image_accessor<
                     sycl::vec<float, 4>, 1, sycl::access_mode::read>>);
}

TEST_CASE("the image header defines the sampled_image_accessor class",
          "[khr_split_headers][image]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<
                 sycl::sampled_image_accessor<sycl::vec<float, 4>, 1>>);
}

TEST_CASE("the image header defines the host_unsampled_image_accessor class",
          "[khr_split_headers][image]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<
                 sycl::host_unsampled_image_accessor<sycl::vec<float, 4>, 1>>);
}

TEST_CASE("the image header defines the host_sampled_image_accessor class",
          "[khr_split_headers][image]") {
  STATIC_REQUIRE(sycl_cts::util::is_complete_v<
                 sycl::host_sampled_image_accessor<sycl::vec<float, 4>, 1>>);
}

}  // namespace khr_split_headers::tests
