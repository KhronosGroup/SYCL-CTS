/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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
