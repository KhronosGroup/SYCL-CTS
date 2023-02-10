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

#ifndef SYCL_CTS_IMAGE_DEFAULT_IMAGE_H
#define SYCL_CTS_IMAGE_DEFAULT_IMAGE_H

#if !(defined(SYCL_CTS_COMPILING_WITH_HIPSYCL) ||    \
      defined(SYCL_CTS_COMPILING_WITH_COMPUTECPP) || \
      defined(SYCL_CTS_COMPILING_WITH_DPCPP))

/**
 * Provides functionality to construct a default sampled_image and associated
 * accessor. */
struct default_sampled_image {
  using data_type = sycl::int4;
  using type = sycl::sampled_image<1>;

  static type get() {
    return {
        nullptr, sycl::image_format::r32b32g32a32_sint,
        sycl::image_sampler{sycl::addressing_mode::mirrored_repeat,
                            sycl::filtering_mode::nearest,
                            sycl::coordinate_normalization_mode::normalized},
        sycl::range<1>{}};
  }

  template <sycl::image_target ImageTarget>
  using acc_type = sycl::sampled_image_accessor<sycl::int4, 1, ImageTarget>;

  template <sycl::image_target ImageTarget>
  static acc_type<ImageTarget> get_acc(type &t, sycl::handler &cgh) {
    return acc_type<ImageTarget>{t, cgh};
  }
};

/**
 * Provides functionality to construct a default unsampled_image and associated
 * accessor. */
struct default_unsampled_image {
  using data_type = sycl::int4;
  using type = sycl::unsampled_image<1>;

  static void get() {
    return {sycl::image_format::r32b32g32a32_sint, sycl::range<1>{1}};
  }

  template <sycl::access_mode AccessMode, sycl::image_target ImageTarget>
  using acc_type =
      sycl::sampled_image_accessor<sycl::int4, 1, AccessMode, ImageTarget>;

  template <sycl::access_mode AccessMode, sycl::image_target ImageTarget>
  static acc_type<AccessMode, ImageTarget> get_acc(type &t,
                                                   sycl::handler &cgh) {
    return acc_type<AccessMode, ImageTarget>{t, cgh};
  }
};

#endif

#endif  // SYCL_CTS_IMAGE_DEFAULT_IMAGE_H
