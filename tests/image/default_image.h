/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#ifndef SYCL_CTS_IMAGE_DEFAULT_IMAGE_H
#define SYCL_CTS_IMAGE_DEFAULT_IMAGE_H

#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP

/**
 * Provides functionality to construct a default sampled_image and associated
 * accessor. */
struct default_sampled_image {
  using data_type = sycl::int4;
  using type = sycl::sampled_image<1>;

  static type get() {
    return {nullptr, sycl::image_format::r32b32g32a32_sint,
            sycl::image_sampler{sycl::addressing_mode::mirrored_repeat,
                                sycl::coordinate_normalization_mode::normalized,
                                sycl::filtering_mode::nearest},
            sycl::range<1>{1}};
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

  static type get() {
    return {sycl::image_format::r32b32g32a32_sint, sycl::range<1>{1}};
  }

  template <sycl::access_mode AccessMode, sycl::image_target ImageTarget>
  using acc_type =
      sycl::unsampled_image_accessor<sycl::int4, 1, AccessMode, ImageTarget>;

  template <sycl::access_mode AccessMode, sycl::image_target ImageTarget>
  static acc_type<AccessMode, ImageTarget> get_acc(type &t,
                                                   sycl::handler &cgh) {
    return acc_type<AccessMode, ImageTarget>{t, cgh};
  }
};

#endif

#endif  // SYCL_CTS_IMAGE_DEFAULT_IMAGE_H
