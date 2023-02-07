/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022-2023 The Khronos Group Inc.
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

#ifndef SYCL_CTS_TEST_IMAGE_IMAGE_COMMON_H
#define SYCL_CTS_TEST_IMAGE_IMAGE_COMMON_H

#include "../common/common.h"

namespace {

using namespace sycl_cts;

/**
 * @brief Helps to smooth out some differences between 1D and higher-dimensional
 *        images
 * @tparam dims Number of image dimensions
 * @tparam AllocatorT Allocator type to use to allocate image data
 */
template <int dims, typename AllocatorT = sycl::image_allocator >
struct image_generic {
  /**
   * @brief Compares an image pitch to the provided pitch. If the provided pitch
   *        is null, no comparison is done and true is returned.
   * @param log Logger object
   * @param img Image to retrieve the pitch from
   * @param pitch Pointer to the pitch that the image pitch will be compared to
   * return True if pitches are equal
   */
  static bool compare_pitch(util::logger &log, sycl::image<3, AllocatorT> &img,
                            sycl::range<2> *pitch) {
    if (pitch == nullptr) {
      // Don't compare if pitch is null
      return true;
    }
    if ((!CHECK_VALUE_SCALAR(log, img.get_pitch()[0], pitch->get(0))) ||
        (!CHECK_VALUE_SCALAR(log, img.get_pitch()[1], pitch->get(1)))) {
      return false;
    }
    return true;
  }

  static bool compare_pitch(util::logger &log, sycl::image<2, AllocatorT> &img,
                            sycl::range<1> *pitch) {
    if (pitch == nullptr) {
      // Don't compare if pitch is null
      return true;
    }
    if (!CHECK_VALUE_SCALAR(log, img.get_pitch()[0], pitch->get(0))) {
      return false;
    }
    return true;
  }

  /**
   * @brief Compares pitches of two images
   * @param log Logger object
   * @param imgA First image where the pitch will be retrieved from
   * @param imgB Second image where the pitch will be retrieved from
   * return True if pitches are equal
   */
  static bool compare_pitch(util::logger &log, sycl::image<dims, AllocatorT> &imgA,
                            sycl::image<dims, AllocatorT> &imgB) {
    auto pitch = imgB.get_pitch();
    return compare_pitch(log, imgA, &pitch);
  }

  /**
   * @brief Constructs an image, using a pitch parameter. If the provided pitch
   *        is null, it is not used for construction.
   * @param imageData Host data that will be used to construct the image
   * @param channelOrder Image channel order
   * @param channelType Image channel type
   * @param r Range used of the image
   * @param pitch Image pitch
   * @return The constructed image
   */
    static sycl::image<dims, AllocatorT> create_with_pitch_and_allocator(
      void *imageData, sycl::image_channel_order channelOrder,
      sycl::image_channel_type channelType, sycl::range<dims> &r,
      sycl::range<dims - 1> *pitch) {
    if (pitch == nullptr) {
      return sycl::image<dims, AllocatorT>(imageData, channelOrder,
                                               channelType, r);
    } else {
      return sycl::image<dims, AllocatorT>(imageData, channelOrder,
                                               channelType, r, *pitch);
    }
  }

  /**
   * @brief Constructs an image, using a pitch parameter. If the provided pitch
   *        is null, it is not used for construction.
   * @param imageData Host data that will be used to construct the image
   * @param channelOrder Image channel order
   * @param channelType Image channel type
   * @param r Range used of the image
   * @param pitch Image pitch
   * @return The constructed image
   */
  static sycl::image<dims> create_with_pitch(
      void *imageData, sycl::image_channel_order channelOrder,
      sycl::image_channel_type channelType, sycl::range<dims> &r,
      sycl::range<dims - 1> *pitch) {
    return create_with_pitch_and_allocator(
        imageData, channelOrder, channelType, r, pitch);
  }

  /**
   * @brief Multiplies the pitch with a custom multiplier
   * @param pitch Image pitch
   * @param multiplier What to multiply the pitch by
   */
  static void multiply_pitch(sycl::range<dims - 1> *pitch,
                             size_t multiplier) {
    if (pitch == nullptr) {
      return;
    }
    *pitch *= multiplier;
  }
};
/**
 * @brief Specialization for one dimension
 */
template <typename AllocatorT>
struct image_generic<1, AllocatorT> {
  static bool compare_pitch(util::logger &log, sycl::image<1, AllocatorT> &img,
                            void *pitch) {
    // 1D images don't have a get_pitch() method
    return true;
  }
  static bool compare_pitch(util::logger &log, sycl::image<1, AllocatorT> &imgA,
                            sycl::image<1, AllocatorT> &imgB) {
    // 1D images don't have a get_pitch() method
    return true;
  }
  static sycl::image<1, AllocatorT> create_with_pitch_and_allocator(
      void *imageData, sycl::image_channel_order channelOrder,
      sycl::image_channel_type channelType, sycl::range<1> &r,
      void *pitch) {
    // 1D images cannot be constructed with a pitch
    return sycl::image<1, AllocatorT>(static_cast<void *>(imageData),
                                          channelOrder, channelType, r);
  }
  static sycl::image<1> create_with_pitch(
      void *imageData, sycl::image_channel_order channelOrder,
      sycl::image_channel_type channelType, sycl::range<1> &r,
      void *pitch) {
    // 1D images cannot be constructed with a pitch
    return create_with_pitch_and_allocator(
        imageData, channelOrder, channelType, r, pitch);
  }

  static void multiply_pitch(void *pitch, size_t multiplier) {
    // There is no pitch for 1D images
  }
};

/**
 * @brief Map entry type that specifies the size and name (as a string) for an
 *        image channel type
 */
struct channel_type_size {
  sycl::image_channel_type type;
  unsigned int size;
  const char *str;
};

/**
 * @brief Map that stores the sizes and names (as strings) for all image
 *        channel types
 */
channel_type_size g_channelTypeSize[] = {
    {sycl::image_channel_type::snorm_int8, 1, "snorm_int8"},
    {sycl::image_channel_type::snorm_int16, 2, "snorm_int16"},
    {sycl::image_channel_type::unorm_int8, 1, "unorm_int8"},
    {sycl::image_channel_type::unorm_int16, 2, "unorm_int16"},
    {sycl::image_channel_type::unorm_short_555, 2, "unorm_short_555"},
    {sycl::image_channel_type::unorm_short_565, 2, "unorm_short_565"},
    {sycl::image_channel_type::unorm_int_101010, 4, "unorm_int_101010"},
    {sycl::image_channel_type::signed_int8, 1, "signed_int8"},
    {sycl::image_channel_type::signed_int16, 2, "signed_int16"},
    {sycl::image_channel_type::signed_int32, 4, "signed_int32"},
    {sycl::image_channel_type::unsigned_int8, 1, "unsigned_int8"},
    {sycl::image_channel_type::unsigned_int16, 2, "unsigned_int16"},
    {sycl::image_channel_type::unsigned_int32, 4, "unsigned_int32"},
    {sycl::image_channel_type::fp16, sizeof(sycl::half), "fp16"},
    {sycl::image_channel_type::fp32, sizeof(float), "fp32"}};

/**
 * @brief The total number of different image channel types
 */
static const int NUM_CHANNEL_TYPES =
    (sizeof(g_channelTypeSize) / sizeof(channel_type_size));

/**
 * @brief Retrieve the size of the image channel type
 */
inline unsigned int get_channel_type_size(sycl::image_channel_type channelType) {
  for (int i = 0; i < NUM_CHANNEL_TYPES; ++i) {
    if (g_channelTypeSize[i].type == channelType) {
      return g_channelTypeSize[i].size;
    }
  }
  // This should never be reached
  return 0;
}
/**
 * @brief Retrieve the name of the image channel type
 */
inline const char *get_channel_type_string(sycl::image_channel_type channelType) {
  for (int i = 0; i < NUM_CHANNEL_TYPES; ++i) {
    if (g_channelTypeSize[i].type == channelType) {
      return g_channelTypeSize[i].str;
    }
  }
  // This should never be reached
  return "";
}

/**
 * @brief Map entry type that specifies the count and name (as a string) for an
 *        image channel order
 */
struct channel_order_count {
  sycl::image_channel_order order;
  unsigned int count;
  const char *str;
};

/**
 * @brief Map that stores the counts and names (as strings) for all image
 *        channel orders
 */
channel_order_count g_channelOrderCount[] = {
    // rgba and bgra are the mandated minimum supported
    {sycl::image_channel_order::rgba, 4, "rgba"},
    {sycl::image_channel_order::bgra, 4, "bgra"},

    {sycl::image_channel_order::argb, 4, "argb"},
    {sycl::image_channel_order::abgr, 4, "abgr"},

    {sycl::image_channel_order::r, 1, "r"},
    {sycl::image_channel_order::a, 1, "a"},

    {sycl::image_channel_order::rg, 2, "rg"},
    {sycl::image_channel_order::ra, 2, "ra"},
    {sycl::image_channel_order::rx, 2, "rx"},
    {sycl::image_channel_order::rgx, 3, "rgx"},

    {sycl::image_channel_order::intensity, 1, "intensity"},
    {sycl::image_channel_order::luminance, 1, "luminance"},

    // These two are special cases
    {sycl::image_channel_order::rgb, 1, "rgb"},
    {sycl::image_channel_order::rgbx, 1, "rgbx"}};

/**
 * @brief The total number of different image channel orders
 */
static const int NUM_CHANNEL_ORDERS =
    (sizeof(g_channelOrderCount) / sizeof(channel_order_count));

/**
 * @brief The number of channel orders that are guaranteed to be supported by
 *        the device - rgba and bgra
 */
static const int MINIMUM_CHANNEL_ORDERS = 2;

/**
 * @brief Retrieve the count of the image channel order
 */
inline unsigned int get_channel_order_count(
    sycl::image_channel_order channelOrder) {
  for (int i = 0; i < NUM_CHANNEL_ORDERS; ++i) {
    if (g_channelOrderCount[i].order == channelOrder) {
      return g_channelOrderCount[i].count;
    }
  }
  // This should never be reached
  return 0;
}

/**
 * @brief Retrieve the name of the image channel order
 */
inline const char *get_channel_order_string(
    sycl::image_channel_order channelOrder) {
  for (int i = 0; i < NUM_CHANNEL_ORDERS; ++i) {
    if (g_channelOrderCount[i].order == channelOrder) {
      return g_channelOrderCount[i].str;
    }
  }
  // This should never be reached
  return "";
}

/**
 * @brief Map entry type that specifies valid image channel types for an image
 *        channel order
 */
struct collection {
  sycl::image_channel_order order;
  int numChannelTypes;
  sycl::image_channel_type typeArray[NUM_CHANNEL_TYPES];
};

/**
 * @brief Retrieves a test set, which consist of an image channel order and all
 *        valid image channel types for that order.
 * @param order Image channel order to be tested.
 * @return Collection of channel order and all valid channel types for the order
 */
inline collection get_test_set_full(sycl::image_channel_order order) {
  collection testSet;
  testSet.order = order;
  // Not all combinations are allowed
  switch (order) {
    case sycl::image_channel_order::argb:
    case sycl::image_channel_order::bgra:
      testSet.typeArray[0] = sycl::image_channel_type::unorm_int8;
      testSet.typeArray[1] = sycl::image_channel_type::snorm_int8;
      testSet.typeArray[2] = sycl::image_channel_type::signed_int8;
      testSet.typeArray[3] = sycl::image_channel_type::unsigned_int8;
      testSet.numChannelTypes = 4;
      break;
    case sycl::image_channel_order::intensity:
    case sycl::image_channel_order::luminance:
      testSet.typeArray[0] = sycl::image_channel_type::unorm_int8;
      testSet.typeArray[1] = sycl::image_channel_type::unorm_int16;
      testSet.typeArray[2] = sycl::image_channel_type::snorm_int8;
      testSet.typeArray[3] = sycl::image_channel_type::snorm_int16;
      testSet.typeArray[4] = sycl::image_channel_type::fp16;
      testSet.typeArray[5] = sycl::image_channel_type::fp32;
      testSet.numChannelTypes = 6;
      break;
    case sycl::image_channel_order::rgb:
    case sycl::image_channel_order::rgbx:
      testSet.typeArray[0] = sycl::image_channel_type::unorm_short_555;
      testSet.typeArray[1] = sycl::image_channel_type::unorm_short_565;
      testSet.typeArray[2] = sycl::image_channel_type::unorm_int_101010;
      testSet.numChannelTypes = 3;
      break;
    default:
      // Most channel orders allow most channel types
      auto &itTestSet = testSet.numChannelTypes;
      itTestSet = 0;
      for (int itType = 0; itType < NUM_CHANNEL_TYPES; ++itType) {
        auto channelType = g_channelTypeSize[itType].type;
        switch (channelType) {
          case sycl::image_channel_type::unorm_short_555:
          case sycl::image_channel_type::unorm_short_565:
          case sycl::image_channel_type::unorm_int_101010:
            // These types are exceptions, can only be used with specific
            // channel orders
            break;
          default:
            testSet.typeArray[itTestSet] = g_channelTypeSize[itType].type;
            ++itTestSet;
            break;
        }
      }
      break;
  }
  return testSet;
}

/**
 * @brief Retrieves a test set, which consist of an image channel order and the
 *        mandated minimum of valid image channel types for that order.
 * @param order Image channel order to be tested.
 * @return Collection of channel order and the minimum of valid channel types
 *         for the order
 */
inline collection get_test_set_minimum(sycl::image_channel_order order) {
  collection testSet;
  testSet.order = order;
  switch (order) {
    case sycl::image_channel_order::rgba:
      testSet.typeArray[0] = sycl::image_channel_type::unorm_int8;
      testSet.typeArray[1] = sycl::image_channel_type::unorm_int16;
      testSet.typeArray[2] = sycl::image_channel_type::signed_int8;
      testSet.typeArray[3] = sycl::image_channel_type::signed_int16;
      testSet.typeArray[4] = sycl::image_channel_type::signed_int32;
      testSet.typeArray[5] = sycl::image_channel_type::unsigned_int8;
      testSet.typeArray[6] = sycl::image_channel_type::unsigned_int16;
      testSet.typeArray[7] = sycl::image_channel_type::unsigned_int32;
      testSet.typeArray[8] = sycl::image_channel_type::fp16;
      testSet.typeArray[9] = sycl::image_channel_type::fp32;
      testSet.numChannelTypes = 10;
      break;
    case sycl::image_channel_order::bgra:
      testSet.typeArray[0] = sycl::image_channel_type::unorm_int8;
      testSet.numChannelTypes = 1;
      break;
    default:
      // No other combinations are required
      testSet.numChannelTypes = 0;
      break;
  }
  return testSet;
}

template <int dims>
std::vector<sycl::byte> get_image_host(
    unsigned int numElems, unsigned int channelCount,
    unsigned int channelTypeSize) {
  const auto sizePerChannelType = (dims * numElems * channelCount);
  const auto size = (sizePerChannelType * channelTypeSize);
  std::vector<sycl::byte> imageHost(size);
  for (unsigned int ii = 0; ii < sizePerChannelType; ii++) {
    for (unsigned int ij = 0; ij < channelTypeSize; ij++) {
      imageHost[ii * channelTypeSize + ij] = ij + 1;
    }
  }
  return imageHost;
}

template <int dims>
struct image_kernel_read;
template <int dims>
struct image_kernel_write;
template <int dims, typename T, typename AllocatorT>
struct image_kernel_access;
class empty_kernel {
 public:
  void operator()() const {}
};

template <int dims>
class image_api_check {
  int calc_numElems(util::logger &log, sycl::range<dims> &r,
                    sycl::range<dims - 1> *p) const {
    auto numElems = 1;
    switch (dims) {
      case 1:
        log.note("Testing image combination: dims[%d], range[%d]", dims, r[0]);
        numElems = r[0];
        break;
      case 2:
        log.note(
            "Testing image combination: dims[%d], range[%d, %d], pitch[%d]",
            dims, r[0], r[1], (p != nullptr));
        numElems = r[0] * r[1];
        break;
      case 3:
        log.note(
            "Testing image combination: dims[%d], range[%d, %d, %d], pitch[%d]",
            dims, r[0], r[1], r[2], (p != nullptr));
        numElems = r[0] * r[1] * r[2];
        break;
      default:
        break;
    }
    return numElems;
  }

  template <typename AllocatorT = sycl::image_allocator>
  void check_api(util::logger &log, sycl::image<dims, AllocatorT> &img,
                  sycl::range<dims> &r, sycl::range<dims - 1> *p,
                  int numElems, int elementSize,
                  std::vector<sycl::byte> &finalData) const {

    // Check get_range()
    if (dims == 3) {
      if ((!CHECK_VALUE_SCALAR(log, img.get_range()[0], r.get(0))) ||
          (!CHECK_VALUE_SCALAR(log, img.get_range()[1], r.get(1))) ||
          (!CHECK_VALUE_SCALAR(log, img.get_range()[2], r.get(2)))) {
        FAIL(log, "Ranges are not the same.");
      }
    }
    if (dims == 2) {
      if ((!CHECK_VALUE_SCALAR(log, img.get_range()[0], r.get(0))) ||
          (!CHECK_VALUE_SCALAR(log, img.get_range()[1], r.get(1)))) {
        FAIL(log, "Ranges are not the same.");
      }
    }
    if (dims == 1) {
      if ((!CHECK_VALUE_SCALAR(log, img.get_range()[0], r.get(0)))) {
        FAIL(log, "Ranges are not the same.");
      }
    }

    // Check get_pitch()
    if (!image_generic<dims, AllocatorT>::compare_pitch(log, img, p)) {
      FAIL(log, "Pitches are not the same.");
    }

    // Check get_size()
    if (img.get_size() < (numElems * elementSize)) {
      std::string message = "Sizes are not the same: expected at least " +
          std::to_string(numElems * elementSize) + ", got " +
          std::to_string(img.get_size());
      FAIL(log, message);
    }

    // Check get_count()
    if (!CHECK_VALUE_SCALAR(log, img.get_count(), numElems)) {
      FAIL(log, "Counts are not the same.");
    }

    // Check set_write_back()
    {
      img.set_write_back();
      img.set_write_back(false);
      img.set_write_back(true);
    }

    // Check set_final_data()
    {
      auto rawPtr = finalData.data();
      auto rawPtrVoid = static_cast<void *>(rawPtr);
      auto rawPtrFloat = static_cast<float *>(rawPtrVoid);
      auto sharedPtrVoid =
          std::shared_ptr<void>(rawPtrVoid, [](void *) {});
      auto sharedPtrFloat =
          std::shared_ptr<float>(rawPtrFloat, [](float *) {});
      auto weakPtrVoid = std::weak_ptr<void>(sharedPtrVoid);
      auto weakPtrFloat = std::weak_ptr<float>(sharedPtrFloat);
      auto iterator = finalData.begin();

      img.set_final_data();
      img.set_final_data(nullptr);
      img.set_final_data(rawPtr);
      img.set_final_data(rawPtrVoid);
      img.set_final_data(rawPtrFloat);
      img.set_final_data(sharedPtrVoid);
      img.set_final_data(sharedPtrFloat);
      img.set_final_data(weakPtrVoid);
      img.set_final_data(weakPtrFloat);
      img.set_final_data(iterator);
    }

  }

  template <typename AllocatorT = sycl::image_allocator>
  void check_image_properties(util::logger &log, sycl::image<dims, AllocatorT> &img) const {
    /* check has_property() */

      auto hasHostPtrProperty =
          img.template has_property<sycl::property::image::use_host_ptr>();
      check_return_type<bool>(log, hasHostPtrProperty,
                              "has_property<use_host_ptr>()");

      auto hasUseMutexProperty =
          img.template has_property<sycl::property::image::use_mutex>();
      check_return_type<bool>(log, hasUseMutexProperty,
                              "has_property<use_mutex>()");

      auto hasContentBoundProperty = img.template has_property<
          sycl::property::image::context_bound>();
      check_return_type<bool>(log, hasContentBoundProperty,
                              "has_property<context_bound>()");

      /* check get_property() */

      auto hostPtrProperty =
          img.template get_property<sycl::property::image::use_host_ptr>();
      check_return_type<sycl::property::image::use_host_ptr>(
          log, hostPtrProperty, "get_property<use_host_ptr>()");

      auto useMutexProperty =
          img.template get_property<sycl::property::image::use_mutex>();
      check_return_type<sycl::property::image::use_mutex>(
          log, useMutexProperty, "get_property<use_mutex>()");
      check_return_type<std::mutex *>(
          log, useMutexProperty.get_mutex_ptr(),
          "image::use_mutex::get_mutex_ptr()");

      auto contextBoundProperty = img.template get_property<
          sycl::property::image::context_bound>();
      check_return_type<sycl::property::image::context_bound>(
          log, contextBoundProperty, "get_property<context_bound>()");
      check_return_type<sycl::context>(
          log, contextBoundProperty.get_context(),
          "image::context_bound::get_context()");
  }

 public:
  template <typename T, typename AllocatorT>
  void check_get_access(util::logger &log, sycl::image_channel_order channelOrder,
                        sycl::image_channel_type channelType, sycl::range<dims> &r,
                        sycl::range<dims - 1> *p, bool multiply) const {

      auto numElems = calc_numElems(log, r, p);

      const auto channelCount = get_channel_order_count(channelOrder);
      const auto channelTypeSize = get_channel_type_size(channelType);
      const auto elementSize = channelTypeSize * channelCount;

      // Pitch has to be at least as large as the multiple of element size
      if (multiply) {
        image_generic<dims>::multiply_pitch(p, elementSize);
      }

      // Create image host data
      auto imageHost =
          get_image_host<dims>(numElems, channelTypeSize, channelCount);

      // Creates an image, either with a pitch or without
      // The pitch is not used for a 1D image or when null
      sycl::image<dims, AllocatorT> img =
              image_generic<dims, AllocatorT>::create_with_pitch_and_allocator(
                static_cast<void *>(imageHost.data()), channelOrder, channelType, r, p);
      {
        // target::host_image
        img.template get_access<T, sycl::access_mode::read_write>();
      }

      auto testQueue = sycl_cts::util::get_cts_object::queue();
      try {
        testQueue.submit([&](sycl::handler &cgh) {
          // target::image
          auto imgAcc =
              img.template get_access<T, sycl::access_mode::write>(cgh);
          using kname = image_kernel_access<dims, T, AllocatorT>;
          cgh.single_task<kname>(empty_kernel());
        });

        testQueue.wait_and_throw();
      } catch (const sycl::feature_not_supported &fnse) {
        if (!testQueue.get_device()
                 .template get_info<sycl::info::device::image_support>()) {
          log.note("device does not support images -- skipping check");
        } else {
          throw;
        }
      }
  }

  void operator()(util::logger &log, sycl::range<dims> &r,
                  sycl::range<dims - 1> *p = nullptr) const {
    auto numElems = calc_numElems(log, r, p);

    // This combination is required to be supported by the OpenCL spec
    const auto channelOrder = sycl::image_channel_order::rgba;
    const auto channelType = sycl::image_channel_type::fp32;

    // Prepare variables
    const auto channelCount = get_channel_order_count(channelOrder);
    const auto channelTypeSize = get_channel_type_size(channelType);
    const auto elementSize = channelTypeSize * channelCount;

    // Pitch has to be at least as large as the multiple of element size
    image_generic<dims>::multiply_pitch(p, elementSize);

    // Create image host data
    auto imageHost =
        get_image_host<dims>(numElems, channelTypeSize, channelCount);

    // Create final data
    // Must outlive the image
    auto finalData =
        get_image_host<dims>(numElems, channelTypeSize, channelCount);

    using AllocatorT = std::allocator<sycl::byte>;

    // Creates an image, either with a pitch or without
    // The pitch is not used for a 1D image or when null
    sycl::image<dims> img = image_generic<dims>::create_with_pitch(
        static_cast<void *>(imageHost.data()), channelOrder, channelType, r, p);

    check_api(log, img, r, p, numElems, elementSize, finalData);

    auto imgAlloc = image_generic<dims, AllocatorT>::create_with_pitch_and_allocator
                        (static_cast<void *>(imageHost.data()), channelOrder,
                          channelType, r, p);

    check_api<AllocatorT>(log, imgAlloc, r, p, numElems, elementSize, finalData);


    // Check get_allocator()
    {
      auto defaultAllocator = img.get_allocator();

      check_return_type<sycl::image_allocator>(log, defaultAllocator, "get_allocator()");

      /* create another image with a custom allocator */
      auto imgAlloc2 =
          image_generic<dims, AllocatorT>::create_with_pitch_and_allocator
                        (static_cast<void *>(imageHost.data()), channelOrder,
                          channelType, r, p);

      auto allocator = imgAlloc2.get_allocator();

      check_return_type<AllocatorT>(log, allocator, "get_allocator()");

      auto ptr = allocator.allocate(1);
      if (ptr == nullptr) {
        FAIL(log, "get_allocator() returned an invalid allocator");
      }
      allocator.deallocate(ptr, 1);
    }

    /* Check image properties */
    {
      std::mutex mutex;
      auto context = util::get_cts_object::context();
      const sycl::property_list propList{
          sycl::property::image::use_host_ptr(),
          sycl::property::image::use_mutex(mutex),
          sycl::property::image::context_bound(context)};

      // Create another image
      auto img2 = sycl::image<dims>(static_cast<void *>(imageHost.data()),
                                        channelOrder, channelType, r, propList);

      check_image_properties(log, img2);

      auto imgAlloc2 = sycl::image<dims, AllocatorT>(static_cast<void *>(imageHost.data()),
                                        channelOrder, channelType, r, AllocatorT(), propList);
      check_image_properties<AllocatorT>(log, imgAlloc2);
    }

    // Check get_access()
    {
      check_get_access<sycl::cl_int4, sycl::image_allocator>(log,
              channelOrder, sycl::image_channel_type::signed_int32, r, p, false);
      check_get_access<sycl::cl_uint4, sycl::image_allocator>(log,
              channelOrder, sycl::image_channel_type::unsigned_int32, r, p, false);
      check_get_access<sycl::cl_float4, sycl::image_allocator>(log,
              channelOrder, channelType, r, p, false);

      check_get_access<sycl::cl_int4, AllocatorT>(log, channelOrder,
                            sycl::image_channel_type::signed_int32, r, p, false);
      check_get_access<sycl::cl_uint4, AllocatorT>(log, channelOrder,
                            sycl::image_channel_type::unsigned_int32, r, p, false);
      check_get_access<sycl::cl_float4, AllocatorT>(log, channelOrder,
                              channelType, r, p, false);
    }
    const auto expected = 0.5f;

    // Check read/write APIs.
    {
      auto queue = util::get_cts_object::queue();

      if (queue.get_device()
                 .template get_info<sycl::info::device::image_support>()) {
        {
          sycl::image<dims> img2 = image_generic<dims>::create_with_pitch(
              static_cast<void *>(imageHost.data()), channelOrder, channelType,
              r, p);
          queue.submit([&](sycl::handler &cgh) {
          auto img_acc =
              img2.template get_access<sycl::float4,
                                      sycl::access_mode::read>(cgh);

          auto sampler = sycl::sampler(
              sycl::coordinate_normalization_mode::unnormalized,
              sycl::addressing_mode::clamp,
              sycl::filtering_mode::nearest);

          auto myKernel = [img_acc, sampler](sycl::item<dims> item) {
              // Read image data using integer coordinates
              sycl::float4 dataFromInt =
                              img_acc.read(image_access<dims>::get_int(item));
              (void)dataFromInt;  // silent warning
              // Read image data using integer coordinates and a sampler
              sycl::float4 dataFromIntWithSampler =
                  img_acc.read(image_access<dims>::get_int(item), sampler);
              (void)dataFromIntWithSampler;  // silent warning
              // Read image data using floating point coordinates
              // Only works with a sampler
              sycl::float4 dataFromFloat =
                  img_acc.read(image_access<dims>::get_float(item), sampler);
              (void)dataFromFloat;  // silent warning
          };
          cgh.parallel_for<image_kernel_read<dims>>(r, myKernel);
          });

          queue.submit([&](sycl::handler &cgh) {
            auto img_acc = img2.template get_access<sycl::float4,
                                          sycl::access_mode::write>(cgh);
            auto myKernel = [expected, img_acc](sycl::item<dims> item) {
              // Write image data
              img_acc.write(image_access<dims>::get_int(item),
                                                  sycl::float4(expected));
            };
            cgh.parallel_for<image_kernel_write<dims>>(r, myKernel);
          });

          queue.wait_and_throw();
        }  // End of scope for img2 object.

        // Check image values
        const auto floatData = reinterpret_cast<float *>(imageHost.data());
        for (int i = 0; i < numElems; ++i) {
          if (!CHECK_VALUE(log, floatData[i], 0.5f, i)) {
            FAIL(log, "Image contains wrong values.");
          }
        }
      }
    }
  }
};

}  // namespace

#endif  // SYCL_CTS_TEST_IMAGE_IMAGE_COMMON_H
