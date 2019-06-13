/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
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
 */
template <int dims>
struct image_generic {
  /**
   * @brief Compares an image pitch to the provided pitch. If the provided pitch
   *        is null, no comparison is done and true is returned.
   * @param log Logger object
   * @param img Image to retrieve the pitch from
   * @param pitch Pointer to the pitch that the image pitch will be compared to
   * return True if pitches are equal
   */
  static bool compare_pitch(util::logger &log, cl::sycl::image<3> &img,
                            cl::sycl::range<2> *pitch) {
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

  static bool compare_pitch(util::logger &log, cl::sycl::image<2> &img,
                            cl::sycl::range<1> *pitch) {
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
  static bool compare_pitch(util::logger &log, cl::sycl::image<dims> &imgA,
                            cl::sycl::image<dims> &imgB) {
    auto pitch = imgB.get_pitch();
    return compare_pitch(log, imgA, &pitch);
  }

  /**
   * @brief Constructs an image, using a pitch parameter. If the provided pitch
   *        is null, it is not used for construction.
   * @tparam AllocatorT Allocator type to use to allocate image data
   * @param imageData Host data that will be used to construct the image
   * @param channelOrder Image channel order
   * @param channelType Image channel type
   * @param r Range used of the image
   * @param pitch Image pitch
   * @return The constructed image
   */
  template <class AllocatorT>
  static cl::sycl::image<dims, AllocatorT> create_with_pitch_and_allocator(
      void *imageData, cl::sycl::image_channel_order channelOrder,
      cl::sycl::image_channel_type channelType, cl::sycl::range<dims> &r,
      cl::sycl::range<dims - 1> *pitch) {
    if (pitch == nullptr) {
      return cl::sycl::image<dims, AllocatorT>(imageData, channelOrder,
                                               channelType, r);
    } else {
      return cl::sycl::image<dims, AllocatorT>(imageData, channelOrder,
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
  static cl::sycl::image<dims> create_with_pitch(
      void *imageData, cl::sycl::image_channel_order channelOrder,
      cl::sycl::image_channel_type channelType, cl::sycl::range<dims> &r,
      cl::sycl::range<dims - 1> *pitch) {
    return create_with_pitch_and_allocator<cl::sycl::image_allocator>(
        imageData, channelOrder, channelType, r, pitch);
  }

  /**
   * @brief Multiplies the pitch with a custom multiplier
   * @param pitch Image pitch
   * @param multiplier What to multiply the pitch by
   */
  static void multiply_pitch(cl::sycl::range<dims - 1> *pitch,
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
template <>
struct image_generic<1> {
  static bool compare_pitch(util::logger &log, cl::sycl::image<1> &img,
                            void *pitch) {
    // 1D images don't have a get_pitch() method
    return true;
  }
  static bool compare_pitch(util::logger &log, cl::sycl::image<1> &imgA,
                            cl::sycl::image<1> &imgB) {
    // 1D images don't have a get_pitch() method
    return true;
  }
  template <class AllocatorT>
  static cl::sycl::image<1, AllocatorT> create_with_pitch_and_allocator(
      void *imageData, cl::sycl::image_channel_order channelOrder,
      cl::sycl::image_channel_type channelType, cl::sycl::range<1> &r,
      void *pitch) {
    // 1D images cannot be constructed with a pitch
    return cl::sycl::image<1, AllocatorT>(static_cast<void *>(imageData),
                                          channelOrder, channelType, r);
  }
  static cl::sycl::image<1> create_with_pitch(
      void *imageData, cl::sycl::image_channel_order channelOrder,
      cl::sycl::image_channel_type channelType, cl::sycl::range<1> &r,
      void *pitch) {
    // 1D images cannot be constructed with a pitch
    return create_with_pitch_and_allocator<cl::sycl::image_allocator>(
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
  cl::sycl::image_channel_type type;
  unsigned int size;
  const char *str;
};

/**
 * @brief Map that stores the sizes and names (as strings) for all image
 *        channel types
 */
channel_type_size g_channelTypeSize[] = {
    {cl::sycl::image_channel_type::snorm_int8, 1, "snorm_int8"},
    {cl::sycl::image_channel_type::snorm_int16, 2, "snorm_int16"},
    {cl::sycl::image_channel_type::unorm_int8, 1, "unorm_int8"},
    {cl::sycl::image_channel_type::unorm_int16, 2, "unorm_int16"},
    {cl::sycl::image_channel_type::unorm_short_555, 2, "unorm_short_555"},
    {cl::sycl::image_channel_type::unorm_short_565, 2, "unorm_short_565"},
    {cl::sycl::image_channel_type::unorm_int_101010, 4, "unorm_int_101010"},
    {cl::sycl::image_channel_type::signed_int8, 1, "signed_int8"},
    {cl::sycl::image_channel_type::signed_int16, 2, "signed_int16"},
    {cl::sycl::image_channel_type::signed_int32, 4, "signed_int32"},
    {cl::sycl::image_channel_type::unsigned_int8, 1, "unsigned_int8"},
    {cl::sycl::image_channel_type::unsigned_int16, 2, "unsigned_int16"},
    {cl::sycl::image_channel_type::unsigned_int32, 4, "unsigned_int32"},
    {cl::sycl::image_channel_type::fp16, sizeof(cl::sycl::half), "fp16"},
    {cl::sycl::image_channel_type::fp32, sizeof(float), "fp32"}};

/**
 * @brief The total number of different image channel types
 */
static const int NUM_CHANNEL_TYPES =
    (sizeof(g_channelTypeSize) / sizeof(channel_type_size));

/**
 * @brief Retrieve the size of the image channel type
 */
unsigned int get_channel_type_size(cl::sycl::image_channel_type channelType) {
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
const char *get_channel_type_string(cl::sycl::image_channel_type channelType) {
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
  cl::sycl::image_channel_order order;
  unsigned int count;
  const char *str;
};

/**
 * @brief Map that stores the counts and names (as strings) for all image
 *        channel orders
 */
channel_order_count g_channelOrderCount[] = {
    // rgba and bgra are the mandated minimum supported
    {cl::sycl::image_channel_order::rgba, 4, "rgba"},
    {cl::sycl::image_channel_order::bgra, 4, "bgra"},

    {cl::sycl::image_channel_order::argb, 4, "argb"},

    {cl::sycl::image_channel_order::r, 1, "r"},
    {cl::sycl::image_channel_order::a, 1, "a"},

    {cl::sycl::image_channel_order::rg, 2, "rg"},
    {cl::sycl::image_channel_order::ra, 2, "ra"},
    {cl::sycl::image_channel_order::rx, 2, "rx"},
    {cl::sycl::image_channel_order::rgx, 3, "rgx"},

    {cl::sycl::image_channel_order::intensity, 1, "intensity"},
    {cl::sycl::image_channel_order::luminance, 1, "luminance"},

    // These two are special cases
    {cl::sycl::image_channel_order::rgb, 1, "rgb"},
    {cl::sycl::image_channel_order::rgbx, 1, "rgbx"}};

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
unsigned int get_channel_order_count(
    cl::sycl::image_channel_order channelOrder) {
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
const char *get_channel_order_string(
    cl::sycl::image_channel_order channelOrder) {
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
  cl::sycl::image_channel_order order;
  int numChannelTypes;
  cl::sycl::image_channel_type typeArray[NUM_CHANNEL_TYPES];
};

/**
 * @brief Retrieves a test set, which consist of an image channel order and all
 *        valid image channel types for that order.
 * @param order Image channel order to be tested.
 * @return Collection of channel order and all valid channel types for the order
 */
collection get_test_set_full(cl::sycl::image_channel_order order) {
  collection testSet;
  testSet.order = order;
  // Not all combinations are allowed
  switch (order) {
    case cl::sycl::image_channel_order::argb:
    case cl::sycl::image_channel_order::bgra:
      testSet.typeArray[0] = cl::sycl::image_channel_type::unorm_int8;
      testSet.typeArray[1] = cl::sycl::image_channel_type::snorm_int8;
      testSet.typeArray[2] = cl::sycl::image_channel_type::signed_int8;
      testSet.typeArray[3] = cl::sycl::image_channel_type::unsigned_int8;
      testSet.numChannelTypes = 4;
      break;
    case cl::sycl::image_channel_order::intensity:
    case cl::sycl::image_channel_order::luminance:
      testSet.typeArray[0] = cl::sycl::image_channel_type::unorm_int8;
      testSet.typeArray[1] = cl::sycl::image_channel_type::unorm_int16;
      testSet.typeArray[2] = cl::sycl::image_channel_type::snorm_int8;
      testSet.typeArray[3] = cl::sycl::image_channel_type::snorm_int16;
      testSet.typeArray[4] = cl::sycl::image_channel_type::fp16;
      testSet.typeArray[5] = cl::sycl::image_channel_type::fp32;
      testSet.numChannelTypes = 6;
      break;
    case cl::sycl::image_channel_order::rgb:
    case cl::sycl::image_channel_order::rgbx:
      testSet.typeArray[0] = cl::sycl::image_channel_type::unorm_short_555;
      testSet.typeArray[1] = cl::sycl::image_channel_type::unorm_short_565;
      testSet.typeArray[2] = cl::sycl::image_channel_type::unorm_int_101010;
      testSet.numChannelTypes = 3;
      break;
    default:
      // Most channel orders allow most channel types
      auto &itTestSet = testSet.numChannelTypes;
      itTestSet = 0;
      for (int itType = 0; itType < NUM_CHANNEL_TYPES; ++itType) {
        auto channelType = g_channelTypeSize[itType].type;
        switch (channelType) {
          case cl::sycl::image_channel_type::unorm_short_555:
          case cl::sycl::image_channel_type::unorm_short_565:
          case cl::sycl::image_channel_type::unorm_int_101010:
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
collection get_test_set_minimum(cl::sycl::image_channel_order order) {
  collection testSet;
  testSet.order = order;
  switch (order) {
    case cl::sycl::image_channel_order::rgba:
      testSet.typeArray[0] = cl::sycl::image_channel_type::unorm_int8;
      testSet.typeArray[1] = cl::sycl::image_channel_type::unorm_int16;
      testSet.typeArray[2] = cl::sycl::image_channel_type::signed_int8;
      testSet.typeArray[3] = cl::sycl::image_channel_type::signed_int16;
      testSet.typeArray[4] = cl::sycl::image_channel_type::signed_int32;
      testSet.typeArray[5] = cl::sycl::image_channel_type::unsigned_int8;
      testSet.typeArray[6] = cl::sycl::image_channel_type::unsigned_int16;
      testSet.typeArray[7] = cl::sycl::image_channel_type::unsigned_int32;
      testSet.typeArray[8] = cl::sycl::image_channel_type::fp16;
      testSet.typeArray[9] = cl::sycl::image_channel_type::fp32;
      testSet.numChannelTypes = 10;
      break;
    case cl::sycl::image_channel_order::bgra:
      testSet.typeArray[0] = cl::sycl::image_channel_type::unorm_int8;
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
cl::sycl::vector_class<cl::sycl::byte> get_image_host(
    unsigned int numElems, unsigned int channelCount,
    unsigned int channelTypeSize) {
  const auto sizePerChannelType = (dims * numElems * channelCount);
  const auto size = (sizePerChannelType * channelTypeSize);
  cl::sycl::vector_class<cl::sycl::byte> imageHost(size);
  for (unsigned int ii = 0; ii < sizePerChannelType; ii++) {
    for (unsigned int ij = 0; ij < channelTypeSize; ij++) {
      imageHost[ii * channelTypeSize + ij] = ij + 1;
    }
  }
  return imageHost;
}

}  // namespace

#endif  // SYCL_CTS_TEST_IMAGE_IMAGE_COMMON_H
