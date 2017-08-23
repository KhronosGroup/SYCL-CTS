/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

namespace {

using namespace sycl_cts;
using namespace cl::sycl;

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
  static bool compare_pitch(util::logger &log, image<dims> &img,
                            range<dims - 1> *pitch) {
    if (pitch == nullptr) {
      // Don't compare if pitch is null
      return true;
    }
    if ((!CHECK_VALUE_SCALAR(log, img.get_pitch()[0], pitch->get(0))) ||
        (!CHECK_VALUE_SCALAR(log, img.get_pitch()[1], pitch->get(1))) ||
        (!CHECK_VALUE_SCALAR(log, img.get_pitch()[2], pitch->get(2)))) {
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
  static bool compare_pitch(util::logger &log, image<dims> &imgA,
                            image<dims> &imgB) {
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
  static image<dims, AllocatorT> create_with_pitch_and_allocator(
      void *imageData, image_channel_order channelOrder,
      image_channel_type channelType, range<dims> &r, range<dims - 1> *pitch) {
    if (pitch == nullptr) {
      return image<dims, AllocatorT>(imageData, channelOrder, channelType, r);
    } else {
      return image<dims, AllocatorT>(imageData, channelOrder, channelType, r,
                                     *pitch);
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
  static image<dims> create_with_pitch(void *imageData,
                                       image_channel_order channelOrder,
                                       image_channel_type channelType,
                                       range<dims> &r, range<dims - 1> *pitch) {
    return create_with_pitch_and_allocator<cl::sycl::image_allocator>(
        imageData, channelOrder, channelType, r, pitch);
  }
};
/**
 * @brief Specialization for one dimension
 */
template <>
struct image_generic<1> {
  static bool compare_pitch(util::logger &log, image<1> &img, void *pitch) {
    // 1D images don't have a get_pitch() method
    return true;
  }
  static bool compare_pitch(util::logger &log, image<1> &imgA, image<1> &imgB) {
    // 1D images don't have a get_pitch() method
    return true;
  }
  template <class AllocatorT>
  static image<1, AllocatorT> create_with_pitch_and_allocator(
      void *imageData, image_channel_order channelOrder,
      image_channel_type channelType, range<1> &r, void *pitch) {
    // 1D images cannot be constructed with a pitch
    return image<1, AllocatorT>(static_cast<void *>(imageData), channelOrder,
                                channelType, r);
  }
  static image<1> create_with_pitch(void *imageData,
                                    image_channel_order channelOrder,
                                    image_channel_type channelType, range<1> &r,
                                    void *pitch) {
    // 1D images cannot be constructed with a pitch
    return create_with_pitch_and_allocator<cl::sycl::image_allocator>(
        imageData, channelOrder, channelType, r, pitch);
  }
};

/**
 * @brief Helps with retrieving the right access type for reading/writing
 *        an image
 * @tparam dims Number of image dimensions
 */
template <int dims>
struct image_access;
/**
 * @brief Specialization for one dimension
 */
template <>
struct image_access<1> {
  using int_type = cl::sycl::cl_int;
  using float_type = cl::sycl::cl_float;
  static int_type get_int(item<1> i) { return int_type(i.get(0)); }
  static float_type get_float(item<1> i) {
    return float_type(static_cast<float>(i.get(0)));
  }
};
/**
 * @brief Specialization for two dimensions
 */
template <>
struct image_access<2> {
  using int_type = cl::sycl::cl_int2;
  using float_type = cl::sycl::cl_float2;
  static int_type get_int(item<2> i) { return int_type(i.get(0), i.get(1)); }
  static float_type get_float(item<2> i) {
    return float_type(static_cast<float>(i.get(0)),
                      static_cast<float>(i.get(1)));
  }
};
/**
 * @brief Specialization for three dimensions
 */
template <>
struct image_access<3> {
  using int_type = cl::sycl::cl_int4;
  using float_type = cl::sycl::cl_float4;
  static int_type get_int(item<3> i) {
    return int_type(i.get(0), i.get(1), i.get(2), 0);
  }
  static float_type get_float(item<3> i) {
    return float_type(static_cast<float>(i.get(0)),
                      static_cast<float>(i.get(1)),
                      static_cast<float>(i.get(2)), .0f);
  }
};

/**
 * @brief Map entry type that specifies the size and name (as a string) for an
 *        image channel type
 */
struct channel_type_size {
  image_channel_type type;
  unsigned int size;
  const char *str;
};

/**
 * @brief Map that stores the sizes and names (as strings) for all image
 *        channel types
 */
channel_type_size g_channelTypeSize[] = {
    {image_channel_type::snorm_int8, 1, "snorm_int8"},
    {image_channel_type::snorm_int16, 2, "snorm_int16"},
    {image_channel_type::unorm_int8, 1, "unorm_int8"},
    {image_channel_type::unorm_int16, 2, "unorm_int16"},
    {image_channel_type::unorm_short_555, 2, "unorm_short_555"},
    {image_channel_type::unorm_short_565, 2, "unorm_short_565"},
    {image_channel_type::unorm_int_101010, 4, "unorm_int_101010"},
    {image_channel_type::signed_int8, 1, "signed_int8"},
    {image_channel_type::signed_int16, 2, "signed_int16"},
    {image_channel_type::signed_int32, 4, "signed_int32"},
    {image_channel_type::unsigned_int8, 1, "unsigned_int8"},
    {image_channel_type::unsigned_int16, 2, "unsigned_int16"},
    {image_channel_type::unsigned_int32, 4, "unsigned_int32"},
    {image_channel_type::fp16, sizeof(cl::sycl::half), "fp16"},
    {image_channel_type::fp32, sizeof(float), "fp32"}};

/**
 * @brief The total number of different image channel types
 */
static const int NUM_CHANNEL_TYPES =
    (sizeof(g_channelTypeSize) / sizeof(channel_type_size));

/**
 * @brief Retrieve the size of the image channel type
 */
unsigned int get_channel_type_size(image_channel_type channelType) {
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
const char *get_channel_type_string(image_channel_type channelType) {
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
  image_channel_order order;
  unsigned int count;
  const char *str;
};

/**
 * @brief Map that stores the counts and names (as strings) for all image
 *        channel orders
 */
channel_order_count g_channelOrderCount[] = {
    // rgba and bgra are the mandated minimum supported
    {image_channel_order::rgba, 4, "rgba"},
    {image_channel_order::bgra, 4, "bgra"},

    {image_channel_order::argb, 4, "argb"},

    {image_channel_order::r, 1, "r"},
    {image_channel_order::a, 1, "a"},

    {image_channel_order::rg, 2, "rg"},
    {image_channel_order::ra, 2, "ra"},
    {image_channel_order::rx, 2, "rx"},
    {image_channel_order::rgx, 3, "rgx"},

    {image_channel_order::intensity, 1, "intensity"},
    {image_channel_order::luminance, 1, "luminance"},

    // These two are special cases
    {image_channel_order::rgb, 1, "rgb"},
    {image_channel_order::rgbx, 1, "rgbx"}};

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
unsigned int get_channel_order_count(image_channel_order channelOrder) {
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
const char *get_channel_order_string(image_channel_order channelOrder) {
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
  image_channel_order order;
  int numChannelTypes;
  image_channel_type typeArray[NUM_CHANNEL_TYPES];
};

/**
 * @brief Retrieves a test set, which consist of an image channel order and all
 *        valid image channel types for that order.
 * @param order Image channel order to be tested.
 * @return Collection of channel order and all valid channel types for the order
 */
collection get_test_set_full(image_channel_order order) {
  collection testSet;
  testSet.order = order;
  // Not all combinations are allowed
  switch (order) {
    case image_channel_order::argb:
    case image_channel_order::bgra:
      testSet.typeArray[0] = image_channel_type::unorm_int8;
      testSet.typeArray[1] = image_channel_type::snorm_int8;
      testSet.typeArray[2] = image_channel_type::signed_int8;
      testSet.typeArray[3] = image_channel_type::unsigned_int8;
      testSet.numChannelTypes = 4;
      break;
    case image_channel_order::intensity:
    case image_channel_order::luminance:
      testSet.typeArray[0] = image_channel_type::unorm_int8;
      testSet.typeArray[1] = image_channel_type::unorm_int16;
      testSet.typeArray[2] = image_channel_type::snorm_int8;
      testSet.typeArray[3] = image_channel_type::snorm_int16;
      testSet.typeArray[4] = image_channel_type::fp16;
      testSet.typeArray[5] = image_channel_type::fp32;
      testSet.numChannelTypes = 6;
      break;
    case image_channel_order::rgb:
    case image_channel_order::rgbx:
      testSet.typeArray[0] = image_channel_type::unorm_short_555;
      testSet.typeArray[1] = image_channel_type::unorm_short_565;
      testSet.typeArray[2] = image_channel_type::unorm_int_101010;
      testSet.numChannelTypes = 3;
      break;
    default:
      // Most channel orders allow most channel types
      auto &itTestSet = testSet.numChannelTypes;
      itTestSet = 0;
      for (int itType = 0; itType < NUM_CHANNEL_TYPES; ++itType) {
        auto channelType = g_channelTypeSize[itType].type;
        switch (channelType) {
          case image_channel_type::unorm_short_555:
          case image_channel_type::unorm_short_565:
          case image_channel_type::unorm_int_101010:
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
collection get_test_set_minimum(image_channel_order order) {
  collection testSet;
  testSet.order = order;
  switch (order) {
    case image_channel_order::rgba:
      testSet.typeArray[0] = image_channel_type::unorm_int8;
      testSet.typeArray[1] = image_channel_type::unorm_int16;
      testSet.typeArray[2] = image_channel_type::signed_int8;
      testSet.typeArray[3] = image_channel_type::signed_int16;
      testSet.typeArray[4] = image_channel_type::signed_int32;
      testSet.typeArray[5] = image_channel_type::unsigned_int8;
      testSet.typeArray[6] = image_channel_type::unsigned_int16;
      testSet.typeArray[7] = image_channel_type::unsigned_int32;
      testSet.typeArray[8] = image_channel_type::fp16;
      testSet.typeArray[9] = image_channel_type::fp32;
      testSet.numChannelTypes = 10;
      break;
    case image_channel_order::bgra:
      testSet.typeArray[0] = image_channel_type::unorm_int8;
      testSet.numChannelTypes = 1;
      break;
    default:
      // No other combinations are required
      testSet.numChannelTypes = 0;
      break;
  }
  return testSet;
}

template <int numElems>
unique_ptr_class<char[]> get_image_host(unsigned int channelCount,
                                        unsigned int channelTypeSize) {
  unique_ptr_class<char[]> imageHost(
      new char[numElems * channelTypeSize * channelCount]);
  for (unsigned int ii = 0; ii < numElems * channelCount; ii++) {
    for (unsigned int ij = 0; ij < channelTypeSize; ij++) {
      imageHost[ii * channelTypeSize + ij] = ij + 1;
    }
  }
  return imageHost;
}

}  // namespace
