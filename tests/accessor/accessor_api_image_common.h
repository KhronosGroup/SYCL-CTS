/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_IMAGE_COMMON_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_IMAGE_COMMON_H

#include "../common/common.h"
#include "./../../util/accuracy.h"
#include "./../../util/math_helper.h"
#include "accessor_api_utility.h"
#include "accessor_api_common_all.h"

#include <array>
#include <numeric>
#include <sstream>
#include <type_traits>

namespace {

using namespace sycl_cts;
using namespace accessor_utility;

/**
 * @brief Determines the number of dimensions required for storing an image
 *        when given an accessor. Image array accessors require an image of
 *        (dims + 1) dimensions.
 * @tparam dims Number of accessor dimensions
 * @tparam target Access target of the accessor
 */
template <int dims, sycl::target target>
using image_dims = std::integral_constant<
    int,
    ((target == sycl::target::image_array) ? (dims + 1) : dims)>;

/**
 * @brief Alias to sycl::id using the proper number of dimensions.
 *        Required for image array accessors.
 * @tparam dims Number of accessor dimensions
 * @tparam target Access target of the accessor
 */
template <int dims, sycl::target target>
using image_id_t = sycl::id<image_dims<dims, target>::value>;

/**
 * @brief Alias to image_id_t targetting image array accessors
 * @tparam dims Number of accessor dimensions
 * @tparam target Access target of the accessor
 */
template <int dims>
using image_array_id_t =
    image_id_t<dims, sycl::target::image_array>;

/**
 * @brief Alias to sycl::range using the proper number of dimensions.
 *        Required for image array accessors.
 * @tparam dims Number of accessor dimensions
 * @tparam target Access target of the accessor
 */
template <int dims, sycl::target target>
using image_range_t = sycl::range<image_dims<dims, target>::value>;

/**
 * @brief Alias to image_range_t targetting image array accessors
 * @tparam dims Number of accessor dimensions
 * @tparam target Access target of the accessor
 */
template <int dims>
using image_array_range_t =
    image_range_t<dims, sycl::target::image_array>;

/**
 * @brief Namespace that defines CoordT tags
 */
namespace acc_coord_tag {
struct generic {};
struct use_float : generic {};
struct use_int : generic {};
struct use_normalized : use_float {};
struct use_normalized_lower : use_normalized {};
struct use_normalized_upper : use_normalized {};

/**
 * @brief Provide pixel tag from coordinate tag
 */
pixel_tag::lower get_pixel_tag(acc_coord_tag::use_normalized_lower) {
  return pixel_tag::lower{};
}
pixel_tag::upper get_pixel_tag(acc_coord_tag::use_normalized_upper) {
  return pixel_tag::upper{};
}
}

/**
 * @brief Namespace that defines image accessor data type tags
 */
namespace acc_data_tag {
struct generic {};
struct use_float : generic {};
struct use_int : generic {};

template <typename dataT>
struct get;
template <>
struct get<sycl::cl_int4> {
  using type = use_int;
};
template <>
struct get<sycl::cl_uint4> {
  using type = use_int;
};
template <>
struct get<sycl::cl_float4> {
  using type = use_float;
};
template <>
struct get<sycl::cl_half4> {
  using type = use_float;
};
}

/**
 * @brief Helper struct for retrieving the image access coordinates
 *        when testing an image array accessor
 * @tparam dims Number of accessor dimensions
 */
template <int dims>
struct image_array_coords {
  /**
   * @brief Retrieves the image access coordinates for an image array accessor
   * @param idx Work-item ID
   * @return Coordinates of dimension 1
   */
  template <typename ... rangeT>
  static auto get(acc_coord_tag::use_int, image_array_id_t<dims> idx,
                  rangeT ...)
      -> typename image_access<dims>::int_type {
    return image_access<dims>::get_int(
        sycl_cts::util::get_cts_object::id<dims>::get(idx));
  }
  /**
   * @brief Overload for cl_float type usage
   */
  template <typename ... rangeT>
  static auto get(acc_coord_tag::use_float, image_array_id_t<dims> idx,
                  rangeT...)
      -> typename image_access<dims>::float_type {
    return image_access<dims>::get_float(
        sycl_cts::util::get_cts_object::id<dims>::get(idx));
  }

  /**
   * @brief Overload for cl_float type usage with normalized coordinates
   */
  template <typename coordT,
            typename pixelT = decltype(acc_coord_tag::get_pixel_tag(coordT{}))>
  static auto get(coordT,
                  image_array_id_t<dims> idx,
                  image_array_range_t<dims> range)
      -> typename image_access<dims>::float_type {
    const auto resizedId =
        sycl_cts::util::get_cts_object::id<dims>::get(idx);
    const auto resizedRange =
        sycl_cts::util::get_cts_object::range<dims>::get(range);

    return image_access<dims>::get_normalized(pixelT{},
                                              resizedId, resizedRange);
  }
};

/**
 * @brief Constructs a range for testing
 * @tparam imageDims Number of image dimensions
 * @param count Number of elements to store in entire range
 * @param isImageArray True if creating a test range for an image array accessor
 * @return Range of same dimensions as the image
 */
template <int imageDims>
sycl::range<imageDims> make_test_range(size_t count,
                                           bool isImageArray = false);

/**
 * @brief Constructs a range for testing, specialization for 1D images
 * @param count Number of elements to store in entire range
 * @return 1D range
 */
template <>
sycl::range<1> make_test_range<1>(size_t count, bool /*isImageArray*/) {
  return {count};
}

/**
 * @brief Constructs a range for testing, specialization for 2D images
 * @param count Number of elements to store in entire range
 * @param isImageArray True if creating a test range for an image array accessor
 * @return 2D range
 */
template <>
sycl::range<2> make_test_range<2>(size_t count, bool isImageArray) {
  const auto dim1 = static_cast<size_t>(isImageArray ? 1 : 4);
  return {count / dim1, dim1};
}

/**
 * @brief Constructs a range for testing, specialization for 3D images
 * @param count Number of elements to store in entire range
 * @param isImageArray True if creating a test range for an image array accessor
 * @return 3D range
 */
template <>
sycl::range<3> make_test_range<3>(size_t count, bool isImageArray) {
  const auto dim2 = static_cast<size_t>(isImageArray ? 1 : 2);
  const size_t dim1 = 4;
  const auto dim0 = count / (dim1 * dim2);
  return {dim0, dim1, dim2};
}

/** unique dummy_functor per file
 */
template <typename T>
class dummy_accessor_api_image {};
template <typename T>
using dummy_functor = ::dummy_functor<dummy_accessor_api_image<T>>;

/** image format channel order and type
*/
template <typename T>
struct image_format_channel;

/** image format channel order and type (specialization for cl_int4)
*/
template <>
struct image_format_channel<sycl::cl_int4> {
  static constexpr sycl::image_channel_type type =
      sycl::image_channel_type::signed_int8;
  static constexpr sycl::image_channel_order order =
      sycl::image_channel_order::rgba;
  using storage_t = sycl::cl_char;
  static constexpr size_t elementSize = 4 * sizeof(storage_t);
};

/** image format channel order and type (specialization for cl_uint4)
*/
template <>
struct image_format_channel<sycl::cl_uint4> {
  static constexpr sycl::image_channel_type type =
      sycl::image_channel_type::unsigned_int8;
  static constexpr sycl::image_channel_order order =
      sycl::image_channel_order::rgba;
  using storage_t = sycl::cl_uchar;
  static constexpr size_t elementSize = 4 * sizeof(storage_t);
};

/** image format channel order and type (specialization for cl_float4)
*/
template <>
struct image_format_channel<sycl::cl_float4> {
  static constexpr sycl::image_channel_type type =
      sycl::image_channel_type::fp32;
  static constexpr sycl::image_channel_order order =
      sycl::image_channel_order::rgba;
  using storage_t = sycl::cl_float;
  static constexpr size_t elementSize = 4 * sizeof(storage_t);
};

/** image format channel order and type (specialization for cl_half4)
 */
template <>
struct image_format_channel<sycl::cl_half4> {
  static constexpr sycl::image_channel_type type =
      sycl::image_channel_type::fp16;
  static constexpr sycl::image_channel_order order =
      sycl::image_channel_order::rgba;
  using storage_t = sycl::cl_half;
  static constexpr size_t elementSize = 4 * sizeof(storage_t);
};

/** image border color based on test image format channel order
 */
template <typename T>
T image_border_color() {
  return T{0};
}

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. 1.0f by default.
*/
template <typename elementT>
struct use_normalization_coefficient : std::false_type {};

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. Specialization for sycl::cl_float4.
*/
template <>
struct use_normalization_coefficient<sycl::cl_float4> : std::true_type {};

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. Specialization for sycl::cl_half4.
*/
template <>
struct use_normalization_coefficient<sycl::cl_half4> : std::true_type {};

/**
 * @brief Retrieves the expected single image value when reading to
 *        or writing from an image accessor
 * @tparam T Type used in the image accessor
 * @return Some constant value of the underlying type of T
 */
template <typename T>
typename image_format_channel<T>::storage_t get_expected_image_value() {
  using storage_t = typename image_format_channel<T>::storage_t;
  if constexpr (use_normalization_coefficient<T>::value) {
    return static_cast<storage_t>(0.2f);
  } else {
    return static_cast<storage_t>(17);
  }
}

/**
 * @brief Retrieves the expected single image value when reading to
 *        or writing from an image accessor
 * @tparam T Type used in the image accessor
 * @return Some constant element of type T
 */
template <typename T>
T get_expected_image_elem() {
  return T{get_expected_image_value<T>()};
}

/**
 * @brief Converts actual image data to a container of bytes
 * @tparam T Type used in the image accessor
 * @tparam storage_t Underlying type of T, deduced
 * @param storageData Data to pass to the image
 * @param byteSize Size of image in bytes
 * @return Image data represented as bytes
 */
template <typename T,
          typename storage_t = typename image_format_channel<T>::storage_t>
std::vector<sycl::byte> convert_image_data_to_bytes(
    const std::vector<storage_t> &storageData, size_t byteSize) {
  using byte_t = sycl::byte;
  const auto byteDataPtr = reinterpret_cast<const byte_t *>(storageData.data());
  auto byteData =
      std::vector<byte_t>(byteDataPtr, byteDataPtr + byteSize);
  return byteData;
}

/**
 * @brief Converts a container of bytes to actual image data
 * @tparam T Type used in the image accessor
 * @tparam storage_t Underlying type of T, deduced
 * @param byteData Bytes used when constructing an image
 * @return Image data represented as actual values
 */
template <typename T,
          typename storage_t = typename image_format_channel<T>::storage_t>
std::vector<storage_t> convert_image_bytes_to_data(
    const std::vector<sycl::byte> &byteData) {
  const auto dataCount = (byteData.size() / sizeof(storage_t));
  const auto storageDataPtr =
      reinterpret_cast<const storage_t *>(byteData.data());
  auto storageData = std::vector<storage_t>(
      storageDataPtr, storageDataPtr + dataCount);
  return storageData;
}

/**
 * @brief Retrieves a container of bytes to be passed to the image constructor
 * @tparam T Type used in the image accessor
 * @param byteSize Size of image in bytes
 * @param initialize Whether to initialize the data or not.
 *        If true, the data is initialized using get_expected_image_value().
 *        If false, the data is set to zero.
 * @return Byte container ready to be passed to an image constructor
 */
template <typename T>
std::vector<sycl::byte> get_image_input_data(
    size_t byteSize, bool initialize = true) {
  using storage_t = typename image_format_channel<T>::storage_t;
  const auto dataCount = (byteSize / sizeof(storage_t));
  auto singleElem = storage_t{0};
  if (initialize) {
    singleElem = get_expected_image_value<T>();
  }
  const auto data = std::vector<storage_t>(dataCount, singleElem);
  return convert_image_data_to_bytes<T>(data, byteSize);
}

template <typename T, int dims, sycl::target target,
          sycl::access_mode mode>
T read_image_acc(const sycl::accessor<T, dims, mode, target> &acc,
                 sycl::id<dims> idx) {
  return acc.read(image_access<dims>::get_int(idx));
}

template <typename T, int dims, sycl::access_mode mode>
T read_image_acc(const sycl::accessor<T, dims, mode,
                                    sycl::target::image_array> &acc,
                 image_array_id_t<dims> idx) {
  // Verify __image_array_slice__ read
  using coordT = acc_coord_tag::use_int;
  const auto coords = image_array_coords<dims>::get(coordT{}, idx);
  return acc[idx[dims]].read(coords);
}

template <typename T, int dims, sycl::target target,
          sycl::access_mode mode, typename coordT>
T read_image_acc_sampled(const sycl::accessor<T, dims, mode, target> &acc,
                         const sycl::sampler& smpl,
                         sycl::id<dims> idx,
                         sycl::range<dims> range,
                         const coordT& coordTag) {
  if constexpr (std::is_same_v<coordT, acc_coord_tag::use_int>) {
    // Verify read using integer unnormalized coordinates
    return acc.read(image_access<dims>::get_int(idx), smpl);
  } else if constexpr (std::is_same_v<coordT, acc_coord_tag::use_float>) {
    // Verify read using floating point unnormalized coordinates
    return acc.read(image_access<dims>::get_float(idx), smpl);
  } else {
    // Verify read using normalized coordinates
    const auto pixelTag = acc_coord_tag::get_pixel_tag(coordTag);
    const auto& coords = image_access<dims>::get_normalized(pixelTag, idx, range);
    return acc.read(coords, smpl);
  }
}
template <typename T, int dims, sycl::access_mode mode, typename coordT>
T read_image_acc_sampled(const sycl::accessor<T, dims, mode,
                                    sycl::target::image_array> &acc,
                         sycl::sampler smpl,
                         image_array_id_t<dims> idx,
                         image_array_range_t<dims> range,
                         const coordT& coordTag) {
  // Verify __image_array_slice__ read
  const auto coords = image_array_coords<dims>::get(coordTag, idx, range);
  return acc[idx[dims]].read(coords, smpl);
}

template <typename T, int dims, sycl::target target,
          sycl::access_mode mode>
void write_image_acc(const sycl::accessor<T, dims, mode, target> &acc,
                     sycl::id<dims> idx, T value) {
  const auto coords = image_access<dims>::get_int(idx);
  acc.write(coords, value);
}

template <typename T, int dims, sycl::access_mode mode>
void write_image_acc(
    const sycl::accessor<T, dims, mode, sycl::target::image_array>
        &acc,
    image_array_id_t<dims> idx, T value) {
  // Verify __image_array_slice__ write
  using coordT = acc_coord_tag::use_int;
  const auto coords = image_array_coords<dims>::get(coordT{}, idx);
  acc[idx[dims]].write(coords, value);
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

/**
 *  @brief Provide sampler instance to share with device
 */
struct image_accessor_sampler {
  sycl::sampler instance;
  sycl::coordinate_normalization_mode coordinate_normalization_mode;
  sycl::addressing_mode addressing_mode;
  sycl::filtering_mode filtering_mode;

  image_accessor_sampler(sycl::coordinate_normalization_mode normalization,
                         sycl::addressing_mode addressing,
                         sycl::filtering_mode filtering) :
    instance(normalization, addressing, filtering),
    coordinate_normalization_mode(normalization),
    addressing_mode(addressing),
    filtering_mode(filtering) {}

  static inline bool supports_out_of_range() {
    return false;
  }
  static inline bool supports_out_of_range(const image_accessor_sampler& inst) {
    return inst.addressing_mode != sycl::addressing_mode::none;
  }
};

/**
 *  @brief Provide samplers to use for tests
 */
class image_accessor_samplers {
  using normalization_t = sycl::coordinate_normalization_mode;
  using addressing_t = sycl::addressing_mode;
  using filtering_t = sycl::filtering_mode;

  std::string get_addressing_desc(addressing_t value) const {
    std::string result;

    switch (value) {
      case addressing_t::none:
        result = "none";
        break;
      case addressing_t::clamp:
        result = "clamp";
        break;
      case addressing_t::clamp_to_edge:
        result = "clamp_to_edge";
        break;
      case addressing_t::repeat:
        result = "repeat";
        break;
      case addressing_t::mirrored_repeat:
        result = "mirrored_repeat";
        break;
      default:
        result = "unknown";
    }
    return result;
  }

  std::string get_filtering_desc(filtering_t value) const {
    std::string result;

    switch (value) {
      case filtering_t::nearest:
        result = "nearest";
        break;
      case filtering_t::linear:
        result = "linear";
        break;
      default:
        result = "unknown";
    }
    return result;
  }

  std::string get_normalization_desc(normalization_t value) const {
    std::string result;

    switch (value) {
      case normalization_t::unnormalized:
        result = "unnormalized";
        break;
      case normalization_t::normalized:
        result = "normalized";
        break;
      default:
        result = "unknown";
    }
    return result;
  }

  using storage_t = std::array<image_accessor_sampler, 16>;
  storage_t m_samplers;

 public:
  image_accessor_samplers() : m_samplers {
    image_accessor_sampler(normalization_t::unnormalized,
                           addressing_t::none,
                           filtering_t::nearest),
    image_accessor_sampler(normalization_t::unnormalized,
                           addressing_t::clamp,
                           filtering_t::nearest),
    image_accessor_sampler(normalization_t::unnormalized,
                           addressing_t::clamp_to_edge,
                           filtering_t::nearest),

    image_accessor_sampler(normalization_t::unnormalized,
                           addressing_t::none,
                           filtering_t::linear),
    image_accessor_sampler(normalization_t::unnormalized,
                           addressing_t::clamp,
                           filtering_t::linear),
    image_accessor_sampler(normalization_t::unnormalized,
                           addressing_t::clamp_to_edge,
                           filtering_t::linear),

    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::none,
                           filtering_t::nearest),
    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::clamp,
                           filtering_t::nearest),
    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::clamp_to_edge,
                           filtering_t::nearest),
    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::repeat,
                           filtering_t::nearest),
    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::mirrored_repeat,
                           filtering_t::nearest),

    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::none,
                           filtering_t::linear),
    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::clamp,
                           filtering_t::linear),
    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::clamp_to_edge,
                           filtering_t::linear),
    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::repeat,
                           filtering_t::linear),
    image_accessor_sampler(normalization_t::normalized,
                           addressing_t::mirrored_repeat,
                           filtering_t::linear)
  } {}

  const storage_t& get() const {
    return m_samplers;
  }

  std::string get_description(const image_accessor_sampler& sampler) const {
    std::string desc("sampler (");
    desc += "normalization:" + get_normalization_desc(
        sampler.coordinate_normalization_mode) + "; ";
    desc += "addressing:" + get_addressing_desc(
        sampler.addressing_mode) + "; ";
    desc += "filtering:" + get_filtering_desc(
        sampler.filtering_mode) + ")";
    return desc;
  }
};

/**
 *  @brief Storage for single failed index
 */
template <typename T>
struct image_accessor_failure_item {
  bool triggered;
  T value;
  T expected;

  image_accessor_failure_item() :
    triggered(false),
    value(0),
    expected(0) {}
};
/**
 *  @brief Storage for all failed indexes
 */
template <typename T>
class image_accessor_failure_storage {
  std::vector<image_accessor_failure_item<T>> m_data;

  template <typename dataT>
  struct data_type;
  template <>
  struct data_type<float> {
    using base = std::uint32_t;
  };
  template <>
  struct data_type<double> {
    using base = std::uint64_t;
  };
  template <>
  struct data_type<sycl::half> {
    using base = std::uint16_t;
  };

  template<typename dataT>
  struct printer{
    template <typename valT = dataT>
    auto operator()(std::stringstream& stream, const valT& value)
        -> typename std::enable_if<!is_cl_float_type<valT>::value, void>::type {
      stream << value;
    }
    template <typename valT = dataT>
    auto operator()(std::stringstream& stream, const valT& value)
        -> typename std::enable_if<is_cl_float_type<valT>::value, void>::type {
      const auto representation =
          reinterpret_cast<const typename data_type<valT>::base&>(value);
      stream << value << " 0x" << std::hex << representation;
    }
  };
  template<typename dataT, int dataDims>
  struct printer<sycl::vec<dataT, dataDims>> {
    void operator()(std::stringstream& stream,
                           const sycl::vec<dataT, dataDims>& value) {
      printer<dataT> valuePrinter;

      stream << "(";
      for (int i = 0; i < dataDims-1; ++i) {
        valuePrinter(stream, value[i]);
        stream << ", ";
      }
      valuePrinter(stream, value[dataDims-1]);
      stream << ")";
    }
  };
  template<int rangeDims>
  void printIndex(std::stringstream& stream, size_t linearIndex,
                  const sycl::range<rangeDims>& imageRange,
                  const sycl::range<rangeDims>& verificationRange);
  template<>
  void printIndex<1>(std::stringstream& stream, size_t linearIndex,
                  const sycl::range<1>& imageRange,
                  const sycl::range<1>&) {
    stream << linearIndex << " [" <<
              linearIndex << ":" << imageRange[0] << "]";
  }
  template<>
  void printIndex<2>(std::stringstream& stream, size_t linearIndex,
                  const sycl::range<2>& imageRange,
                  const sycl::range<2>& verificationRange) {
    const auto id1 = linearIndex % verificationRange[1];
    const auto id0 = linearIndex / verificationRange[1];
    stream << linearIndex << " [" <<
              id0 << ":" << imageRange[0] << ", " <<
              id1 << ":" << imageRange[1] << "]";
  }
  template<>
  void printIndex<3>(std::stringstream& stream, size_t linearIndex,
                  const sycl::range<3>& imageRange,
                  const sycl::range<3>& verificationRange) {
    const auto id2 = linearIndex % verificationRange[2];
    const auto idx = linearIndex / verificationRange[2];
    const auto id1 = idx % verificationRange[1];
    const auto id0 = idx / verificationRange[1];
    stream << linearIndex << " [" <<
              id0 << ":" << imageRange[0] << ", " <<
              id1 << ":" << imageRange[1] << ", " <<
              id2 << ":" << imageRange[2] << "]";
  }

 public:
  using item_t = image_accessor_failure_item<T>;
  using buffer_t = sycl::buffer<item_t, 1>;

  image_accessor_failure_storage(size_t size) :
    m_data(size) {}

  image_accessor_failure_item<T>* data() {
    return m_data.data();
  }

  size_t size() const {
    return m_data.size();
  }

  template <int rangeDims>
  void dump(sycl_cts::util::logger& log,
            const sycl::range<rangeDims>& imageRange,
            const sycl::range<rangeDims>& verificationRange) {
    std::stringstream stream;
    printer<T> valuePrinter;

    log.note("Failed elements list:");
    stream.precision(64);
    for (size_t i = 0; i < m_data.size(); ++i) {
      const auto& item = m_data[i];
      if (item.triggered) {
        stream << "#";
        printIndex(stream, i, imageRange, verificationRange);
        stream << ": expected ";
        valuePrinter(stream, item.expected);
        stream << "; retrieved ";
        valuePrinter(stream, item.value);

        const auto message = stream.str();
        log.note(message.c_str());
        stream.str("");
      }
    }
  }
};

/** tests image accessors reads
*/
template <typename T, int dim, sycl::access_mode mode,
          sycl::target target, sycl::target errorTarget>
class image_accessor_api_r {
  using acc_t = sycl::accessor<T, dim, mode, target>;
  using error_acc_t = sycl::accessor<int, 1, errorMode, errorTarget>;
  using failure_acc_t = sycl::accessor<image_accessor_failure_item<T>, 1,
                                           errorMode, errorTarget>;
  acc_t m_acc;
  error_acc_t m_errorAccessor;
  failure_acc_t m_failureAccessor;

  image_range_t<dim, target> m_range;

 public:
  image_accessor_api_r(image_range_t<dim, target>,
                       image_range_t<dim, target> verificationRange,
                       acc_t acc, error_acc_t errorAccessor,
                       failure_acc_t failureAccessor)
      : m_acc(acc),
        m_errorAccessor(errorAccessor),
        m_failureAccessor(failureAccessor),
        m_range(verificationRange) {}

  void operator()(image_id_t<dim, target> idx) const {
    const auto expected = get_expected_image_elem<T>();

    /** check coordinates read syntax
     */
    T elem = read_image_acc(m_acc, idx);

    if (!check_elems_equal(elem, expected)) {
      m_errorAccessor[0] = 1;

      const size_t failureIndex = compute_linear_id(idx, m_range);
      auto& failureItem = m_failureAccessor[failureIndex];
      failureItem.triggered = true;
      failureItem.expected = expected;
      failureItem.value = elem;
    }
  }
};

/** tests image accessors sampled reads
*/
template <typename T, int dim, sycl::access_mode mode,
          sycl::target target, sycl::target errorTarget>
class image_accessor_api_sampled_r {
  using acc_t = sycl::accessor<T, dim, mode, target>;
  using error_acc_t = sycl::accessor<int, 1, errorMode, errorTarget>;
  using failure_acc_t = sycl::accessor<image_accessor_failure_item<T>, 1,
                                           errorMode, errorTarget>;
  acc_t m_acc;
  error_acc_t m_errorAccessor;
  failure_acc_t m_failureAccessor;

  image_accessor_sampler m_sampler;

  image_range_t<dim, target> m_range;
  image_range_t<dim, target> m_verificationRange;

 private:
  template <int idDims>
  bool in_scope(sycl::id<idDims>& idx,
                const sycl::range<idDims>& range,
                const sycl::id<idDims>& delta) const;
  template <>
  bool in_scope<1>(sycl::id<1>& idx, const sycl::range<1>& range,
                   const sycl::id<1>& delta) const {
    // We cannot move sycl::id to the negative values for linear filtering,
    // so we move range instead
    return (idx[0] >= delta[0]) &&
           (idx[0] < (range[0] + delta[0]));
  }
  template <>
  bool in_scope<2>(sycl::id<2>& idx, const sycl::range<2>& range,
                   const sycl::id<2>& delta) const {
    return (idx[0] >= delta[0]) &&
           (idx[1] >= delta[1]) &&
           (idx[0] < (range[0] + delta[0])) &&
           (idx[1] < (range[1] + delta[1]));
  }
  template <>
  bool in_scope<3>(sycl::id<3>& idx, const sycl::range<3>& range,
                   const sycl::id<3>& delta) const {
    return (idx[0] >= delta[0]) &&
           (idx[1] >= delta[1]) &&
           (idx[2] >= delta[2]) &&
           (idx[0] < (range[0] + delta[0])) &&
           (idx[1] < (range[1] + delta[1])) &&
           (idx[2] < (range[2] + delta[2]));
  }

  T get_expected_value_texel(image_id_t<dim, target> idx,
                             image_id_t<dim, target> delta) const {
    /**
     *  For nearest filtering mode the only case of border color usage is clamp
     *    addressing mode.
     *  For linear filtering mode with none, clamp and clamp_to_edge addressing
     *    modes OpenCL 1.2 spec uses address_mode() definition. So Tij, Tijk can
     *    refer to the location outside of image for none and clamp mode only.
     *  For repeat and mirror_repeat addressing mode OpenCL 1.2 spec mentiones
     *    no border color usage at all.
     *  Conformance tests will not use out of scope texels for nearest filtering
     *    mode with none addressing mode, as it is against SYCL spec. Therefore
     *    we can use both none and clamp modes for border color usage triggering
     *    for any filtering mode.
     */
    auto expected = get_expected_image_elem<T>();
    const bool useBorderColor =
        (m_sampler.addressing_mode == sycl::addressing_mode::clamp) ||
        (m_sampler.addressing_mode == sycl::addressing_mode::none);

    if (useBorderColor && !in_scope(idx, m_range, delta)){
      expected = image_border_color<T>();
    }
    return expected;
  }

  template <int idDims>
  T get_expected_value_nearest(sycl::id<idDims> idx) const {
    const auto delta = sycl_cts::util::get_cts_object::id<idDims>::get(0,0,0);

    return get_expected_value_texel(idx, delta);
  }

  template <int idDims>
  T get_expected_value_linear(sycl::id<idDims> idx) const;
  template <>
  T get_expected_value_linear<1>(sycl::id<1> idx) const {
    /**
     *  For none, clamp, clamp_to_edge and repeat addressing modes OpenCL 1.2
     *    spec does not mention 1D images usage with linear filtering.
     *  For mirror_repeat addressing mode OpenCL 1.2 spec states the following
     *    equation for 1D images: "T = (1 – a) * Ti0 + a * Ti1"
     *  There is no reason to use different logic for different addressing
     *    modes with 1D images, so uniform check provided for any addressing
     *    mode.
     *  Also we are using the fact exact coordinates are provided originally for
     *    any normalization mode, so we can simplify weighted equations.
     */
    const auto v0 = get_expected_value_texel(idx, sycl::id<1>(0));
    const auto v1 = get_expected_value_texel(idx, sycl::id<1>(1));
    return (v0 + v1) / 2;
  }
  template <>
  T get_expected_value_linear<2>(sycl::id<2> idx) const {
    const auto v00 = get_expected_value_texel(idx, sycl::id<2>(0,0));
    const auto v01 = get_expected_value_texel(idx, sycl::id<2>(0,1));
    const auto v10 = get_expected_value_texel(idx, sycl::id<2>(1,0));
    const auto v11 = get_expected_value_texel(idx, sycl::id<2>(1,1));
    return (v00 + v01 + v10 + v11) / 4;
  }
  template <>
  T get_expected_value_linear<3>(sycl::id<3> idx) const {
    const auto v000 = get_expected_value_texel(idx, sycl::id<3>(0,0,0));
    const auto v001 = get_expected_value_texel(idx, sycl::id<3>(0,0,1));
    const auto v010 = get_expected_value_texel(idx, sycl::id<3>(0,1,0));
    const auto v011 = get_expected_value_texel(idx, sycl::id<3>(0,1,1));
    const auto v100 = get_expected_value_texel(idx, sycl::id<3>(1,0,0));
    const auto v101 = get_expected_value_texel(idx, sycl::id<3>(1,0,1));
    const auto v110 = get_expected_value_texel(idx, sycl::id<3>(1,1,0));
    const auto v111 = get_expected_value_texel(idx, sycl::id<3>(1,1,1));
    return (v000 + v001 + v010 + v011 + v100 + v101 + v110 + v111) / 8;
  }

  template <typename coordTag>
  T get_expected_value(image_id_t<dim, target> idx) const {
    const bool useLinear =
        m_sampler.filtering_mode == sycl::filtering_mode::linear;

    if (!useLinear) {
      return get_expected_value_nearest(idx);
    }
    constexpr bool useUpper =
        std::is_same_v<coordTag, acc_coord_tag::use_normalized_upper>;
    if constexpr (!useUpper) {
      // Use simplified equation for lower coordinate values
      return get_expected_value_linear(idx);
    } else {
      /** Upper value is exactly 1 ULP lower than the lower value for the next
       *  coordinate. We can ignore this difference because:
       *  - there is no actual precision requirements defined for linear
       *    filtration mode in OpenCL spec.
       *  - we have data values pre-defined for floating type to be relatively
       *    not too big, so error propagation is relatively small
       *
       *  Currently we have the coordinate values
       *  - "u" as the lower one and
       *  - "u + 1 - 1ULP" as the upper one
       *  for floating-point coordinates. Because it would be valuable to verify
       *  also values
       *  - "u - 0.5" and
       *  - "u + 0.5 - 1 ULP"
       *  as the border values for texel selection according to the OpenCL spec:
       *    i0 = address_mode((int)floor(u - 0.5))
       *    j0 = address_mode((int)floor(v - 0.5))
       *    k0 = address_mode((int)floor(w - 0.5))
       *    i1 = address_mode((int)floor(u - 0.5) + 1)
       *    j1 = address_mode((int)floor(v - 0.5) + 1)
       *    k1 = address_mode((int)floor(w - 0.5) + 1)
       *  we may need to provide an exact reference values someday.
       *
       *  During future implementation of such reference functions we may gain
       *  accuracy from using Priest's compensated summation to avoid possible
       *  catastrophic cancellation.
       *  See
       *    Douglas M. Priest. "On Properties of Floating Point Arithmetics:
       *    Numerical Stability and the Cost of Accurate Computations."
       *    PhD thesis, Mathematics Department, University of California,
       *    Berkeley, CA, USA, November 1992. 126 pp.
       *    ftp://ftp.icsi.berkeley.edu/pub/theory/priest-thesis.ps.Z
       *  for details
       */
      return get_expected_value_linear(next_idx(idx));
    }
  }

  template <int dims>
  sycl::id<dims> next_idx(sycl::id<dims> idx) const {
    if constexpr (target == sycl::target::image_array) {
      auto res = idx;
      for (int i = 0; i + 1 < dims; i++) {
        res[i]++;
      }
      return res;
    }
    return idx + 1;
  }


  /**
   *  @brief Error for floating point image data
   */
  template <typename dataT>
  dataT get_ulp_error(acc_data_tag::use_int, dataT) const {
    return 0;
  }
  template <typename dataT>
  dataT get_ulp_error(acc_data_tag::use_float, dataT x) const {
    const bool isNearest =
        m_sampler.filtering_mode == sycl::filtering_mode::nearest;
    const bool isNormalized =
        m_sampler.coordinate_normalization_mode ==
            sycl::coordinate_normalization_mode::normalized;
    const auto texelNumber = 1 << dim;

    if (isNearest && !isNormalized)
      return .0f;
    /**
     *  The relative error or precision is not defined by OpenCL specification
     */
    if (isNearest && isNormalized) {
      /**
       *  The relative error or precision is not defined by OpenCL specification
       *    for this case.
       *  These tests have image dimensions as power of two, so no loss of
       *    precision due coordinate normalization is expected
       */
      return texelNumber * get_ulp_sycl(x);
    }
    /**
     *  The relative error or precision is not defined by OpenCL specification
     *    for case of linear filtration mode.
     */
    return 8192.0f * texelNumber * get_ulp_sycl(x);
  }

  /**
   *  @brief Error for integer image data
   */
  template <typename dataT>
  dataT get_abs_error(acc_data_tag::use_float, dataT) const {
    return 0;
  }
  template <typename dataT>
  dataT get_abs_error(acc_data_tag::use_int, dataT) const {
    const bool isNearest =
        m_sampler.filtering_mode == sycl::filtering_mode::nearest;
    /**
     *  These tests have image dimensions as power of two, so no loss of
     *    precision due coordinate normalization is expected
     */
    if (isNearest)
      return 0;
    /**
     *  The relative error or precision is not defined by OpenCL specification
     *    for case of linear filtration mode.
     */
    return 1;
  }

  bool check_texel_value(const T& value, const T& expected) const {
    bool result = true;

    using dataTypeTag = typename acc_data_tag::get<T>::type;
    const dataTypeTag tag;

    for (int i=0; i<4; ++i) {
      const auto ulpError = get_ulp_error(tag, expected[i]);
      const auto absError = get_abs_error(tag, expected[i]);
      const auto error = ulpError + absError;
      result &= value[i] <= expected[i] + error;
      result &= value[i] + error >= expected[i];
    }
    return result;
  }

  template <typename coordT, int idDims>
  bool check_read(sycl::id<idDims> idx) const {
    const T expected = get_expected_value<coordT>(idx);
    const T elem =
        read_image_acc_sampled(m_acc, m_sampler.instance, idx, m_range,
                               coordT{});

    const bool succeed = check_texel_value(elem, expected);
    if (!succeed) {
      m_errorAccessor[0] = 1;

      const size_t failureIndex = compute_linear_id(idx, m_verificationRange);
      auto& failureItem = m_failureAccessor[failureIndex];
      failureItem.triggered = true;
      failureItem.expected = expected;
      failureItem.value = elem;
    }
    return succeed;
  }

 public:
  image_accessor_api_sampled_r(image_range_t<dim, target> imageRange,
                               image_range_t<dim, target> verificationRange,
                               acc_t acc, error_acc_t errorAccessor,
                               failure_acc_t failureAccessor,
                               image_accessor_sampler sampler)
      : m_acc(acc),
        m_errorAccessor(errorAccessor),
        m_failureAccessor(failureAccessor),
        m_sampler(sampler),
        m_range(imageRange),
        m_verificationRange(verificationRange) {}

  void operator()(image_id_t<dim, target> idx) const {
    /** check coordinates with sampler read syntax
     */
    const bool useNormalized =
        m_sampler.coordinate_normalization_mode ==
            sycl::coordinate_normalization_mode::normalized;
    if (useNormalized) {
      const bool worksForLower =
        check_read<acc_coord_tag::use_normalized_lower>(idx);
      if (worksForLower)
        check_read<acc_coord_tag::use_normalized_upper>(idx);
    } else {
      const bool worksForInteger =
        check_read<acc_coord_tag::use_int>(idx);
      if (worksForInteger)
        check_read<acc_coord_tag::use_float>(idx);
    }
  }
};

/** tests image accessors writes
*/
template <typename T, int dim, sycl::access_mode mode,
          sycl::target target>
class image_accessor_api_w {
  using acc_t = sycl::accessor<T, dim, mode, target>;

  acc_t m_accCoordsSyntax;
  image_range_t<dim, target> m_range;
  size_t size;

 public:
  image_accessor_api_w(size_t size_, acc_t accCoordsSyntax,
                       image_range_t<dim, target> rng)
      : m_accCoordsSyntax(accCoordsSyntax), m_range(rng), size(size_) {}

  void operator()(image_id_t<dim, target> idx) const {
    const auto elem = get_expected_image_elem<T>();

    /** check coords write syntax
    */
    write_image_acc(m_accCoordsSyntax, idx, elem);
  }
};

/** tests image accessors methods
*/
template <typename T, int dims, sycl::access_mode mode,
          sycl::target target>
class check_image_accessor_api_methods {
 public:
  static constexpr auto isImageArray =
      (target == sycl::target::image_array);
  using image_t = sycl::image<(isImageArray ? (dims + 1) : dims)>;

  size_t count;
  size_t size;

  void operator()(util::logger &log, sycl::queue &queue,
                  image_range_t<dims, target> range,
                  const std::string& typeName) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target>("check_image_accessor_api_methods",
                                        typeName, log);
#endif  // VERBOSE_LOG

    auto data = get_image_input_data<T>(size);
    auto image = image_t(data.data(), image_format_channel<T>::order,
                         image_format_channel<T>::type, range);
    check_methods_helper(log, queue, image, range, typeName,
                         acc_type_tag::get<target, acc_placeholder::image>());
  }

 private:
 template <typename expectedT, typename returnT>
  void check_acc_return_type(sycl_cts::util::logger& log, returnT returnVal,
                            const std::string& functionName,
                            const std::string& typeName) const {
    accessor_utility::check_acc_return_type<
        expectedT, T, dims, mode, target>(
            log, returnVal, functionName, typeName);
  }

  template <typename acc_t>
  void check_methods(util::logger &log, const acc_t &accessor,
                     image_range_t<dims, target> range,
                     const std::string& typeName) const {
    {
      // check size() method
      auto accessorCount = accessor.size();
      check_acc_return_type<size_t>(
          log, accessor.size(), "size", typeName);
      if (accessorCount != count) {
        fail_for_accessor<T, dims, mode, target>(log, typeName,
            "accessor does not return the correct count");
      }
    }
    {
      // check get_count() method
      // TODO: mark this check as testing deprecated functionality
      auto accessorCount = accessor.get_count();
      check_acc_return_type<size_t>(log, accessor.get_count(), "get_count",
                                    typeName);
      if (accessorCount != count) {
        fail_for_accessor<T, dims, mode, target>(
            log, typeName, "accessor does not return the correct count");
      }
    }
    {
      // check get_range() method
      auto accessorRange = accessor.get_range();
#ifdef VERBOSE_LOG
      log.note("Checking get_range");
#endif  // VERBOSE_LOG
      check_acc_return_type<image_range_t<dims, target>>(
          log, accessor.get_range(), "get_range()", typeName);
      if (accessorRange != range) {
        fail_for_accessor<T, dims, mode, target>(log, typeName,
            "accessor does not return the correct range");
      }
    }
  }

  /**
   * @brief Checks member functions of a host image accessor
   * @param log The logger object
   * @param image SYCL image to request access to
   */
  void check_methods_helper(util::logger &log, sycl::queue & /*queue*/,
                            image_t &image, image_range_t<dims, target> range,
                            const std::string& typeName,
                            acc_type_tag::host) const {
    auto acc =
        make_accessor<T, dims, mode, target, acc_placeholder::image>(image);
    check_methods(log, acc, range, typeName);
  }

  /**
   * @brief Checks member functions of an image accessor
   * @param log The logger object
   * @param queue SYCL queue where a kernel will be executed
   * @param image SYCL image to request access to
   */
  void check_methods_helper(util::logger &log, sycl::queue &queue,
                            image_t &image, image_range_t<dims, target> range,
                            const std::string& typeName,
                            acc_type_tag::generic) const {
    queue.submit([&](sycl::handler &handler) {
      auto acc =
          make_accessor<T, dims, mode, target, acc_placeholder::image>(
              image, handler);
      check_methods(log, acc, range, typeName);
      handler.single_task(dummy_functor<T>());
    });
  }
};

/** tests image accessors reads
*/
template <typename T, int dims, sycl::access_mode mode,
          sycl::target target>
class check_image_accessor_api_reads {

  template <sycl::target errorTarget>
  using read_verifier_t =
      image_accessor_api_r<T, dims, mode, target, errorTarget>;

  template <sycl::target errorTarget>
  using sampled_read_verifier_t =
      image_accessor_api_sampled_r<T, dims, mode, target, errorTarget>;

  using failure_storage_t = image_accessor_failure_storage<T>;
  using failure_buffer_t = typename failure_storage_t::buffer_t;
  using failure_item_t = typename failure_storage_t::item_t;

  static constexpr bool supportsLinearFilering =
      is_cl_float_type<typename T::element_type>::value;

 public:
  static constexpr auto isImageArray =
      (target == sycl::target::image_array);
  using image_t = sycl::image<(isImageArray ? (dims + 1) : dims)>;

  size_t count;
  size_t size;

  void operator()(util::logger &log, sycl::queue &queue,
                  image_range_t<dims, target> range,
                  const std::string& typeName) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target>("check_image_accessor_api_reads",
                                        typeName, log);
#endif  // VERBOSE_LOG

    const auto tag = acc_type_tag::get<target, acc_placeholder::image>();
    image_accessor_samplers samplers;

    /** Verify direct read method
     */
    {
      auto data = get_image_input_data<T>(size);
      auto errors = get_error_data(1);

      failure_storage_t failures(size);
      {
        image_t image(data.data(),
                      image_format_channel<T>::order,
                      image_format_channel<T>::type, range);
        error_buffer_t errorBuffer(errors.data(),
                                   sycl::range<1>(errors.size()));
        failure_buffer_t failureBuffer(failures.data(),
                                   sycl::range<1>(failures.size()));

        check_command_group_reads<read_verifier_t>(
            tag, queue, range, image, errorBuffer, failureBuffer);
      }
      if (errors[0] != 0) {
        fail_for_accessor<T, dims, mode, target>(log, typeName,
            "read(coords) did not read from the correct index");
        failures.dump(log, range, get_verification_range(range));
      }
    }
    /** Verify sampled read method for:
     *   - all samplers for floating point data
     *   - all samplers except samplers with linear filtering for integer data
     *  According to the OpenCL backend spec:
     *   The read_image{i|ui} calls support a nearest filter only. The
     *   filter_mode specified in sampler must be set to CLK_FILTER_NEAREST;
     *   otherwise the values returned are undefined.
     */
    for (auto&& sampler: samplers.get()) {
      if (!supportsLinearFilering &&
          (sampler.filtering_mode == sycl::filtering_mode::linear))
          continue;

      auto data = get_image_input_data<T>(size);
      auto errors = get_error_data(1);

      failure_storage_t failures(size);
      {
        image_t image(data.data(),
                      image_format_channel<T>::order,
                      image_format_channel<T>::type, range);
        error_buffer_t errorBuffer(errors.data(),
                                   sycl::range<1>(errors.size()));
        failure_buffer_t failureBuffer(failures.data(),
                                       sycl::range<1>(failures.size()));

        check_command_group_reads<sampled_read_verifier_t>(
            tag, queue, range, image, errorBuffer, failureBuffer, sampler);
      }
      if (errors[0] != 0) {
        std::string message =
            "read(coords, sampler) did not read from the correct index for " +
            samplers.get_description(sampler);

        fail_for_accessor<T, dims, mode, target>(log, typeName, message);
        failures.dump(log, range, get_verification_range(range, sampler));
      }
    }
  }

 private:
  /**
   *  @brief Retrieve verification range which may differ from image range for
   *         out of scope sampled read verification
   */
  template <typename ... samplerT>
  image_range_t<dims, target> get_verification_range(
      const image_range_t<dims, target> &imageRange, samplerT ... sampler) {

    const size_t verificationOffset =
        image_accessor_sampler::supports_out_of_range(sampler...);
    return imageRange + verificationOffset;
  }

  /**
   * @brief Checks reading from a host image accessor
   * @tparam verifierT Functor type to use, switches between reading from an
   *         accessor using coordinates only and reading from an accessor
   *         using coordinates and a sampler
   * @tparam samplerT Provided for sampled reads
   * @param range Range of the image
   * @param image Image to use
   * @param errorBuffer Buffer where error will be stored
   * @param sampler Optional parameter, used for sampled reads only
   */
  template <template <sycl::target> class verifierT,
            typename ... samplerT>
  void check_command_group_reads(acc_type_tag::host,
                                 sycl::queue & /*queue*/,
                                 const image_range_t<dims, target> &range,
                                 image_t &image,
                                 error_buffer_t &errorBuffer,
                                 failure_buffer_t &failureBuffer,
                                 samplerT ... sampler) {
    static constexpr auto errorTarget = sycl::target::host_buffer;
    auto accessor =
        make_accessor<T, dims, mode, target, acc_placeholder::image>(image);
    auto errorAccessor =
        make_accessor<int, 1, errorMode, errorTarget, acc_placeholder::error>(
            errorBuffer);
    auto failureAccessor =
        make_accessor<failure_item_t, 1, errorMode, errorTarget,
                      acc_placeholder::error>(failureBuffer);
    auto verificationRange =
        get_verification_range(range, sampler...);

    auto verifier = verifierT<errorTarget>(
        range, verificationRange, accessor, errorAccessor, failureAccessor,
        sampler...);

    /** check image accessor for reads
    */
    auto idList = create_id_list<dims>(verificationRange);
    for (auto id : idList) {
      verifier(id);
    }
  }

  /**
   * @brief Checks reading from an image accessor
   * @tparam verifierT Functor type to use, switches between reading from an
   *         accessor using coordinates only and reading from an accessor
   *         using coordinates and a sampler
   * @tparam samplerT Provided for sampled reads
   * @param queue SYCL queue where a kernel will be executed
   * @param range Range of the image
   * @param image Image to use
   * @param errorBuffer Buffer where error will be stored
   * @param sampler Optional parameter, used for sampled reads only
   */
  template <template <sycl::target> class verifierT,
            typename ... samplerT>
  void check_command_group_reads(acc_type_tag::generic,
                                 sycl::queue &queue,
                                 const image_range_t<dims, target> &range,
                                 image_t &image,
                                 error_buffer_t &errorBuffer,
                                 failure_buffer_t &failureBuffer,
                                 samplerT ... sampler) {
    queue.submit([&](sycl::handler &handler) {
      static constexpr auto errorTarget =
          sycl::target::device;
      auto accessor =
          make_accessor<T, dims, mode, target, acc_placeholder::image>(
              image, handler);
      auto errorAccessor =
          make_accessor<int, 1, errorMode, errorTarget, acc_placeholder::error>(
              errorBuffer, handler);
      auto failureAccessor =
          make_accessor<failure_item_t, 1, errorMode, errorTarget,
                        acc_placeholder::error>(failureBuffer, handler);

      auto verificationRange =
        get_verification_range(range, sampler...);

      auto verifier = verifierT<errorTarget>(
          range, verificationRange, accessor, errorAccessor, failureAccessor,
          sampler...);

      /** check image accessor for reads
      */
      handler.parallel_for(verificationRange, verifier);
    });
  }
};

/** tests image accessors writes
*/
template <typename T, int dims, sycl::access_mode mode,
          sycl::target target>
class check_image_accessor_api_writes {
 public:
  static constexpr auto isImageArray =
      (target == sycl::target::image_array);
  using image_t = sycl::image<(isImageArray ? (dims + 1) : dims)>;

  size_t count;
  size_t size;

  void operator()(util::logger &log, sycl::queue &queue,
                  image_range_t<dims, target> range,
                  const std::string typeName) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target>("check_image_accessor_api_writes",
                                        typeName, log);
#endif  // VERBOSE_LOG

    static constexpr bool initialize = false;
    auto dataCoordsSyntax = get_image_input_data<T>(size, initialize);

    {
      auto imgCoordsSyntax =
          image_t(dataCoordsSyntax.data(), image_format_channel<T>::order,
                  image_format_channel<T>::type, range);

      check_command_group_writes(queue, range, imgCoordsSyntax, typeName,
                                 acc_type_tag::get<target,
                                                   acc_placeholder::image>());
    }

    const auto storageData = convert_image_bytes_to_data<T>(dataCoordsSyntax);
    const auto expected = get_expected_image_value<T>();

    bool success = true;
    for (const auto actual : storageData) {
      if (!check_elems_equal(actual, expected)) {
        success = false;
      }
    }
    if (!success) {
      fail_for_accessor<T, dims, mode, target>(log, typeName,
          "write(coords) did not assign to the correct index");
    }
  }

 private:
  /**
   * @brief Checks writing to a host image accessor
   * @param range Range of the image
   * @param imgCoordsSyntax Image to use when writing to an accessor
   *        using coordinates
   */
  void check_command_group_writes(sycl::queue &queue,
                                  const image_range_t<dims, target> &range,
                                  image_t &imgCoordsSyntax,
                                  const std::string typeName,
                                  acc_type_tag::host) {
    auto accCoordsSyntax =
        make_accessor<T, dims, mode, target, acc_placeholder::image>(
            imgCoordsSyntax);

    static_cast<void>(typeName); //no failure log here currently

    /** check image accessor for writes
    */
    auto idList = create_id_list<dims>(range);
    for (auto id : idList) {
      image_accessor_api_w<T, dims, mode, target>(size, accCoordsSyntax,
                                                  range)(id);
    }
  }

  /**
   * @brief Checks writing to an image accessor
   * @param queue SYCL queue where a kernel will be executed
   * @param range Range of the image
   * @param imgCoordsSyntax Image to use when writing to an accessor
   *        using coordinates
   */
  void check_command_group_writes(sycl::queue &queue,
                                  const image_range_t<dims, target> &range,
                                  image_t &imgCoordsSyntax,
                                  const std::string typeName,
                                  acc_type_tag::generic) {
    static_cast<void>(typeName); //no failure log here currently

    queue.submit([&](sycl::handler &handler) {
      auto accCoordsSyntax =
          make_accessor<T, dims, mode, target, acc_placeholder::image>(
              imgCoordsSyntax, handler);

      /** check image accessor for writes
      */
      handler.parallel_for(range, image_accessor_api_w<T, dims, mode, target>(
                                      size, accCoordsSyntax, range));
    });
  }
};

////////////////////////////////////////////////////////////////////////////////
// Enable tests for all combinations
////////////////////////////////////////////////////////////////////////////////

/** tests image accessors with different modes
*/

template <typename T, int dims, sycl::access_mode mode,
          sycl::target target>
void check_image_accessor_api_mode(util::logger &log,
                                   const std::string typeName,
                                   size_t count, size_t size,
                                   sycl::queue &queue,
                                   image_range_t<dims, target> range,
                                   acc_mode_tag::generic) {
  check_image_accessor_api_reads<T, dims, mode, target>{count, size}(
      log, queue, range, typeName);
  check_image_accessor_api_writes<T, dims, mode, target>{count, size}(
      log, queue, range, typeName);
}

template <typename T, int dims, sycl::access_mode mode,
          sycl::target target>
void check_image_accessor_api_mode(util::logger &log,
                                   const std::string typeName,
                                   size_t count, size_t size,
                                   sycl::queue &queue,
                                   image_range_t<dims, target> range,
                                   acc_mode_tag::write_only) {
  check_image_accessor_api_writes<T, dims, mode, target>{count, size}(
      log, queue, range, typeName);
}

template <typename T, int dims, sycl::access_mode mode,
          sycl::target target>
void check_image_accessor_api_mode(util::logger &log,
                                   const std::string typeName,
                                   size_t count, size_t size,
                                   sycl::queue &queue,
                                   image_range_t<dims, target> range,
                                   acc_mode_tag::read_only) {
  check_image_accessor_api_reads<T, dims, mode, target>{count, size}(
      log, queue, range, typeName);
}

template <typename T, int dims, sycl::access_mode mode,
          sycl::target target>
void check_image_accessor_api_mode(util::logger &log,
                                   const std::string typeName,
                                   size_t count, size_t size,
                                   sycl::queue &queue,
                                   image_range_t<dims, target> range) {
#ifdef VERBOSE_LOG
  log_accessor<T, dims, mode, target>("", typeName, log);
#endif

  /** check image accessor members
   */
  check_accessor_members<T, dims, mode, target>(log, typeName);

  /** check image accessor methods
  */
  check_image_accessor_api_methods<T, dims, mode, target>{count, size}(
      log, queue, range, typeName);

  check_image_accessor_api_mode<T, dims, mode, target>(
      log, typeName, count, size, queue, range, acc_mode_tag::get<mode>());
}

/**
 *  @brief Test image and image array accessors for all modes
 */
template <typename T, int dims, sycl::target target,
          typename ... argsT>
void check_image_accessor_api_target(acc_target_tag::generic,
                                     argsT&& ... args) {
  {
    constexpr auto mode = sycl::access_mode::read;
    check_image_accessor_api_mode<T, dims, mode, target>(
        std::forward<argsT>(args)...);
  }
  {
    constexpr auto mode = sycl::access_mode::write;
    check_image_accessor_api_mode<T, dims, mode, target>(
        std::forward<argsT>(args)...);
  }
  {
    constexpr auto mode = sycl::access_mode::discard_write;
    check_image_accessor_api_mode<T, dims, mode, target>(
        std::forward<argsT>(args)...);
  }
}

/**
 *  @brief Test host image accessors for all modes
 */
template <typename T, int dims, sycl::target target,
          typename ... argsT>
void check_image_accessor_api_target(acc_target_tag::host,
                                     argsT&& ... args) {
  {
    constexpr auto mode = sycl::access_mode::read;
    check_image_accessor_api_mode<T, dims, mode, target>(
        std::forward<argsT>(args)...);
  }
  {
    constexpr auto mode = sycl::access_mode::write;
    check_image_accessor_api_mode<T, dims, mode, target>(
        std::forward<argsT>(args)...);
  }
  {
    constexpr auto mode = sycl::access_mode::read_write;
    check_image_accessor_api_mode<T, dims, mode, target>(
        std::forward<argsT>(args)...);
  }
  {
    constexpr auto mode = sycl::access_mode::discard_write;
    check_image_accessor_api_mode<T, dims, mode, target>(
        std::forward<argsT>(args)...);
  }
}

/**
 *  @brief Tests image accessors with different targets for all modes
 */
template <typename T, int dims, sycl::target target,
          typename ... argsT>
void check_image_accessor_api_target_wrapper(argsT&& ... args) {

  static const auto tagretTag = acc_target_tag::get<T, target>();

  check_image_accessor_api_target<T, dims, target>(
      tagretTag, std::forward<argsT>(args)...);
}

/**
 *  @brief Run tests for 1 and 2 dimensions
 */
template <typename T, int dims, typename ... argsT>
void check_image_accessor_api_dim(acc_dims_tag::generic, util::logger &log,
                                  const std::string typeName,
                                  size_t count, argsT&& ... args) {

  const auto imageRange = make_test_range<dims>(count);

  /** check image accessor api for image
   */
  check_image_accessor_api_target_wrapper<T, dims,
                                          sycl::target::image>(
      log, typeName, count, std::forward<argsT>(args)..., imageRange);

  /** check image accessor api for host_image
   */
  check_image_accessor_api_target_wrapper<T, dims,
                                          sycl::target::host_image>(
      log, typeName, count, std::forward<argsT>(args)..., imageRange);

  /** check image accessor api for image_array
   */
  {
    static constexpr auto imageArrayTarget =
        sycl::target::image_array;
    static constexpr bool isImageArray = true;
    const auto imageArrayRange =
        make_test_range<image_dims<dims, imageArrayTarget>::value>(
            count, isImageArray);

    check_image_accessor_api_target_wrapper<T, dims, imageArrayTarget>(
        log, typeName, count, std::forward<argsT>(args)..., imageArrayRange);
  }
}

/**
 *  @brief Run tests for 3 dimensions
 */
template <typename T, int dims, typename ... argsT>
void check_image_accessor_api_dim(acc_dims_tag::num_dims<3>, util::logger &log,
                                  const std::string typeName,
                                  size_t count, argsT&& ... args) {

  const auto imageRange = make_test_range<dims>(count);

  /** check image accessor api for image
   */
  check_image_accessor_api_target_wrapper<T, dims,
                                          sycl::target::image>(
      log, typeName, count, std::forward<argsT>(args)..., imageRange);

  /** check image accessor api for host_image
   */
  check_image_accessor_api_target_wrapper<T, dims,
                                          sycl::target::host_image>(
      log, typeName, count, std::forward<argsT>(args)..., imageRange);

  /** image_array accessors only exist for 1D and 2D
   */
}

/** tests image accessors with different dimensions
*/
template <typename T, int dims, typename ... argsT>
void check_image_accessor_api_dim(util::logger &log, argsT&& ... args) {
  check_image_accessor_api_dim<T, dims>(acc_dims_tag::get<dims>(),
                                        log, std::forward<argsT>(args)...);
}

/** tests image accessors with different types
*/
template <typename T, typename /*extensionTag*/>
class check_image_accessor_api_type {
  static constexpr auto count = 8;
  static constexpr auto size = count * image_format_channel<T>::elementSize;

 public:
  void operator()(util::logger &log, sycl::queue &queue,
                  const std::string& typeName) {
    /**
     *  check image accessor api for all dimensions
     */
    check_image_accessor_api_dim<T, 1>(log, typeName, count, size, queue);
    check_image_accessor_api_dim<T, 2>(log, typeName, count, size, queue);
    check_image_accessor_api_dim<T, 3>(log, typeName, count, size, queue);
  }
};

}  // namespace

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_IMAGE_COMMON_H
