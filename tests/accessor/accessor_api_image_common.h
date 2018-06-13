/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_IMAGE_COMMON_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_IMAGE_COMMON_H

#include "../common/common.h"
#include "./../../util/math_helper.h"
#include "accessor_utility.h"
#include "accessor_api_common_all.h"

#include <array>
#include <numeric>
#include <sstream>

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
template <int dims, cl::sycl::access::target target>
using image_dims = std::integral_constant<
    int,
    ((target == cl::sycl::access::target::image_array) ? (dims + 1) : dims)>;

/**
 * @brief Alias to cl::sycl::id using the proper number of dimensions.
 *        Required for image array accessors.
 * @tparam dims Number of accessor dimensions
 * @tparam target Access target of the accessor
 */
template <int dims, cl::sycl::access::target target>
using image_id_t = cl::sycl::id<image_dims<dims, target>::value>;

/**
 * @brief Alias to image_id_t targetting image array accessors
 * @tparam dims Number of accessor dimensions
 * @tparam target Access target of the accessor
 */
template <int dims>
using image_array_id_t =
    image_id_t<dims, cl::sycl::access::target::image_array>;

/**
 * @brief Alias to cl::sycl::range using the proper number of dimensions.
 *        Required for image array accessors.
 * @tparam dims Number of accessor dimensions
 * @tparam target Access target of the accessor
 */
template <int dims, cl::sycl::access::target target>
using image_range_t = cl::sycl::range<image_dims<dims, target>::value>;

/**
 * @brief Helper struct for retrieving the image access coordinates
 *        when testing an image array accessor
 * @tparam dims Number of accessor dimensions
 */
template <int dims>
struct image_array_coords;

/**
 * @brief Helper struct for retrieving the image access coordinates
 *        when testing an image array accessor, specialization for 1D
 */
template <>
struct image_array_coords<1> {
  /**
   * @brief Retrieves the image access coordinates for an image array accessor
   * @param idx Work-item ID
   * @return Coordinates of dimension 1
   */
  static auto get(image_array_id_t<1> idx)
      -> decltype(image_access<1>::get_int(std::declval<cl::sycl::id<1>>())) {
    return image_access<1>::get_int(cl::sycl::id<1>{idx[0]});
  }
};

/**
 * @brief Helper struct for retrieving the image access coordinates
 *        when testing an image array accessor, specialization for 2D
 */
template <>
struct image_array_coords<2> {
  /**
   * @brief Retrieves the image access coordinates for an image array accessor
   * @param idx Work-item ID
   * @return Coordinates of dimension 2
   */
  static auto get(image_array_id_t<2> idx)
      -> decltype(image_access<2>::get_int(std::declval<cl::sycl::id<2>>())) {
    return image_access<2>::get_int(cl::sycl::id<2>{idx[0], idx[1]});
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
cl::sycl::range<imageDims> make_test_range(size_t count,
                                           bool isImageArray = false);

/**
 * @brief Constructs a range for testing, specialization for 1D images
 * @param count Number of elements to store in entire range
 * @return 1D range
 */
template <>
cl::sycl::range<1> make_test_range<1>(size_t count, bool /*isImageArray*/) {
  return {count};
}

/**
 * @brief Constructs a range for testing, specialization for 2D images
 * @param count Number of elements to store in entire range
 * @param isImageArray True if creating a test range for an image array accessor
 * @return 2D range
 */
template <>
cl::sycl::range<2> make_test_range<2>(size_t count, bool isImageArray) {
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
cl::sycl::range<3> make_test_range<3>(size_t count, bool isImageArray) {
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
struct image_format_channel<cl::sycl::cl_int4> {
  static constexpr cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::signed_int8;
  static constexpr cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
  using storage_t = cl::sycl::cl_char;
  static constexpr size_t elementSize = 4 * sizeof(storage_t);
};

/** image format channel order and type (specialization for cl_uint4)
*/
template <>
struct image_format_channel<cl::sycl::cl_uint4> {
  static constexpr cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::unsigned_int8;
  static constexpr cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
  using storage_t = cl::sycl::cl_uchar;
  static constexpr size_t elementSize = 4 * sizeof(storage_t);
};

/** image format channel order and type (specialization for cl_float4)
*/
template <>
struct image_format_channel<cl::sycl::cl_float4> {
  static constexpr cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::fp32;
  static constexpr cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
  using storage_t = cl::sycl::cl_float;
  static constexpr size_t elementSize = 4 * sizeof(storage_t);
};

/** image format channel order and type (specialization for cl_half4)
 */
template <>
struct image_format_channel<cl::sycl::cl_half4> {
  static constexpr cl::sycl::image_channel_type type =
      cl::sycl::image_channel_type::fp16;
  static constexpr cl::sycl::image_channel_order order =
      cl::sycl::image_channel_order::rgba;
  using storage_t = cl::sycl::cl_half;
  static constexpr size_t elementSize = 4 * sizeof(storage_t);
};

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. 1.0f by default.
*/
template <typename elementT>
struct use_normalization_coefficient : std::false_type {};

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. Specialization for cl::sycl::cl_float4.
*/
template <>
struct use_normalization_coefficient<cl::sycl::cl_float4> : std::true_type {};

/** specialized struct for defining the normalization coefficient for an image
 * accessor type. Specialization for cl::sycl::cl_half4.
*/
template <>
struct use_normalization_coefficient<cl::sycl::cl_half4> : std::true_type {};

/**
 * @brief Retrieves the expected single image value when reading to
 *        or writing from an image accessor
 * @tparam T Type used in the image accessor
 * @return Some constant value of the underlying type of T
 */
template <typename T>
typename image_format_channel<T>::storage_t get_expected_image_value() {
  using storage_t = typename image_format_channel<T>::storage_t;
  if (use_normalization_coefficient<T>::value) {
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
cl::sycl::vector_class<cl::sycl::byte> convert_image_data_to_bytes(
    const cl::sycl::vector_class<storage_t> &storageData, size_t byteSize) {
  using byte_t = cl::sycl::byte;
  const auto byteDataPtr = reinterpret_cast<const byte_t *>(storageData.data());
  auto byteData =
      cl::sycl::vector_class<byte_t>(byteDataPtr, byteDataPtr + byteSize);
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
cl::sycl::vector_class<storage_t> convert_image_bytes_to_data(
    const cl::sycl::vector_class<cl::sycl::byte> &byteData) {
  const auto dataCount = (byteData.size() / sizeof(storage_t));
  const auto storageDataPtr =
      reinterpret_cast<const storage_t *>(byteData.data());
  auto storageData = cl::sycl::vector_class<storage_t>(
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
cl::sycl::vector_class<cl::sycl::byte> get_image_input_data(
    size_t byteSize, bool initialize = true) {
  using storage_t = typename image_format_channel<T>::storage_t;
  const auto dataCount = (byteSize / sizeof(storage_t));
  auto singleElem = storage_t{0};
  if (initialize) {
    singleElem = get_expected_image_value<T>();
  }
  const auto data = cl::sycl::vector_class<storage_t>(dataCount, singleElem);
  return convert_image_data_to_bytes<T>(data, byteSize);
}

template <typename T, int dims, cl::sycl::access::target target,
          cl::sycl::access::mode mode>
T read_image_acc(cl::sycl::accessor<T, dims, mode, target> &acc,
                 cl::sycl::id<dims> idx) {
  return acc.read(image_access<dims>::get_int(idx));
}

template <typename T, int dims, cl::sycl::access::mode mode>
T read_image_acc(cl::sycl::accessor<T, dims, mode,
                                    cl::sycl::access::target::image_array> &acc,
                 image_array_id_t<dims> idx) {
  const auto coords = image_array_coords<dims>::get(idx);
  return acc[idx[dims]].read(coords);
}

template <typename T, int dims, cl::sycl::access::target target,
          cl::sycl::access::mode mode>
T read_image_acc_sampled(cl::sycl::accessor<T, dims, mode, target> &acc,
                         cl::sycl::sampler smpl, cl::sycl::id<dims> idx) {
  return acc.read(image_access<dims>::get_int(idx), smpl);
}

template <typename T, int dims, cl::sycl::access::mode mode>
T read_image_acc_sampled(
    cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::image_array>
        &acc,
    cl::sycl::sampler smpl, image_array_id_t<dims> idx) {
  const auto coords = image_array_coords<dims>::get(idx);
  return acc[idx[dims]].read(coords, smpl);
}

template <typename T, int dims, cl::sycl::access::target target,
          cl::sycl::access::mode mode>
void write_image_acc(cl::sycl::accessor<T, dims, mode, target> &acc,
                     cl::sycl::id<dims> idx, T value) {
  const auto coords = image_access<dims>::get_int(idx);
  acc.write(coords, value);
}

template <typename T, int dims, cl::sycl::access::mode mode>
void write_image_acc(
    cl::sycl::accessor<T, dims, mode, cl::sycl::access::target::image_array>
        &acc,
    image_array_id_t<dims> idx, T value) {
  const auto coords = image_array_coords<dims>::get(idx);
  acc[idx[dims]].write(coords, value);
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

/** tests image accessors reads
*/
template <typename T, int dim, cl::sycl::access::mode mode,
          cl::sycl::access::target target, cl::sycl::access::target errorTarget>
class image_accessor_api_r {
  using acc_t = cl::sycl::accessor<T, dim, mode, target>;
  using error_acc_t = cl::sycl::accessor<int, 1, errorMode, errorTarget>;

  acc_t m_accCoordsSyntax;
  acc_t m_accCoordsSamplerSyntax;
  cl::sycl::sampler m_sampler;

  error_acc_t m_errorAccessor;
  image_range_t<dim, target> m_range;
  size_t size;

 public:
  image_accessor_api_r(size_t size_, acc_t accCoordsSyntax,
                       acc_t accCoordsSamplerSyntax, cl::sycl::sampler smpl,
                       error_acc_t errorAccessor,
                       image_range_t<dim, target> rng)
      : m_accCoordsSyntax(accCoordsSyntax),
        m_accCoordsSamplerSyntax(accCoordsSamplerSyntax),
        m_sampler(smpl),
        m_errorAccessor(errorAccessor),
        m_range(rng),
        size(size_) {}

  void operator()(image_id_t<dim, target> idx) {
    const auto expected = get_expected_image_elem<T>();
    T elem;

    /** check coordinates read syntax
     */
    elem = read_image_acc(m_accCoordsSyntax, idx);

    if (!check_elems_equal(elem, expected)) {
      m_errorAccessor[0] = 1;
    }

    /** check coordinates with sampler read syntax
     */
    elem = read_image_acc_sampled(m_accCoordsSamplerSyntax, m_sampler, idx);

    if (!check_elems_equal(elem, expected)) {
      m_errorAccessor[1] = 1;
    }
  }
};

/** tests image accessors writes
*/
template <typename T, int dim, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class image_accessor_api_w {
  using acc_t = cl::sycl::accessor<T, dim, mode, target>;

  acc_t m_accCoordsSyntax;
  image_range_t<dim, target> m_range;
  size_t size;

 public:
  image_accessor_api_w(size_t size_, acc_t accCoordsSyntax,
                       image_range_t<dim, target> rng)
      : m_accCoordsSyntax(accCoordsSyntax), m_range(rng), size(size_) {}

  void operator()(image_id_t<dim, target> idx) {
    const auto elem = get_expected_image_elem<T>();

    /** check coords write syntax
    */
    write_image_acc(m_accCoordsSyntax, idx, elem);
  }
};

/** tests image accessors methods
*/
template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class check_image_accessor_api_methods {
 public:
  static constexpr auto isImageArray =
      (target == cl::sycl::access::target::image_array);
  using image_t = cl::sycl::image<(isImageArray ? (dims + 1) : dims)>;

  size_t count;
  size_t size;

  void operator()(util::logger &log, cl::sycl::queue &queue,
                  image_range_t<dims, target> range) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target>("check_image_accessor_api_methods",
                                        log);
#endif  // VERBOSE_LOG

    auto data = get_image_input_data<T>(size);
    auto image = image_t(data.data(), image_format_channel<T>::order,
                         image_format_channel<T>::type, range);
    check_methods_helper(log, queue, image, acc_type_tag::get<target>());
  }

 private:
  template <typename acc_t>
  void check_methods(util::logger &log, const acc_t &accessor) const {
    /** check get_count() method
     */
    auto accessorCount = accessor.get_count();
    check_return_type<size_t>(log, accessor.get_count(), "get_count");
    if (accessorCount != count) {
      FAIL(log, "accessor does not return the correct count");
    }

    /** check get_size() method
     */
    auto accessorSize = accessor.get_size();
    check_return_type<size_t>(log, accessor.get_size(), "get_size");
    if (accessorSize != size) {
      FAIL(log, "accessor does not return the correct size");
    }
  }

  /**
   * @brief Checks member functions of a host image accessor
   * @param log The logger object
   * @param image SYCL image to request access to
   */
  void check_methods_helper(util::logger &log, cl::sycl::queue & /*queue*/,
                            image_t &image, acc_type_tag::host) const {
    check_methods(log, make_accessor<T, dims, mode, target>(image));
  }

  /**
   * @brief Checks member functions of an image accessor
   * @param log The logger object
   * @param queue SYCL queue where a kernel will be executed
   * @param image SYCL image to request access to
   */
  void check_methods_helper(util::logger &log, cl::sycl::queue &queue,
                            image_t &image, acc_type_tag::generic) const {
    queue.submit([&](cl::sycl::handler &handler) {
      check_methods(log, make_accessor<T, dims, mode, target>(image, handler));
      handler.single_task(dummy_functor<T>());
    });
  }
};

/** tests image accessors reads
*/
template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class check_image_accessor_api_reads {
 public:
  static constexpr auto isImageArray =
      (target == cl::sycl::access::target::image_array);
  using image_t = cl::sycl::image<(isImageArray ? (dims + 1) : dims)>;

  size_t count;
  size_t size;

  void operator()(util::logger &log, cl::sycl::queue &queue,
                  image_range_t<dims, target> range) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target>("check_image_accessor_api_reads", log);
#endif  // VERBOSE_LOG

    auto dataCoordsSyntax = get_image_input_data<T>(size);
    auto dataCoordsSamplerSyntax = get_image_input_data<T>(size);
    auto errors = get_error_data(2);

    {
      image_t imgCoordsSyntax(dataCoordsSyntax.data(),
                              image_format_channel<T>::order,
                              image_format_channel<T>::type, range);
      image_t imgCoordsSamplerSyntax(dataCoordsSamplerSyntax.data(),
                                     image_format_channel<T>::order,
                                     image_format_channel<T>::type, range);
      error_buffer_t errorBuffer(errors.data(),
                                 cl::sycl::range<1>(errors.size()));

      check_command_group_reads(queue, range, imgCoordsSyntax,
                                imgCoordsSamplerSyntax, errorBuffer,
                                acc_type_tag::get<target>());
    }

    if (errors[0] != 0) {
      FAIL(log, "read(coords) did not read from the correct index");
    }
    if (errors[1] != 0) {
      FAIL(log, "read(coords, sampler) did not read from the correct index");
    }
  }

 private:
  /**
   * @brief Checks reading from a host image accessor
   * @param range Range of the image
   * @param imgCoordsSyntax Image to use when reading from an accessor
   *        using coordinates only
   * @param imgCoordsSamplerSyntax Image to use when reading from an accessor
   *        using coordinates and a sampler
   * @param errorBuffer Buffer where errors will be stored
   */
  void check_command_group_reads(cl::sycl::queue & /*queue*/,
                                 const image_range_t<dims, target> &range,
                                 image_t &imgCoordsSyntax,
                                 image_t &imgCoordsSamplerSyntax,
                                 error_buffer_t &errorBuffer,
                                 acc_type_tag::host) {
    static constexpr auto errorTarget = cl::sycl::access::target::host_buffer;
    auto accCoordsSyntax =
        make_accessor<T, dims, mode, target>(imgCoordsSyntax);
    auto accCoordsSamplerSyntax =
        make_accessor<T, dims, mode, target>(imgCoordsSamplerSyntax);
    auto errorAccessor =
        make_accessor<int, 1, errorMode, errorTarget>(errorBuffer);
    auto sampler = cl::sycl::sampler(
        cl::sycl::coordinate_normalization_mode::unnormalized,
        cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::nearest);
    /** check image accessor for reads
    */
    auto idList = create_id_list<dims>(range);
    for (auto id : idList) {
      image_accessor_api_r<T, dims, mode, target, errorTarget>(
          size, accCoordsSyntax, accCoordsSamplerSyntax, sampler, errorAccessor,
          range)(id);
    }
  }

  /**
   * @brief Checks reading from an image accessor
   * @param queue SYCL queue where a kernel will be executed
   * @param range Range of the image
   * @param imgCoordsSyntax Image to use when reading from an accessor
   *        using coordinates only
   * @param imgCoordsSamplerSyntax Image to use when reading from an accessor
   *        using coordinates and a sampler
   * @param errorBuffer Buffer where errors will be stored
   */
  void check_command_group_reads(cl::sycl::queue &queue,
                                 const image_range_t<dims, target> &range,
                                 image_t &imgCoordsSyntax,
                                 image_t &imgCoordsSamplerSyntax,
                                 error_buffer_t &errorBuffer,
                                 acc_type_tag::generic) {
    queue.submit([&](cl::sycl::handler &handler) {
      static constexpr auto errorTarget =
          cl::sycl::access::target::global_buffer;
      auto accCoordsSyntax =
          make_accessor<T, dims, mode, target>(imgCoordsSyntax, handler);
      auto accCoordsSamplerSyntax =
          make_accessor<T, dims, mode, target>(imgCoordsSamplerSyntax, handler);
      auto errorAccessor =
          make_accessor<int, 1, errorMode, errorTarget>(errorBuffer, handler);
      auto sampler = cl::sycl::sampler(
          cl::sycl::coordinate_normalization_mode::unnormalized,
          cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::nearest);

      /** check image accessor for reads
      */
      handler.parallel_for(
          range, image_accessor_api_r<T, dims, mode, target, errorTarget>(
                     size, accCoordsSyntax, accCoordsSamplerSyntax, sampler,
                     errorAccessor, range));
    });
  }
};

/** tests image accessors writes
*/
template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
class check_image_accessor_api_writes {
 public:
  static constexpr auto isImageArray =
      (target == cl::sycl::access::target::image_array);
  using image_t = cl::sycl::image<(isImageArray ? (dims + 1) : dims)>;

  size_t count;
  size_t size;

  void operator()(util::logger &log, cl::sycl::queue &queue,
                  image_range_t<dims, target> range) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target>("check_image_accessor_api_writes", log);
#endif  // VERBOSE_LOG

    static constexpr bool initialize = false;
    auto dataCoordsSyntax = get_image_input_data<T>(size, initialize);

    {
      auto imgCoordsSyntax =
          image_t(dataCoordsSyntax.data(), image_format_channel<T>::order,
                  image_format_channel<T>::type, range);

      check_command_group_writes(queue, range, imgCoordsSyntax,
                                 acc_type_tag::get<target>());
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
      FAIL(log, "write(coords) did not assign to the correct index");
    }
  }

 private:
  /**
   * @brief Checks writing to a host image accessor
   * @param range Range of the image
   * @param imgCoordsSyntax Image to use when writing to an accessor
   *        using coordinates
   */
  void check_command_group_writes(cl::sycl::queue &queue,
                                  const image_range_t<dims, target> &range,
                                  image_t &imgCoordsSyntax,
                                  acc_type_tag::host) {
    auto accCoordsSyntax =
        make_accessor<T, dims, mode, target>(imgCoordsSyntax);

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
  void check_command_group_writes(cl::sycl::queue &queue,
                                  const image_range_t<dims, target> &range,
                                  image_t &imgCoordsSyntax,
                                  acc_type_tag::generic) {
    queue.submit([&](cl::sycl::handler &handler) {
      auto accCoordsSyntax =
          make_accessor<T, dims, mode, target>(imgCoordsSyntax, handler);

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

template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
void check_image_accessor_api_mode(util::logger &log, size_t count, size_t size,
                                   cl::sycl::queue &queue,
                                   image_range_t<dims, target> range,
                                   acc_mode_tag::generic) {
  check_image_accessor_api_writes<T, dims, mode, target>{count, size}(
      log, queue, range);
}

template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
void check_image_accessor_api_mode(util::logger &log, size_t count, size_t size,
                                   cl::sycl::queue &queue,
                                   image_range_t<dims, target> range,
                                   acc_mode_tag::read_only) {
  check_image_accessor_api_reads<T, dims, mode, target>{count, size}(log, queue,
                                                                     range);
}

template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target>
void check_image_accessor_api_mode(util::logger &log, size_t count, size_t size,
                                   cl::sycl::queue &queue,
                                   image_range_t<dims, target> range) {
  log_accessor<T, dims, mode, target>("", log);

  /** check image accessor members
   */
  check_accessor_members<T, dims, mode, target>(log);

  /** check image accessor methods
  */
  check_image_accessor_api_methods<T, dims, mode, target>{count, size}(
      log, queue, range);

  check_image_accessor_api_mode<T, dims, mode, target>(
      log, count, size, queue, range, acc_mode_tag::get<mode>());
}

/** tests image accessors with different targets
*/
template <typename T, int dims, cl::sycl::access::target target>
void check_image_accessor_api_target(util::logger &log, size_t count,
                                     size_t size, cl::sycl::queue &queue,
                                     image_range_t<dims, target> range) {
  /** check image accessor api for read
  */
  check_image_accessor_api_mode<T, dims, cl::sycl::access::mode::read, target>(
      log, count, size, queue, range);

  /** check image accessor api for write
  */
  check_image_accessor_api_mode<T, dims, cl::sycl::access::mode::write, target>(
      log, count, size, queue, range);
}

template <typename T, int dims>
void check_image_accessor_api_dim(util::logger &log, size_t count, size_t size,
                                  cl::sycl::queue &queue,
                                  acc_dims_tag::generic) {
  const auto imageRange = make_test_range<dims>(count);

  /** check image accessor api for image
   */
  check_image_accessor_api_target<T, dims, cl::sycl::access::target::image>(
      log, count, size, queue, imageRange);

  /** check image accessor api for host_image
   */
  check_image_accessor_api_target<T, dims,
                                  cl::sycl::access::target::host_image>(
      log, count, size, queue, imageRange);

  /** check image accessor api for image_array
   */
  {
    static constexpr auto imageArrayTarget =
        cl::sycl::access::target::image_array;
    static constexpr bool isImageArray = true;
    const auto imageArrayRange =
        make_test_range<image_dims<dims, imageArrayTarget>::value>(
            count, isImageArray);

    check_image_accessor_api_target<T, dims, imageArrayTarget>(
        log, count, size, queue, imageArrayRange);
  }
}

template <typename T, int dims>
void check_image_accessor_api_dim(util::logger &log, size_t count, size_t size,
                                  cl::sycl::queue &queue,
                                  acc_dims_tag::num_dims<3>) {
  const auto imageRange = make_test_range<dims>(count);

  /** check image accessor api for image
   */
  check_image_accessor_api_target<T, dims, cl::sycl::access::target::image>(
      log, count, size, queue, imageRange);

  /** check image accessor api for host_image
   */
  check_image_accessor_api_target<T, dims,
                                  cl::sycl::access::target::host_image>(
      log, count, size, queue, imageRange);

  /** image_array accessors only exist for 1D and 2D
   */
}

/** tests image accessors with different dimensions
*/
template <typename T, int dims>
void check_image_accessor_api_dim(util::logger &log, size_t count, size_t size,
                                  cl::sycl::queue &queue) {
  check_image_accessor_api_dim<T, dims>(log, count, size, queue,
                                        acc_dims_tag::get<dims>());
}

/** tests image accessors with different types
*/
template <typename T>
void check_image_accessor_api_type(util::logger &log, cl::sycl::queue &queue) {
  const size_t count = 8;
  const size_t size = count * image_format_channel<T>::elementSize;

  /** check image accessor api for 1 dimension
   */
  check_image_accessor_api_dim<T, 1>(log, count, size, queue);

  /** check image accessor api for 2 dimension
   */
  check_image_accessor_api_dim<T, 2>(log, count, size, queue);

  /** check image accessor api for 3 dimension
   */
  check_image_accessor_api_dim<T, 3>(log, count, size, queue);
}

}  // namespace

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_IMAGE_COMMON_H
