/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
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
#ifndef __SYCLCTS_TESTS_HANDLER_COPY_COMMON_H
#define __SYCLCTS_TESTS_HANDLER_COPY_COMMON_H

#include <memory>
#include <mutex>
#include <regex>
#include <sstream>

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"

namespace handler_copy_common {
using namespace sycl_cts;

using mode_t = sycl::access_mode;
using target_t = sycl::target;

// TODO: Also test image accessors
// TODO: Also test copying between buffers of different data types.

/**
 * @brief Helper class designed to construct useful failure messages for all
 * test case permutations.
 */
class log_helper {
 public:
  log_helper() {}

  template <typename dataT>
  log_helper set_data_type() const {
    auto result = *this;
    if (std::is_same<dataT, char>::value) result.dataType = "char";
    if (std::is_same<dataT, short>::value) result.dataType = "short";
    if (std::is_same<dataT, int>::value) result.dataType = "int";
    if (std::is_same<dataT, long>::value) result.dataType = "long";
    if (std::is_same<dataT, float>::value) result.dataType = "float";
    if (std::is_same<dataT, double>::value) result.dataType = "double";
    if (std::is_same<dataT, sycl::char2>::value)
      result.dataType = "sycl::char2";
    if (std::is_same<dataT, sycl::short3>::value)
      result.dataType = "sycl::short3";
    if (std::is_same<dataT, sycl::int4>::value)
      result.dataType = "sycl::int4";
    if (std::is_same<dataT, sycl::long8>::value)
      result.dataType = "sycl::long8";
    if (std::is_same<dataT, sycl::float8>::value)
      result.dataType = "sycl::float8";
    if (std::is_same<dataT, sycl::double16>::value)
      result.dataType = "sycl::double16";
    return result;
  }

  log_helper set_dim_src(int dim) const {
    auto result = *this;
    result.dimSrc = dim;
    return result;
  }

  log_helper set_dim_dst(int dim) const {
    auto result = *this;
    result.dimDst = dim;
    return result;
  }

  log_helper set_mode_src(mode_t mode) const {
    auto result = *this;
    result.modeSrc = get_mode_string(mode);
    return result;
  }

  log_helper set_mode_dst(mode_t mode) const {
    auto result = *this;
    result.modeDst = get_mode_string(mode);
    return result;
  }

  log_helper set_target(target_t target) const {
    auto result = *this;
    result.target = get_target_string(target);
    return result;
  }

  log_helper set_line(int line) const {
    auto result = *this;
    result.line = line;
    return result;
  }

  log_helper set_extra_info(const std::string& info) const {
    auto result = *this;
    result.info = info;
    return result;
  }

  log_helper set_op(const std::string& pattern) const {
    auto result = *this;
    result.pattern = pattern;
    return result;
  }

  void fail(const std::string& reason) const {
    FAIL(make_description() + " failed: " + reason, line);
  }

  void note(const std::string& message) const {
    WARN(make_description() + " info: " + message);
  }

 private:
  std::string dataType = "(unknown data type)";
  int dimSrc = -1;
  int dimDst = -1;
  std::string modeSrc = "(unknown mode)";
  std::string modeDst = "(unknown mode)";
  std::string target = "(unknown target)";
  int line = __LINE__;
  std::string info = "";
  std::string pattern = "";

  static std::string get_mode_string(mode_t mode) {
    switch (mode) {
      case mode_t::read:
        return "read";
      case mode_t::write:
        return "write";
      case mode_t::read_write:
        return "read_write";
      case mode_t::discard_write:
        return "discard_write";
      case mode_t::discard_read_write:
        return "discard_read_write";
      case mode_t::atomic:
        return "atomic";
      default:
        return "(unknown mode)";
    }
  }

  static std::string get_target_string(target_t target) {
    switch (target) {
      case target_t::device:
        return "device";
      case target_t::constant_buffer:
        return "constant_buffer";
      default:
        return "(unknown target)";
    }
  }

  std::string make_description() const {
    auto desc = pattern;
    desc = std::regex_replace(desc, std::regex("\\$dataT"), dataType);
    desc = std::regex_replace(desc, std::regex("\\$dim_src"),
                              std::to_string(dimSrc));
    desc = std::regex_replace(desc, std::regex("\\$dim_dst"),
                              std::to_string(dimDst));
    desc = std::regex_replace(desc, std::regex("\\$mode_src"), modeSrc);
    desc = std::regex_replace(desc, std::regex("\\$mode_dst"), modeDst);
    desc = std::regex_replace(desc, std::regex("\\$target"), target);
    if (!info.empty()) {
      desc += " [" + info + "]";
    }
    return desc;
  }
};

/**
 * @brief Helper class that provides a uniform creation and comparison interface
 * for different data types, in particular scalar and vector types (see
 * specialization below).
 */
template <typename T>
struct type_helper {
  static T make(size_t v) { return static_cast<T>(v); }
  static size_t value(const T& v) { return static_cast<size_t>(v); }
  static bool equal(const T& lhs, const T& rhs) {
    return value(lhs) == value(rhs);
  }
};

template <typename dataT, int numElements>
struct type_helper<sycl::vec<dataT, numElements>> {
  using T = sycl::vec<dataT, numElements>;
  static T make(size_t v) {
    return T{static_cast<typename T::element_type>(v)};
  }
  static size_t value(const T& v) { return static_cast<size_t>(v.s0()); }
  static bool equal(const T& lhs, const T& rhs) {
    // Ideally we'd check that all components are equal, however unfortunately
    // SYCL 1.2.1 doesn't specify a generic subscript operator for vector types.
    return value(lhs) == value(rhs);
  }
};

template <typename dataT, int dims>
struct scalar_init_op {
  dataT value;
  scalar_init_op(dataT value) : value(value) {}
  dataT operator()(sycl::id<dims>) const { return value; }
};

template <typename dataT, int dims>
struct encode_index_init_op {
  dataT operator()(sycl::id<dims> id) const {
    size_t result = 10000 * id[0];
    if (dims > 1) result += 100 * id[1];
    if (dims > 2) result += id[2];
    // If dataT can't fit the value the result is implementation defined,
    // and we may weaken the robustness of the test.
    // However, it's still better than using a constant value throughout.
    return type_helper<dataT>::make(result);
  }
};

template <typename dataT, int dims, typename initOp>
class fill_kernel;

template <typename dataT, int dims, template <typename, int> class initOp>
void fill_buffer(sycl::queue& queue, sycl::buffer<dataT, dims>& buf,
                 initOp<dataT, dims> init_op) {
  queue.submit([&](sycl::handler& cgh) {
    auto acc = buf.template get_access<mode_t::discard_write>(cgh);
    cgh.parallel_for<fill_kernel<dataT, dims, initOp<dataT, dims>>>(
        buf.get_range(), [=](sycl::id<dims> id) { acc[id] = init_op(id); });
  });
}

/**
 * Helper class that facilitates the usage of SYCL ranges and ids in
 * generic contexts. It provides a uniform factory interface for all
 * dimensionalities, and allows to cast between ranges and ids of different
 * dimensionalities (initializing empty entries to @p default_value, where
 * necessary).
 *
 * This template is not meant to be used directly, instead see range_helper and
 * id_helper for concrete instantiations.
 */
template <template <int> class T, int dims, size_t default_value>
struct range_id_helper {};

template <template <int> class T, size_t default_value>
struct range_id_helper<T, 1, default_value> {
  static T<1> make(size_t d0, size_t, size_t) { return T<1>{d0}; }

  template <template <int> class other, int other_dims,
            size_t dv = default_value>
  static T<1> cast(const other<other_dims>& o) {
    return T<1>{o[0]};
  }
};

template <template <int dims> class T, size_t default_value>
struct range_id_helper<T, 2, default_value> {
  static T<2> make(size_t d0, size_t d1, size_t) { return T<2>{d0, d1}; }

  template <template <int> class other, int other_dims,
            size_t dv = default_value>
  static T<2> cast(const other<other_dims>& o) {
    return T<2>{o[0], other_dims >= 2 ? o[1] : dv};
  }
};

template <template <int dims> class T, size_t default_value>
struct range_id_helper<T, 3, default_value> {
  static T<3> make(size_t d0, size_t d1, size_t d2) { return T<3>{d0, d1, d2}; }

  template <template <int> class other, int other_dims,
            size_t dv = default_value>
  static T<3> cast(const other<other_dims>& o) {
    return T<3>{o[0], other_dims >= 2 ? o[1] : dv, other_dims == 3 ? o[2] : dv};
  }
};

template <int dims>
using range_helper = range_id_helper<sycl::range, dims, 1>;

template <int dims>
using id_helper = range_id_helper<sycl::id, dims, 0>;

template <int dim>
sycl::range<3> default_large_range() {
  return range_helper<3>::cast(range_helper<dim>::make(5, 7, 9));
}

template <int dim_large, int dim_small, bool transposed_copy = false>
sycl::range<3> transform_large_range_into_small(sycl::range<3> largeBufRange) {
  sycl::range<3> smallBufRange = sycl::range<3>(1, 1, 1);

  // Condense large range into small range so that both
  // have the same size (= same number of items).
  for (int d = 0; d < dim_large; ++d) {
    if (transposed_copy) {
      smallBufRange[std::min(d, dim_small - 1)] *=
          largeBufRange[dim_large - d - 1];
    } else {
      smallBufRange[std::min(d, dim_small - 1)] *= largeBufRange[d];
    }
  }
  return smallBufRange;
}
/**
 * @brief The copy_test_context encapsulates all host and device data required
 * for testing, and provides utility functions for verifying the result
 * of the various explicit memory operations.
 *
 * Based on the provided template parameters, the copy_test_context generates
 * appropriately sized buffers, copy ranges and offsets. If the dimensionality
 * of the source and destination buffers doesn't match, it ensures that the same
 * number of items will be copied regardless.
 *
 * @tparam dataT
 * @tparam dim_src
 * @tparam dim_dst
 * @tparam strided_copy Whether copies should be strided, i.e., with an offset
 * and range smaller than the source and/or destination buffer range.
 * @tparam transposed_copy If dim_src == dim_dst, a transposed copy will
 * generate a target buffer that has its coordinates flipped, e.g., instead of
 * copying a range [4,8] to [4,8], it will copy [4,8] to [8,4]. If the source
 * and target dimensions don't match, dimensions will be condensed in reverse
 * order (see copy_test_context::setup_ranges).
 */
template <typename dataT, int dim_src, int dim_dst, bool strided_copy,
          bool transposed_copy>
class copy_test_context {
  using host_shared_ptr = std::shared_ptr<dataT>;
  using buffer_src_t = sycl::buffer<dataT, dim_src>;
  using buffer_dst_t = sycl::buffer<dataT, dim_dst>;
  using th = type_helper<dataT>;

 public:
  explicit copy_test_context(sycl::queue& queue) : queue(queue) {
    setup_ranges();

    srcBufHostMemory =
        host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());
    srcHostPtr =
        host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());
    dstHostPtr =
        host_shared_ptr(new dataT[numElems], std::default_delete<dataT[]>());

    for (size_t i = 0; i < numElems; ++i) {
      srcBufHostMemory.get()[i] = static_cast<dataT>(hostCanary);
      srcHostPtr.get()[i] = encode_index_init_op<dataT, dim_src>{}(
          reconstruct_index(srcBufRange, i));
      dstHostPtr.get()[i] = static_cast<dataT>(hostCanary);
    }

    srcBuf = std::unique_ptr<buffer_src_t>(new buffer_src_t(
        srcBufHostMemory, srcBufRange,
        sycl::property_list{
            sycl::property::buffer::use_mutex{srcBufHostMemoryMutex}}));
    dstBuf = std::unique_ptr<buffer_dst_t>(new buffer_dst_t(dstBufRange));

    fill_buffer(queue, *srcBuf, encode_index_init_op<dataT, dim_src>());
    fill_buffer(queue, *dstBuf, scalar_init_op<dataT, dim_dst>(deviceCanary));

    queue.wait_and_throw();
  }

  /**
   * @brief Verifies a device to host copy.
   *
   * While the device source may be strided, the target host memory region is
   * always dense.
   */
  template <typename test_fn>
  void verify_d2h_copy(test_fn fn, const log_helper& lh) const {
    run_test_function(fn, lh);

    for (size_t i = 0; i < numElems; ++i) {
      const auto received = dstHostPtr.get()[i];

      if (i < srcCopyRange.size()) {
        const auto srcRelIdx = reconstruct_index(srcCopyRange, i);
        const auto expected =
            encode_index_init_op<dataT, dim_src>{}(srcCopyOffset + srcRelIdx);

        if (!th::equal(received, expected)) {
          log_error(lh, sycl::id<3>(i, 0, 0), received, expected);
          return;
        }
      } else {
        if (!th::equal(received, hostCanary)) {
          log_canary_violation(lh, sycl::id<3>(i, 0, 0), received);
          return;
        }
      }
    }
  }

  /**
   * @brief Verifies that the host memory backing the source buffer has been
   * updated correctly.
   *
   * Note that we don't check any canary values outside of the updated region
   * (when using a ranged accessor) as a SYCL implementation is free to also
   * update other parts of the host memory.
   */
  template <typename test_fn>
  void verify_update_host(test_fn fn, const log_helper& lh) const {
    run_test_function(fn, lh);

    std::lock_guard<std::mutex> lock(srcBufHostMemoryMutex);
    for (size_t i = 0; i < numElems; ++i) {
      const auto idx = reconstruct_index(srcBufRange, i);
      if (is_within_window(srcCopyOffset, srcCopyRange, idx)) {
        const auto expected = encode_index_init_op<dataT, dim_src>{}(idx);
        const auto received = srcBufHostMemory.get()[i];
        if (!th::equal(received, expected)) {
          log_error(lh, sycl::id<3>(i, 0, 0), received, expected);
          return;
        }
      }
    }
  }

  /**
   * @brief Verifies a host to device copy.
   *
   * While the device target may be strided, the source host memory region is
   * always dense.
   */
  template <typename test_fn>
  void verify_h2d_copy(test_fn fn, const log_helper& lh) {
    run_test_function(fn, lh);

    verify_device_copy(
        [this](size_t relativeLinearIdx) {
          // SYCL doesn't support strided H2D copies, so this is dense.
          const auto expected = srcHostPtr.get()[relativeLinearIdx];
          return expected;
        },
        lh);
  }

  /**
   * @brief Verifies a device to device copy.
   */
  template <typename test_fn>
  void verify_d2d_copy(test_fn fn, const log_helper& lh) {
    run_test_function(fn, lh);

    verify_device_copy(
        [this](size_t relativeLinearIdx) {
          // Compute relative index in source copy range.
          const auto srcRelIdx =
              reconstruct_index(srcCopyRange, relativeLinearIdx);
          // Convert to absolute source index and use to compute expected value.
          const auto expected =
              encode_index_init_op<dataT, dim_src>{}(srcCopyOffset + srcRelIdx);
          return expected;
        },
        lh);
  }

  /**
   * @brief Verifies that the accessed device memory region has been filled
   * correctly.
   *
   * @param fn
   * @param expected The value that was used to fill the region.
   * @param lh
   */
  template <typename test_fn>
  void verify_fill(test_fn fn, dataT expected, const log_helper& lh) {
    run_test_function(fn, lh);

    // TODO: Consider verifying directly on device.
    auto acc = dstBuf->template get_access<sycl::access_mode::read>();
    for (size_t i = 0; i < numElems; ++i) {
      const auto idx = reconstruct_index(dstBufRange, i);
      const auto received = acc[idx];
      if (is_within_window(dstCopyOffset, dstCopyRange, idx)) {
        if (!th::equal(received, expected)) {
          log_error(lh, id_helper<3>::cast(idx), received, expected);
          return;
        }
      } else {
        if (!th::equal(received, deviceCanary)) {
          log_canary_violation(lh, id_helper<3>::cast(idx), received);
          return;
        }
      }
    }
  }

  sycl::id<dim_src> getSrcCopyOffset() const { return srcCopyOffset; }
  sycl::id<dim_dst> getDstCopyOffset() const { return dstCopyOffset; }

  sycl::range<dim_src> getSrcCopyRange() const { return srcCopyRange; }
  sycl::range<dim_dst> getDstCopyRange() const { return dstCopyRange; }

  buffer_src_t getSrcBuf() const { return *srcBuf; }
  buffer_dst_t getDstBuf() const { return *dstBuf; }

  host_shared_ptr getSrcHostPtr() const { return srcHostPtr; }
  host_shared_ptr getDstHostPtr() const { return dstHostPtr; }

 private:
  sycl::queue& queue;

  const dataT hostCanary = th::make(12345);
  const dataT deviceCanary = th::make(54321);

  sycl::range<dim_src> srcBufRange = range_helper<dim_src>::make(0, 0, 0);
  sycl::range<dim_dst> dstBufRange = range_helper<dim_dst>::make(0, 0, 0);

  sycl::id<dim_src> srcCopyOffset = id_helper<dim_src>::make(0, 0, 0);
  sycl::id<dim_dst> dstCopyOffset = id_helper<dim_dst>::make(0, 0, 0);

  sycl::range<dim_src> srcCopyRange = srcBufRange;
  sycl::range<dim_dst> dstCopyRange = dstBufRange;

  std::unique_ptr<buffer_src_t> srcBuf;
  std::unique_ptr<buffer_dst_t> dstBuf;

  host_shared_ptr srcHostPtr;
  host_shared_ptr dstHostPtr;

  size_t numElems = 0;

  // Host memory region backing srcBuf,
  // used for testing handler::update_host().
  host_shared_ptr srcBufHostMemory = nullptr;
  mutable std::mutex srcBufHostMemoryMutex;

  template <int dim>
  static sycl::id<dim> reconstruct_index(sycl::range<dim> range,
                                             size_t linearIndex) {
    assert(range.size() > 0);
    const auto r3 = sycl::range<3>(range[0], dim > 1 ? range[1] : 1,
                                       dim > 2 ? range[2] : 1);
    const auto d0 = linearIndex / (r3[1] * r3[2]);
    const auto d1 = linearIndex % (r3[1] * r3[2]) / r3[2];
    const auto d2 = linearIndex % (r3[1] * r3[2]) % r3[2];
    return range_helper<dim>::make(d0, d1, d2);
  }

  template <int dim>
  static size_t compute_relative_linear_id(sycl::id<dim> offset,
                                           sycl::range<dim> range,
                                           sycl::id<dim> absIdx) {
    const auto relIdx3 = id_helper<3>::cast(absIdx - offset);
    const auto range3 = range_helper<3>::cast(range);
    const size_t relLinearIdx = relIdx3[0] * range3[1] * range3[2] +
                                relIdx3[1] * range3[2] + relIdx3[2];
    return relLinearIdx;
  }

  template <int dim>
  static bool is_within_window(sycl::id<dim> windowOffset,
                               sycl::range<dim> windowRange,
                               sycl::id<dim> idx) {
    return ((idx >= windowOffset == id_helper<dim>::make(true, true, true)) &&
            (idx < windowOffset + windowRange ==
             id_helper<dim>::make(true, true, true)));
  }

  /**
   * @brief Verifies the result of a "to device" copy (i.e., either host to
   * device or device to device).
   *
   * @tparam ExpectedValueCallback receives the relative linear index of the
   * copied element and should return the corresponding expected value.
   */
  template <typename ExpectedValueCallback>
  void verify_device_copy(ExpectedValueCallback getExpectedValue,
                          const log_helper& lh) {
    // TODO: Consider verifying directly on device.
    auto acc = dstBuf->template get_access<sycl::access_mode::read>();
    for (size_t i = 0; i < numElems; ++i) {
      const auto dstAbsIdx = reconstruct_index(dstBufRange, i);
      const auto received = acc[dstAbsIdx];
      if (is_within_window(dstCopyOffset, dstCopyRange, dstAbsIdx)) {
        // Compute relative linear index within destination copy range.
        const size_t relLinearIdx =
            compute_relative_linear_id(dstCopyOffset, dstCopyRange, dstAbsIdx);
        const auto expected = getExpectedValue(relLinearIdx);

        if (!th::equal(received, expected)) {
          log_error(lh, id_helper<3>::cast(dstAbsIdx), received, expected);
          return;
        }
      } else {
        if (!th::equal(received, deviceCanary)) {
          log_canary_violation(lh, id_helper<3>::cast(dstAbsIdx), received);
          return;
        }
      }
    }
  }

  static void log_error(const log_helper& lh, sycl::id<3> index,
                        dataT received, dataT expected) {
    std::stringstream ss;
    ss << "Unexpected value at index ";
    ss << "[" << index[0] << "," << index[1] << "," << index[2] << "]: ";
    ss << th::value(received) << " (received) != " << th::value(expected)
       << " (expected)\n";
    lh.fail(ss.str());
  }

  static void log_canary_violation(const log_helper& lh, sycl::id<3> index,
                                   dataT received) {
    std::stringstream ss;
    ss << "Canary violation at index ";
    ss << "[" << index[0] << "," << index[1] << "," << index[2] << "]: ";
    ss << "received " << th::value(received) << "\n";
    lh.fail(ss.str());
  }

  /**
   * @brief Sets up src and dst copy ranges (and offsets) in *interesting* ways.
   *
   * The basic mechanism is to first define the buffer range, copy range and
   * offset (if doing a strided copy) for the "larger" dimension out of
   * dim_src and dim_dst, where "larger" could also mean equal. Then, the
   * larger ranges and offset are "condensed" into the smaller ones, so that the
   * total number of items remains the same. For example, if the large buffer
   * range is range<3>(3,4,2), a 2-dimensional small range will become
   * range<2>(3,4*2 = 8).
   *
   * If the dimensions match, the ranges and offsets will be equal, unless
   * transposed_copy is set, in which case the destination will be transposed.
   */
  void setup_ranges() {
    constexpr auto dim_large = std::max(dim_src, dim_dst);
    constexpr auto dim_small = std::min(dim_src, dim_dst);

    auto largeBufRange = default_large_range<dim_large>();
    auto smallBufRange =
        transform_large_range_into_small<dim_large, dim_small, transposed_copy>(
            largeBufRange);

    assert(smallBufRange.size() == largeBufRange.size());

    auto largeCopyRange = largeBufRange;
    auto smallCopyRange = smallBufRange;

    auto largeCopyOffset = sycl::id<3>(0, 0, 0);
    auto smallCopyOffset = sycl::id<3>(0, 0, 0);

    // When doing a strided copy, we simply add an offset of 1 in every large
    // dimension, and reduce the copy range by 2 (resulting in an 1-item gap
    // before and after every "line" of data). For the small dimension, some
    // additional care has to be taken.
    if (strided_copy) {
      largeCopyOffset = id_helper<3>::cast(id_helper<dim_large>::make(1, 1, 1));
      smallCopyOffset = id_helper<3>::cast(id_helper<dim_large>::make(1, 1, 1));

      // We now need to compute the small offset and copy range in
      // such a way that the same total number of items will be copied.
      // For this we compute the difference in the number of items
      // copied in the (potentially) condensed dimensions.

      size_t condensedItems = 1;
      size_t offsetCondensedItems = 1;
      size_t condensedCopyCount = 1;
      for (int d = dim_small - 1; d < dim_large; ++d) {
        const auto i = transposed_copy ? dim_large - d - 1 : d;
        condensedItems *= largeBufRange[i];
        offsetCondensedItems *= largeBufRange[i] - largeCopyOffset[i];
        condensedCopyCount *= largeBufRange[i] - 2 * largeCopyOffset[i];
      }
      smallCopyOffset[dim_small - 1] = condensedItems - offsetCondensedItems;

      largeCopyRange -= range_helper<3>::cast(2 * largeCopyOffset);
      smallCopyRange -= range_helper<3>::cast(2 * smallCopyOffset);
      smallCopyRange[dim_small - 1] = condensedCopyCount;
    }

    if (dim_src > dim_dst) {
      srcBufRange = range_helper<dim_src>::cast(largeBufRange);
      dstBufRange = range_helper<dim_dst>::cast(smallBufRange);
      srcCopyOffset = id_helper<dim_src>::cast(largeCopyOffset);
      dstCopyOffset = id_helper<dim_dst>::cast(smallCopyOffset);
      srcCopyRange = range_helper<dim_src>::cast(largeCopyRange);
      dstCopyRange = range_helper<dim_dst>::cast(smallCopyRange);
    } else {
      dstBufRange = range_helper<dim_dst>::cast(largeBufRange);
      srcBufRange = range_helper<dim_src>::cast(smallBufRange);
      dstCopyOffset = id_helper<dim_dst>::cast(largeCopyOffset);
      srcCopyOffset = id_helper<dim_src>::cast(smallCopyOffset);
      dstCopyRange = range_helper<dim_dst>::cast(largeCopyRange);
      srcCopyRange = range_helper<dim_src>::cast(smallCopyRange);
    }

    numElems = srcBufRange.size();

    assert(srcBufRange.size() > 0 && dstBufRange.size() > 0);
    assert(srcBufRange.size() == dstBufRange.size());
    assert(srcCopyRange.size() > 0 && dstCopyRange.size() > 0);
    assert(srcCopyRange.size() == dstCopyRange.size());
    assert((srcCopyOffset + srcCopyRange <=
            id_helper<dim_src>::cast(srcBufRange)) ==
           id_helper<dim_src>::make(true, true, true));
    assert((dstCopyOffset + dstCopyRange <=
            id_helper<dim_dst>::cast(dstBufRange)) ==
           id_helper<dim_dst>::make(true, true, true));
  }

  template <typename test_fn>
  void run_test_function(test_fn fn, const log_helper& lh) const {
    // lh.note("Running...");  // Enable for verbose debugging output
    try {
      queue.submit([&](sycl::handler& cgh) { fn(cgh); });
      queue.wait_and_throw();
    } catch (sycl::exception&) {
      lh.fail("Exception thrown during call:");
      throw;
    }
  }
};

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This doesn't include functions that expect a write
 *        accessor.
 */
template <typename dataT, int dim, mode_t mode_src, target_t target,
          bool strided, bool transposed>
static void test_read_acc_copy_functions(log_helper lh,
                                         sycl::queue& queue) {
  lh = lh.set_mode_src(mode_src).set_target(target);
  {
    // Check copy(accessor, shared_ptr_class)
    copy_test_context<dataT, dim, dim, strided, transposed> ctx(queue);
    ctx.verify_d2h_copy(
        [&](sycl::handler& cgh) {
          auto r = ctx.getSrcBuf().template get_access<mode_src, target>(
              cgh, ctx.getSrcCopyRange(), ctx.getSrcCopyOffset());
          cgh.copy(r, ctx.getDstHostPtr());
        },
        lh.set_line(__LINE__).set_op(
            "copy(accessor<$dataT, $dim_src, $mode_src, $target>, "
            "shared_ptr_class<$dataT>)"));
  }
  {
    // Check copy(accessor, dataT*)
    copy_test_context<dataT, dim, dim, strided, transposed> ctx(queue);
    ctx.verify_d2h_copy(
        [&](sycl::handler& cgh) {
          auto r = ctx.getSrcBuf().template get_access<mode_src, target>(
              cgh, ctx.getSrcCopyRange(), ctx.getSrcCopyOffset());
          cgh.copy(r, ctx.getDstHostPtr().get());
        },
        lh.set_line(__LINE__).set_op(
            "copy(accessor<$dataT, $dim_src, $mode_src, $target>, "
            "$dataT*)"));
  }
  {
    // Check update_host(accessor)
    copy_test_context<dataT, dim, dim, strided, transposed> ctx(queue);
    ctx.verify_update_host(
        [&](sycl::handler& cgh) {
          auto r = ctx.getSrcBuf().template get_access<mode_src, target>(
              cgh, ctx.getSrcCopyRange(), ctx.getSrcCopyOffset());
          cgh.update_host(r);
        },
        lh.set_line(__LINE__).set_op(
            "update_host(accessor<$dataT, $dim_src, $mode_src, $target>)"));
  }
}

/**
 * @brief Creates lambdas with the actual tested functionality and passes them
 *        on to be tested. This includes functions that expect a write accessor.
 */
template <typename dataT, int dim_src, int dim_dst, mode_t mode_src,
          mode_t mode_dst, target_t target, bool strided, bool transposed>
static void test_write_acc_copy_functions(log_helper lh,
                                          sycl::queue& queue) {
  lh = lh.set_mode_src(mode_src).set_mode_dst(mode_dst).set_target(target);
  {
    // Check copy(shared_ptr_class, accessor)
    copy_test_context<dataT, dim_src, dim_dst, strided, transposed> ctx(queue);
    ctx.verify_h2d_copy(
        [&](sycl::handler& cgh) {
          auto w = ctx.getDstBuf().template get_access<mode_dst, target>(
              cgh, ctx.getDstCopyRange(), ctx.getDstCopyOffset());
          cgh.copy(ctx.getSrcHostPtr(), w);
        },
        lh.set_line(__LINE__).set_op(
            "copy(shared_ptr_class<$dataT>, accessor<$dataT, $dim_dst, "
            "$mode_dst, $target>)"));
  }
  {
    // Check copy(dataT*, accessor)
    copy_test_context<dataT, dim_src, dim_dst, strided, transposed> ctx(queue);
    ctx.verify_h2d_copy(
        [&](sycl::handler& cgh) {
          auto w = ctx.getDstBuf().template get_access<mode_dst, target>(
              cgh, ctx.getDstCopyRange(), ctx.getDstCopyOffset());
          cgh.copy(ctx.getSrcHostPtr().get(), w);
        },
        lh.set_line(__LINE__).set_op(
            "copy($dataT*, accessor<$dataT, $dim_dst, $mode_dst, $target>)"));
  }
  {
    // Check copy(accessor, accessor)
    copy_test_context<dataT, dim_src, dim_dst, strided, transposed> ctx(queue);
    ctx.verify_d2d_copy(
        [&](sycl::handler& cgh) {
          auto r = ctx.getSrcBuf().template get_access<mode_src, target>(
              cgh, ctx.getSrcCopyRange(), ctx.getSrcCopyOffset());
          auto w = ctx.getDstBuf().template get_access<mode_dst, target>(
              cgh, ctx.getDstCopyRange(), ctx.getDstCopyOffset());
          cgh.copy(r, w);
        },
        lh.set_line(__LINE__).set_op(
            "copy(accessor<$dataT, $dim_src, $mode_src, $target>, "
            "accessor<$dataT, $dim_dst, $mode_dst, $target>)"));
  }
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  {
    if constexpr (mode_src == mode_t::read) {
      // Check copy(accessor, accessor) with constant_buffer target
      copy_test_context<dataT, dim_src, dim_dst, strided, transposed> ctx(
          queue);
      ctx.verify_d2d_copy(
          [&](sycl::handler& cgh) {
            auto r =
                ctx.getSrcBuf()
                    .template get_access<mode_src, target_t::constant_buffer>(
                        cgh, ctx.getSrcCopyRange(), ctx.getSrcCopyOffset());
            auto w = ctx.getDstBuf().template get_access<mode_dst, target>(
                cgh, ctx.getDstCopyRange(), ctx.getDstCopyOffset());
            cgh.copy(r, w);
          },
          lh.set_line(__LINE__).set_op(
              "copy(accessor<$dataT, $dim_src, $mode_src, constant_buffer>, "
              "accessor<$dataT, $dim_dst, $mode_dst, $target>)"));
    };
  }
#endif
  {
    // Check fill(accessor, dataT)
    const auto pattern = type_helper<dataT>::make(117);
    copy_test_context<dataT, dim_src, dim_dst, strided, transposed> ctx(queue);
    ctx.verify_fill(
        [&](sycl::handler& cgh) {
          auto w = ctx.getDstBuf().template get_access<mode_dst, target>(
              cgh, ctx.getDstCopyRange(), ctx.getDstCopyOffset());
          cgh.fill(w, pattern);
        },
        pattern,
        lh.set_line(__LINE__).set_op(
            "fill(accessor<$dataT, $dim_dst, $mode_dst, $target>)"));
  }
}

/**
 * @brief Tests all valid combinations of access modes and buffer targets.
 */
template <typename dataT, int dim_src, bool strided, bool transposed>
static void test_all_read_acc_copy_functions(log_helper lh,
                                             sycl::queue& queue) {
  lh = lh.set_dim_src(dim_src);
  {
    constexpr auto target = target_t::device;
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read, target, strided,
                                 transposed>(lh, queue);
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read_write, target,
                                 strided, transposed>(lh, queue);
  }
  {
    constexpr auto target = target_t::constant_buffer;
    test_read_acc_copy_functions<dataT, dim_src, mode_t::read, target, strided,
                                 transposed>(lh, queue);
  }
}

/**
 * @brief Tests all valid combinations of source and destination access modes.
 */
template <typename dataT, int dim_src, int dim_dst, bool strided,
          bool transposed>
static void test_all_write_acc_copy_functions(log_helper lh,
                                              sycl::queue& queue) {
  lh = lh.set_dim_src(dim_src).set_dim_dst(dim_dst);
  constexpr auto target = target_t::device;

  constexpr auto st = strided;
  constexpr auto tr = transposed;

  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::write, target, st, tr>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::read_write, target, st, tr>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::discard_write, target, st, tr>(lh,
                                                                       queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read,
                                mode_t::discard_read_write, target, st, tr>(
      lh, queue);

  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::write, target, st, tr>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::read_write, target, st, tr>(lh, queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::discard_write, target, st, tr>(lh,
                                                                       queue);
  test_write_acc_copy_functions<dataT, dim_src, dim_dst, mode_t::read_write,
                                mode_t::discard_read_write, target, st, tr>(
      lh, queue);
}

/**
 * @brief Tests all valid combinations of source and destination dimensions.
 */
template <typename dataT, bool strided, bool transposed>
static void test_all_dimensions(log_helper lh, sycl::queue& queue) {
  const std::string strided_note = strided ? "strided" : "";
  const std::string transposed_note = transposed ? "transposed" : "";
  lh = lh.set_extra_info(strided_note + (strided && transposed ? ", " : "") +
                         transposed_note);

  constexpr auto st = strided;
  constexpr auto tr = transposed;

  test_all_read_acc_copy_functions<dataT, 1, st, tr>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 1, 1, st, tr>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 1, 2, st, tr>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 1, 3, st, tr>(lh, queue);

  test_all_read_acc_copy_functions<dataT, 2, st, tr>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 2, 1, st, tr>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 2, 2, st, tr>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 2, 3, st, tr>(lh, queue);

  test_all_read_acc_copy_functions<dataT, 3, st, tr>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 3, 1, st, tr>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 3, 2, st, tr>(lh, queue);
  test_all_write_acc_copy_functions<dataT, 3, 3, st, tr>(lh, queue);
}

/**
 * @brief Tests all combinations of unstrided, strided, transposed and
 *        non-transposed explicit memory operations.
 */
template <typename dataT>
static void test_all_variants(log_helper lh, sycl::queue& queue) {
  lh = lh.set_data_type<dataT>();

  test_all_dimensions<dataT, false, false>(lh, queue);
  test_all_dimensions<dataT, true, false>(lh, queue);
  test_all_dimensions<dataT, false, true>(lh, queue);
  test_all_dimensions<dataT, true, true>(lh, queue);
}

/**
 * @brief Class provides a test that checks if exception is thrown on explicit
 * memory operation copy(acc, acc) in case of destination accessor range less
 * than source accessor range
 */
template <typename DataT, typename DimSrcT, typename DimDstT, typename ModeSrcT,
          typename ModeDstT>
class CheckCopyAccToAccException {
  static constexpr int dim_src = DimSrcT::value;
  static constexpr int dim_dst = DimDstT::value;
  static constexpr mode_t mode_src = ModeSrcT::value;
  static constexpr mode_t mode_dst = ModeDstT::value;

  sycl::range<dim_src> src_copy_range = range_helper<dim_src>::make(0, 0, 0);
  sycl::range<dim_dst> dst_copy_range = range_helper<dim_dst>::make(0, 0, 0);

  void make_ranges() {
    constexpr auto dim_large = std::max(dim_src, dim_dst);
    constexpr auto dim_small = std::min(dim_src, dim_dst);

    auto large_range = default_large_range<dim_large>();
    auto small_range =
        transform_large_range_into_small<dim_large, dim_small>(large_range);

    if (dim_src > dim_dst) {
      src_copy_range = range_helper<dim_src>::cast(large_range);
      // Creating destination range less than source range to force exception on
      // explicit memory operation copy(acc, acc)
      dst_copy_range =
          range_helper<dim_dst>::cast(small_range - sycl::range<3>(1, 1, 1));
    } else {
      src_copy_range = range_helper<dim_src>::cast(small_range);
      // Creating destination range less than source range to force exception on
      // explicit memory operation copy(acc, acc)
      dst_copy_range =
          range_helper<dim_dst>::cast(large_range - sycl::range<3>(1, 1, 1));
    }
  }

  std::string description(const std::string& type_name,
                          const std::string& mode_src_name,
                          const std::string& mode_dst_name,
                          std::string&& src_target_name) {
    std::stringstream ss;
    ss << "Check that exception with error code \"errc::invalid\" is thrown on "
          "explicit memory operation copy(src_acc, dst_acc) in case of dst_acc "
          "with incorrect range size (T: "
       << type_name << " dim src: " << dim_src << "dim dist: " << dim_dst
       << " acc mode src: " << mode_src_name
       << "acc mode dst: " << mode_dst_name
       << " source target: " << src_target_name << ")";
    return ss.str();
  }

 public:
  CheckCopyAccToAccException() { make_ranges(); }
  void operator()(sycl::queue& q, const std::string& type_name,
                  const std::string&, const std::string&,
                  const std::string& mode_src_name,
                  const std::string& mode_dst_name) {
    std::shared_ptr<DataT> src_buf_mem(new DataT[src_copy_range.size()],
                                       std::default_delete<DataT[]>());
    std::shared_ptr<DataT> dst_buf_mem(new DataT[dst_copy_range.size()],
                                       std::default_delete<DataT[]>());

    sycl::buffer<DataT, dim_src> src_buf(src_buf_mem, src_copy_range);
    sycl::buffer<DataT, dim_dst> dst_buf(dst_buf_mem, dst_copy_range);

    {
      auto check_exception_with_invalid_dst_range = [&] {
        q.submit([&](sycl::handler& cgh) {
          auto src_acc =
              src_buf.template get_access<mode_src, target_t::device>(cgh);
          auto dst_acc =
              dst_buf.template get_access<mode_dst, target_t::device>(cgh);
          cgh.copy(src_acc, dst_acc);
        });
      };
      INFO(description(type_name, mode_src_name, mode_dst_name, "device"));
      CHECK_THROWS_MATCHES(
          check_exception_with_invalid_dst_range(), sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
    }

#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    if constexpr (mode_src == mode_t::read) {
      auto check_exception_with_invalid_dst_range_constant_buffer = [&] {
        q.submit([&](sycl::handler& cgh) {
          auto src_acc =
              src_buf.template get_access<mode_src, target_t::constant_buffer>(
                  cgh);
          auto dst_acc =
              dst_buf.template get_access<mode_dst, target_t::device>(cgh);
          cgh.copy(src_acc, dst_acc);
        });
      };
      INFO(description(type_name, mode_src_name, mode_dst_name,
                       "constant_buffer"));
      CHECK_THROWS_MATCHES(
          check_exception_with_invalid_dst_range_constant_buffer(),
          sycl::exception,
          sycl_cts::util::equals_exception(sycl::errc::invalid));
    }
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
  }
};
}  // namespace handler_copy_common
#endif  // __SYCLCTS_TESTS_HANDLER_COPY_COMMON_H
