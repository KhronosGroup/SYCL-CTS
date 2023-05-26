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

#ifndef __SYCLCTS_TESTS_BUFFER_API_COMMON_H
#define __SYCLCTS_TESTS_BUFFER_API_COMMON_H

#include "../../../util/sycl_exceptions.h"
#include "../common/common.h"
#include "../common/get_group_range.h"
#include "../common/once_per_unit.h"

namespace buffer_api_common {
using namespace sycl_cts;

template <int dims, typename alloc>
struct write_id {};

/** empty_kernel.
 * Empty kernel, required since command groups
 * are required to have a kernel.
 */
class empty_kernel {
 public:
  void operator()() const {}
};

/*!
@brief used to calculate the ranges based on the dimensionality of the buffer
*/
template <size_t dims>
inline void precalculate(sycl::range<dims>& rangeIn,
                         sycl::range<dims>& rangeOut, size_t& elementsCount,
                         size_t elementsIn, size_t elementsOut);

template <>
inline void precalculate<1>(sycl::range<1>& rangeIn, sycl::range<1>& rangeOut,
                            size_t& elementsCount, size_t elementsIn,
                            size_t elementsOut) {
  rangeIn = sycl::range<1>(elementsIn);
  rangeOut = sycl::range<1>(elementsOut);
  elementsCount = elementsOut;
}

template <>
inline void precalculate<2>(sycl::range<2>& rangeIn, sycl::range<2>& rangeOut,
                            size_t& elementsCount, size_t elementsIn,
                            size_t elementsOut) {
  rangeIn = sycl::range<2>(elementsIn, elementsIn);
  rangeOut = sycl::range<2>(elementsOut, elementsIn);
  elementsCount = (elementsOut * elementsIn);
}

template <>
inline void precalculate<3>(sycl::range<3>& rangeIn, sycl::range<3>& rangeOut,
                            size_t& elementsCount, size_t elementsIn,
                            size_t elementsOut) {
  rangeIn = sycl::range<3>(elementsIn, elementsIn, elementsIn);
  rangeOut = sycl::range<3>(elementsOut, elementsIn, elementsIn);
  elementsCount = (elementsOut * elementsIn * elementsIn);
}

// Enable when SYCL_CTS_SUPPORT_HAS_ERRC_ENUM is defined for ComputeCPP
template <typename prop, typename buffer_t>
void check_throw_matches(buffer_t& buf, const char* prop_name) {
#if !SYCL_CTS_COMPILING_WITH_COMPUTECPP
  auto action = [&] { auto no_prop = buf.template get_property<prop>(); };
  INFO("Check that get_property() throws errc::invalid " << prop_name);
  CHECK_THROWS_MATCHES(action(), sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));
#endif
}

template <typename buffer_t>
void check_get_prop_throws(buffer_t& buf) {
  const bool use_host_ptr =
      buf.template has_property<sycl::property::buffer::use_host_ptr>();
  const bool use_mutex =
      buf.template has_property<sycl::property::buffer::use_mutex>();
  const bool context_bound =
      buf.template has_property<sycl::property::buffer::context_bound>();

  if (use_host_ptr) {
    if (use_mutex) {
      check_throw_matches<sycl::property::buffer::context_bound>(
          buf, "context_bound");
    } else if (context_bound) {
      check_throw_matches<sycl::property::buffer::use_mutex>(buf, "use_mutex");
    } else {
      check_throw_matches<sycl::property::buffer::use_mutex>(buf, "use_mutex");
      check_throw_matches<sycl::property::buffer::context_bound>(
          buf, "context_bound");
    }
  } else if (use_mutex) {
    if (context_bound) {
      check_throw_matches<sycl::property::buffer::use_host_ptr>(buf,
                                                                "use_host_ptr");
    } else {
      check_throw_matches<sycl::property::buffer::use_host_ptr>(buf,
                                                                "use_host_ptr");
      check_throw_matches<sycl::property::buffer::context_bound>(
          buf, "context_bound");
    }
  } else if (context_bound) {
    check_throw_matches<sycl::property::buffer::use_host_ptr>(buf,
                                                              "use_host_ptr");
    check_throw_matches<sycl::property::buffer::use_mutex>(buf, "use_mutex");
  }
}

/*!
@brief Used to produce and test the reinterpreted buffer denoted by the template
arguments. It does so by using the provided data array as a multidimensional
buffer
@tparam TIn the type of the original buffer
@tparam TOut the type of the reinterpreted buffer
*/
template <typename TIn, typename TOut>
class test_buffer_reinterpret {
 public:
  size_t elementsIn, elementsOut;

  template <size_t dims>
  void check(TIn* data, util::logger& log) {
    auto rangeIn = util::get_cts_object::range<dims>::get(1, 1, 1);
    auto rangeOut = util::get_cts_object::range<dims>::get(1, 1, 1);
    size_t elementsCount = 0;
    precalculate<dims>(rangeIn, rangeOut, elementsCount, elementsIn,
                       elementsOut);

    sycl::buffer<TIn, dims> a(data, rangeIn);
    auto r = a.template reinterpret<TOut, dims>(rangeOut);

    CHECK(r.byte_size() == (elementsCount * sizeof(TOut)));
  }
};

/**
 * @brief Test buffer reinterpret without specifying a range
 * @tparam TIn Underlying type of the input buffer
 * @tparam TOut Type to reinterpret to
 * @tparam inputDim Number of input dimensions
 * @tparam outputDim Number of output dimensions
 */
template <typename TIn, typename TOut, int inputDim, int outputDim>
class test_buffer_reinterpret_no_range {
 public:
  static_assert((outputDim == 1) ||
                    ((inputDim == outputDim) && (sizeof(TIn) == sizeof(TOut))),
                "Can only omit range when targetting 1D or when reinterpreting "
                "to a type of same size");

  static void check(TIn* data, const size_t inputElemsPerDim,
                    util::logger& log) {
    size_t size = sizeof(TOut) * inputElemsPerDim / sizeof(TIn);
    sycl::range<inputDim> rangeIn =
        sycl_cts::util::get_cts_object::range<inputDim>::get(size, size, size);
    sycl::buffer<TIn, inputDim> buf1(data, rangeIn);

    auto buf2 = buf1.template reinterpret<TOut>();

    CHECK(buf2.size() == buf1.size());
    CHECK(buf2.byte_size() == buf1.byte_size());
    CHECK(buf2.get_range() == buf1.get_range());
  }
};

/**
 * @brief Performs the actual check for reinterpreted buffers
 *        without providing a range, specialization for reinterpreting to 1D.
 * @tparam TIn Underlying type of the input buffer
 * @tparam TOut Type to reinterpret to
 * @tparam inputDim Number of input dimensions
 */
template <typename TIn, typename TOut, int inputDim>
class test_buffer_reinterpret_no_range<TIn, TOut, inputDim, 1> {
 public:
  static void check(TIn* data, const size_t inputElemsPerDim,
                    util::logger& log) {
    size_t size = sizeof(TOut) * inputElemsPerDim / sizeof(TIn);
    sycl::range<inputDim> rangeIn =
        sycl_cts::util::get_cts_object::range<inputDim>::get(size, size, size);
    sycl::buffer<TIn, inputDim> buf1{data, rangeIn};
    const auto expectedOutputCount = buf1.byte_size() / sizeof(TOut);

    auto buf2 = buf1.template reinterpret<TOut, 1>();

    CHECK(buf2.size() == expectedOutputCount);
    CHECK(buf2.byte_size() == buf1.byte_size());
    CHECK(buf2.get_range() == sycl::range<1>{expectedOutputCount});
  }
};

/**
 * @brief Helper class for flipping the signedness of a type.
 *
 * Required because make_signed and similar fail to instantiate
 * for types that don't have an inherent signedness.
 *
 * @tparam T Input type to have its signedness flipped
 * @tparam isIntegral Whether the type is integral or not
 */
template <class T, bool isIntegral>
struct flip_signedness_helper {
  /// Just return the same type
  using type = T;
};

/**
 * @brief Helper class for flipping the signedness of a type,
 *        specialization for integral types.
 * @tparam T
 */
template <class T>
struct flip_signedness_helper<T, true> {
  /// Make the type signed or unsigned based on the input type
  using type =
      typename std::conditional<std::is_signed<T>::value,
                                typename std::make_unsigned<T>::type,
                                typename std::make_signed<T>::type>::type;
};

/**
 * @brief Flips the signedness of the data type, if possible.
 *        Otherwise returns the same type.
 * @tparam T Input type to have its signedness flipped
 */
template <class T>
using flip_signedness_t =
    typename flip_signedness_helper<T, std::is_integral<T>::value>::type;

template <typename T, int size, int dims, typename alloc,
          sycl::access_mode access_mode, sycl::target target>
class kernel_buffer_accessor_type;

/**
 * Generic buffer API test function
 */
template <typename T, int size, int dims, typename alloc>
void test_buffer(util::logger& log, sycl::range<dims>& r, sycl::id<dims>& i) {
  {
    std::unique_ptr<T[]> data(new T[size]);
    std::fill(data.get(), (data.get() + size), 0);

    // Create a default offset with indices 0.
    sycl::id<dims> offset;

    /* create a SYCL buffer from the host buffer */
    sycl::buffer<T, dims, alloc> buf(data.get(), r);

    /* check the buffer returns a range */
    auto ret_range = buf.get_range();
    check_return_type<sycl::range<dims>>(log, ret_range,
                                         "sycl::buffer::get_range()");

    /* Check alias types */
    {
      { check_type_existence<typename sycl::buffer<T, dims>::value_type>(); }
      { check_type_existence<typename sycl::buffer<T, dims>::reference>(); }
      {
        check_type_existence<typename sycl::buffer<T, dims>::const_reference>();
      }
      {
        check_type_existence<typename sycl::buffer<
            T, dims, sycl::buffer_allocator<T>>::allocator_type>();
      }
    }

    /* Check that ret_range is the correct size */
    for (int i = 0; i < dims; ++i) {
      CHECK(ret_range[i] == r[i]);
    }

    /* check the buffer returns the correct element count */
    auto count = buf.size();
    check_return_type<size_t>(log, count, "sycl::buffer::size()");

    CHECK(count == size);

    /* check the buffer returns the correct element count
       with deprecated get_count */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    auto count_depr = buf.get_count();
    check_return_type<size_t>(log, count_depr, "sycl::buffer::get_count()");

    CHECK(count_depr == size);
#endif  // SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS

    /* check the buffer returns the correct byte size */
    auto ret_size = buf.byte_size();
    check_return_type<size_t>(log, ret_size, "sycl::buffer::byte_size()");

    CHECK(ret_size == size * sizeof(T));

    /* check the buffer returns the correct byte size
     with deprecated get_size*/
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    auto ret_size_depr = buf.get_size();
    check_return_type<size_t>(log, ret_size_depr, "sycl::buffer::get_size()");

    CHECK(ret_size_depr == size * sizeof(T));
#endif

    auto q = once_per_unit::get_queue();

    /* check the buffer returns the correct type of accessor */
    q.submit([&](sycl::handler& cgh) {
      using kname = kernel_buffer_accessor_type<T, size, dims, alloc,
                                                sycl::access_mode::read_write,
                                                sycl::target::device>;
      auto acc = buf.template get_access<sycl::access_mode::read_write>(cgh);
      check_return_type<sycl::accessor<T, dims, sycl::access_mode::read_write,
                                       sycl::target::device>>(
          log, acc, "sycl::buffer::get_access<read_write>(handler&)");
      cgh.single_task<kname>(empty_kernel());
    });

    /* check the buffer returns the correct type of accessor */
    q.submit([&](sycl::handler& cgh) {
      using kname = kernel_buffer_accessor_type<T, size, dims, alloc,
                                                sycl::access_mode::read,
                                                sycl::target::constant_buffer>;
      auto acc = buf.template get_access<sycl::access_mode::read,
                                         sycl::target::constant_buffer>(cgh);
      check_return_type<sycl::accessor<T, dims, sycl::access_mode::read,
                                       sycl::target::constant_buffer>>(
          log, acc,
          "sycl::buffer::get_access<read, constant_buffer>(handler&)");
      cgh.single_task<kname>(empty_kernel());
    });

    /* check the buffer returns the correct type of accessor */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    {
      auto acc = buf.template get_access<sycl::access_mode::read_write>();
      check_return_type<sycl::accessor<T, dims, sycl::access_mode::read_write,
                                       sycl::target::host_buffer>>(
          log, acc, "sycl::buffer::get_access<read_write, host_buffer>()");
    }
#endif

    /* check the buffer returns the correct type of accessor */
    q.submit([&](sycl::handler& cgh) {
      using kname = kernel_buffer_accessor_type<T, size, dims, alloc,
                                                sycl::access_mode::read_write,
                                                sycl::target::device>;
      auto acc = buf.template get_access<sycl::access_mode::read_write>(cgh, r,
                                                                        offset);
      check_return_type<sycl::accessor<T, dims, sycl::access_mode::read_write,
                                       sycl::target::device>>(
          log, acc,
          "sycl::buffer::get_access<read_write, device>(handler&, "
          "range<>, id<>)");
      cgh.single_task<kname>(empty_kernel());
    });

    /* check the buffer returns the correct type of accessor */
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS
    {
      auto acc =
          buf.template get_access<sycl::access_mode::read_write>(r, offset);
      check_return_type<sycl::accessor<T, dims, sycl::access_mode::read_write,
                                       sycl::target::host_buffer>>(
          log, acc,
          "sycl::buffer::get_access<read_write, host_buffer>(range<>, "
          "id<>)");
    }
#endif

    /* check get_allocator() */
    {
      sycl::buffer<T, dims, alloc> bufAlloc(data.get(), r);

      auto allocator = bufAlloc.get_allocator();

      check_return_type<alloc>(log, allocator, "get_allocator()");

      auto ptr = allocator.allocate(1);
      CHECK(ptr != nullptr);
      allocator.deallocate(ptr, 1);
    }

    /* check is_sub_buffer() */
    {
      sycl::buffer<T, dims> buf(r);
      sycl::range<dims> sub_r = r;
      sub_r[0] = r[0] - i[0];
      sycl::buffer<T, dims> buf_sub(buf, i, sub_r);
      auto isSubBuffer = buf_sub.is_sub_buffer();
      auto isOrigSubBuffer = buf.is_sub_buffer();
      check_return_type<bool>(log, isSubBuffer, "is_sub_buffer()");
      CHECK(isSubBuffer);
      CHECK_FALSE(isOrigSubBuffer);
    }

    /* check buffer properties */
    {
      std::mutex mutex;
      auto context = util::get_cts_object::context();
      const sycl::property_list pl{
          sycl::property::buffer::use_mutex(mutex),
          sycl::property::buffer::context_bound(context)};

      sycl::buffer<T, dims> buf(r, pl);

      /* check has_property() */

      auto hasUseMutexProperty =
          buf.template has_property<sycl::property::buffer::use_mutex>();
      check_return_type<bool>(log, hasUseMutexProperty,
                              "has_property<use_mutex>()");
      CHECK(hasUseMutexProperty);

      auto hasContentBoundProperty =
          buf.template has_property<sycl::property::buffer::context_bound>();
      check_return_type<bool>(log, hasContentBoundProperty,
                              "has_property<context_bound>()");
      CHECK(hasContentBoundProperty);

      /* check get_property() */

      auto useMutexProperty =
          buf.template get_property<sycl::property::buffer::use_mutex>();
      check_return_type<sycl::property::buffer::use_mutex>(
          log, useMutexProperty, "get_property<use_mutex>()");
      check_return_type<std::mutex*>(
          log, useMutexProperty.get_mutex_ptr(),
          "sycl::property::buffer::use_mutex::get_mutex_ptr()");

      auto contentBoundProperty =
          buf.template get_property<sycl::property::buffer::context_bound>();
      check_return_type<sycl::property::buffer::context_bound>(
          log, contentBoundProperty, "get_property<context_bound>()");
      check_return_type<sycl::context>(
          log, contentBoundProperty.get_context(),
          "sycl::property::buffer::context_bound::get_context()");

      /* Check that get_property() throws an exception with the errc::invalid
       * error code if buffer was not constructed with use_host_ptr property
       */
      {
        sycl::buffer<T, dims> buf_host_ptr(
            data.get(), r, {sycl::property::buffer::use_host_ptr()});
        sycl::buffer<T, dims> buf_mutex(
            data.get(), r, {sycl::property::buffer::use_mutex(mutex)});
        sycl::buffer<T, dims> buf_context(
            data.get(), r, {sycl::property::buffer::context_bound(context)});

        sycl::buffer<T, dims> buf_host_ptr_mutex(
            data.get(), r,
            {sycl::property::buffer::use_host_ptr(),
             sycl::property::buffer::use_mutex(mutex)});
        sycl::buffer<T, dims> buf_host_ptr_context(
            data.get(), r,
            {sycl::property::buffer::use_host_ptr(),
             sycl::property::buffer::context_bound(context)});
        sycl::buffer<T, dims> buf_mutex_context(
            data.get(), r,
            {sycl::property::buffer::use_mutex(mutex),
             sycl::property::buffer::context_bound(context)});

        check_get_prop_throws(buf_host_ptr);
        check_get_prop_throws(buf_mutex);
        check_get_prop_throws(buf_context);
        check_get_prop_throws(buf_host_ptr_mutex);
        check_get_prop_throws(buf_host_ptr_context);
        check_get_prop_throws(buf_mutex_context);
      }
    }

    q.wait_and_throw();
  }
}

/**
 * @brief Tests reinterpreting a buffer
 * @tparam T Underlying data type of the input buffer
 * @tparam numDims Number of input dimensions
 * @param log Logger object
 */
template <typename T, int numDims>
void test_type_reinterpret(util::logger& log) {
  static constexpr size_t inputElemsPerDim = 4;
  static constexpr size_t numElems = [](size_t elemsPerDim) {
    for (int i = 1; i < numDims; ++i) elemsPerDim *= elemsPerDim;
    return elemsPerDim;
  }(inputElemsPerDim);
  alignas(alignof(T)) unsigned char reinterpretInputData[sizeof(T) * numElems];
  using ReinterpretT = flip_signedness_t<T>;

  // Check reinterpreting with a range
  test_buffer_reinterpret<unsigned char, T>{sizeof(T) * numElems, numElems}
      .template check<numDims>(reinterpretInputData, log);

  // Check reinterpreting without a range to 1D
  test_buffer_reinterpret_no_range<unsigned char, T, numDims, 1>::check(
      reinterpretInputData, inputElemsPerDim, log);

  // Check reinterpreting without a range to the same dimension
  test_buffer_reinterpret_no_range<T, ReinterpretT, numDims, numDims>::check(
      reinterpret_cast<T*>(reinterpretInputData), inputElemsPerDim, log);
}

template <typename T>
class check_buffer_api_for_type {
  template <typename alloc>
  void check_with_alloc(util::logger& log) {
    const int size = 8;
    sycl::range<1> range1d(size);
    sycl::range<2> range2d(size, size);
    sycl::range<3> range3d(size, size, size);

    sycl::id<1> id1d(2);
    sycl::id<2> id2d(2, 0);
    sycl::id<3> id3d(2, 0, 0);

    test_buffer<T, size, 1, alloc>(log, range1d, id1d);
    test_buffer<T, size * size, 2, alloc>(log, range2d, id2d);
    test_buffer<T, size * size * size, 3, alloc>(log, range3d, id3d);

    /* check reinterpret() */
    test_type_reinterpret<T, 1>(log);
    test_type_reinterpret<T, 2>(log);
    test_type_reinterpret<T, 3>(log);
  }

 public:
  void operator()(util::logger& log, const std::string& typeName) {
    INFO("testing: " + type_name_string<T>::get(typeName));
    check_with_alloc<sycl::buffer_allocator<T>>(log);
    check_with_alloc<std::allocator<T>>(log);
  }
};

/**
 * @brief Test buffer linearization
 */
class check_buffer_linearization {
 public:
  void operator()(util::logger& log) {
    constexpr int g_size = 4;  // global range size
    constexpr int l_size = 2;  // local range size
    auto q = once_per_unit::get_queue();

    // global ranges
    sycl::range<1> g_range1d = sycl_cts::util::work_group_range<1>(q, g_size);
    sycl::range<2> g_range2d =
        sycl_cts::util::work_group_range<2>(q, g_size * g_size);
    sycl::range<3> g_range3d =
        sycl_cts::util::work_group_range<3>(q, g_size * g_size * g_size);

    // local ranges
    sycl::range<1> l_range1d = sycl_cts::util::work_group_range<1>(q, l_size);
    sycl::range<2> l_range2d =
        sycl_cts::util::work_group_range<2>(q, l_size * l_size);
    sycl::range<3> l_range3d =
        sycl_cts::util::work_group_range<3>(q, l_size * l_size * l_size);

    sycl::nd_range<1> nd_range1d(g_range1d, l_range1d);
    sycl::nd_range<2> nd_range2d(g_range2d, l_range2d);
    sycl::nd_range<3> nd_range3d(g_range3d, l_range3d);

    INFO("testing: sycl::buffer_allocator<size_t>");
    test_buffer_linearization<1, sycl::buffer_allocator<size_t>>(log,
                                                                 nd_range1d);
    test_buffer_linearization<2, sycl::buffer_allocator<size_t>>(log,
                                                                 nd_range2d);
    test_buffer_linearization<3, sycl::buffer_allocator<size_t>>(log,
                                                                 nd_range3d);

    INFO("testing: std::allocator<size_t>");
    test_buffer_linearization<1, std::allocator<size_t>>(log, nd_range1d);
    test_buffer_linearization<2, std::allocator<size_t>>(log, nd_range2d);
    test_buffer_linearization<3, std::allocator<size_t>>(log, nd_range3d);
  }

 private:
  /**
   * Buffer linearization test
   */
  template <int dims, typename alloc>
  void test_buffer_linearization(util::logger& log, sycl::nd_range<dims>& r) {
    static_assert(dims >= 1 && dims < 4,
                  "Linearization test requires dims to be one of {1;2;3}.");
    INFO("testing: linearization in " + std::to_string(dims) + " dimensions.");
    auto q = once_per_unit::get_queue();

    sycl::buffer<size_t, dims, alloc> buf(r.get_global_range());
    q.submit([&](sycl::handler& cgh) {
      auto acc = buf.get_access(cgh, sycl::write_only, sycl::no_init);
      cgh.parallel_for<write_id<dims, alloc>>(r, [=](sycl::nd_item<dims> i) {
        // clang-format off
        if constexpr (dims == 3) {
          acc[i.get_global_id()] =
              i.get_global_id(0) * i.get_global_range(1) * i.get_global_range(2) +
              i.get_global_id(1) * i.get_global_range(2) +
              i.get_global_id(2);
        }
        else if (dims == 2) {
          acc[i.get_global_id()] =
              i.get_global_id(0) * i.get_global_range(1) +
              i.get_global_id(1);
        }
        else {
          acc[i.get_global_id()] =
              i.get_global_id(0);
        }
        // clang-format on
      });
    });

    std::vector<size_t> v(buf.size());
    std::iota(v.begin(), v.end(), 0);

    std::vector<size_t> w(buf.size());
    q.copy(sycl::accessor{buf}, w.data()).wait_and_throw();

    CHECK(std::equal(v.cbegin(), v.cend(), w.cbegin()));
  }
};

}  // namespace buffer_api_common
#endif  // __SYCLCTS_TESTS_BUFFER_API_COMMON_H
