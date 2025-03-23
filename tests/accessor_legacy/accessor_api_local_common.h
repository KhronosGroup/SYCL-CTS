/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
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
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_LOCAL_COMMON_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_LOCAL_COMMON_H

#include "../common/common.h"
#include "./../../util/extensions.h"
#include "./../../util/math_helper.h"
#include "accessor_api_common_all.h"
#include "accessor_api_common_buffer_local.h"
#include "accessor_api_utility.h"

#include <utility>

namespace {

using namespace sycl_cts;
using namespace accessor_utility;

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

static constexpr auto target = sycl::target::local;

/** tests local accessor methods
*/
template <typename T, typename kernelName, int dims,
          sycl::access_mode mode>
class check_local_accessor_api_methods {
 public:
  size_t count;
  size_t size;

  using acc_t = decltype(make_local_accessor_generic<T, dims, mode>(
      std::declval<sycl_range_t<dims>>(), std::declval<sycl::handler &>()));

  void operator()(util::logger &log, sycl::queue &queue,
                  sycl_range_t<dims> range, const std::string& typeName) {
#if SYCL_CTS_ENABLE_VERBOSE_LOG
    log_accessor<T, dims, mode, target>("check_local_accessor_api_methods",
                                        typeName, log);
#endif  // SYCL_CTS_ENABLE_VERBOSE_LOG
    static constexpr auto errorTarget = sycl::target::device;

    auto errors = get_error_data(2);
    {
      auto kernelRange =
          util::get_cts_object::range<data_dim<dims>::value>::get(1, 1, 1);
      error_buffer_t errorBuffer(errors.get(), sycl::range<1>(2));

      queue.submit([&](sycl::handler& h) {
        auto acc = make_local_accessor_generic<T, dims, mode>(range, h);
        {
          /** check get_count() method
           */
          // TODO: mark this check as testing deprecated functionality
          auto accessorCount = acc.get_count();
          check_return_type<size_t>(log, accessorCount, "get_count()");
          const auto expectedCount = ((dims == 0) ? 1 : count);
          if (accessorCount != expectedCount) {
            fail_for_accessor<T, dims, mode, target>(
                log, typeName, "accessor does not return the correct count");
          }
        }
        {
          /** check get_size() method
          */
          auto accessorSize = acc.get_size();
          check_return_type<size_t>(log, accessorSize, "get_size()");
          const auto expectedSize = ((dims == 0) ? sizeof(T) : size);
          if (accessorSize != expectedSize) {
            fail_for_accessor<T, dims, mode, target>(log, typeName,
                "accessor does not return the correct size");
          }
        }
        check_get_range(log, acc, range, typeName, is_zero_dim<dims>{});
        if constexpr (target == sycl::access::target::host_buffer) {
          /** check get_pointer() method for deprecated accessor targets
           */
          auto pointer = acc.get_pointer();
          check_return_type<explicit_pointer_t<T, mode, target>>(
              log, pointer, "get_pointer()");
          if (pointer == nullptr) {
            fail_for_accessor<T, dims, mode, target>(log, typeName,
                "accessor does not return the correct pointer");
          }
        }
        /** dummy kernel, as no kernel is required for these checks
        */
        auto errorAccessor = make_accessor<int, 1, errorMode, errorTarget,
                                           acc_placeholder::error>(
            errorBuffer, h);
        constexpr auto dataDims = data_dim<dims>::value;
        auto accessOffset =
            sycl_cts::util::get_cts_object::id<dataDims>::get(0, 0, 0);
        auto verifier =
            buffer_accessor_get_pointer<T, dims, mode, target, errorTarget,
                                        acc_placeholder::local>(
                acc, errorAccessor, accessOffset);
        using kernel_name =
            buffer_accessor_get_pointer_kernel<
                kernelName, dims, mode, target, acc_placeholder::local>;

        h.parallel_for<kernel_name>(
            sycl::nd_range<data_dim<dims>::value>(kernelRange, kernelRange),
            verifier);
      });
    }

    using error_code_t = buffer_accessor_api_pointer_error_code;
    if (errors.get()[error_code_t::pointer_read_access] != 0) {
      fail_for_accessor<T, dims, mode, target>(log, typeName,
          "accessor did not read from the correct pointer");
    }
    if (errors.get()[error_code_t::pointer_write_access] != 0) {
      fail_for_accessor<T, dims, mode, target>(log, typeName,
          "accessor did not write to the correct pointer");
    }
  }

private:
  void check_get_range(util::logger &, acc_t, sycl_range_t<dims>,
                       const std::string&, zero_dim_tag) {
    // Not available with 0 dimensions
  }

  void check_get_range(util::logger &log, acc_t acc, sycl_range_t<dims> range,
                       const std::string& typeName, generic_dim_tag) {
    // check get_range() method
    auto accessorRange = acc.get_range();
    check_return_type<sycl_range_t<dims>>(log, accessorRange, "get_range()");
    if (accessorRange != range) {
      fail_for_accessor<T, dims, mode, target>(log, typeName,
          "accessor does not return the correct range");
    }
  }
};

/** tests local accessor reads and writes
*/
template <typename T, typename kernelName, int dims,
          sycl::access_mode mode>
class check_local_accessor_api_reads_and_writes {
 public:
  size_t count;
  size_t size;

  void operator()(util::logger &log, sycl::queue &queue,
                  sycl_range_t<dims> range, const std::string& typeName) {
#if SYCL_CTS_ENABLE_VERBOSE_LOG
    log_accessor<T, dims, mode, target>(
        "check_local_accessor_api_reads_and_writes", typeName, log);
#endif  // SYCL_CTS_ENABLE_VERBOSE_LOG

    auto errors = get_error_data(4);

    static constexpr auto errorTarget = sycl::target::device;

    {
      error_buffer_t errorBuffer(errors.get(), sycl::range<1>(4));
      queue.submit([&](sycl::handler &handler) {
        auto accIdSyntax =
            make_local_accessor_generic<T, dims, mode>(
                range, handler);
        auto accMultiDimSyntax =
            make_local_accessor_generic<T, dims, mode>(
                range, handler);
        auto errorAccessor = make_accessor<int, 1, errorMode, errorTarget,
                                           acc_placeholder::error>(
            errorBuffer, handler);
        /** check buffer accessor subscript operators for reads and writes
        */
        using kernel_name =
          buffer_accessor_api_kernel<
              kernelName, dims, mode, target, acc_placeholder::local>;

        handler.parallel_for<kernel_name>(
            sycl::nd_range<data_dim<dims>::value>(range, range),
            buffer_accessor_api_rw<T, dims, mode, target, errorTarget,
                                   acc_placeholder::local>(
                size, accIdSyntax, accMultiDimSyntax, errorAccessor, range));
      });
    }

    using error_code_t = buffer_accessor_api_subscripts_error_code;
    if (dims == 0) {
      // Cannot check for read data
      if (errors.get()[error_code_t::zero_dim_access] != 0) {
        fail_for_accessor<T, dims, mode, target>(log, typeName,
            "operator dataT&() did not write to the correct index");
      }
    } else {
      if (errors.get()[error_code_t::multi_dim_read_id] != 0) {
        fail_for_accessor<T, dims, mode, target>(log, typeName,
            "operator[id<N>] did not read from the correct index");
      }
      if (errors.get()[error_code_t::multi_dim_read_size_t] != 0) {
        fail_for_accessor<T, dims, mode, target>(log, typeName,
            "operator[size_t][size_t][size_t] did not read from the "
            "correct index");
      }

      if (errors.get()[error_code_t::multi_dim_write_id] != 0) {
        fail_for_accessor<T, dims, mode, target>(log, typeName,
            "operator[id<N>] did not write to the correct index");
      }
      if (errors.get()[error_code_t::multi_dim_write_size_t] != 0) {
        fail_for_accessor<T, dims, mode, target>(log, typeName,
            "operator[size_t][size_t][size_t] did not write to the correct "
            "index");
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Enable tests for all combinations
////////////////////////////////////////////////////////////////////////////////

/** @brief tests local accessors with different dimensions
*/
template <typename T, typename kernelName, int dims,
          sycl::access_mode mode>
void check_local_accessor_api_mode(util::logger &log,
                                   const std::string& typeName,
                                   size_t count, size_t size,
                                   sycl::queue &queue,
                                   sycl_range_t<dims> range) {
#if SYCL_CTS_ENABLE_VERBOSE_LOG
  log_accessor<T, dims, mode, target>("", typeName, log);
#endif

  /** check local accessor members
   */
  check_accessor_members<T, dims, mode, target>(log, typeName);

  /** check local accessor methods
   */
  using verifier_methods =
      check_local_accessor_api_methods<T, kernelName, dims, mode>;

  verifier_methods{count, size}(log, queue, range, typeName);

  /** check local accessor subscript operators
   */
  using verifier_api =
      check_local_accessor_api_reads_and_writes<T, kernelName, dims, mode>;

  verifier_api{count, size}( log, queue, range, typeName);
}

/**
 *  @brief Run checks with different access modes for different dims and
 *         for atomic64 or generic code path
 */
template <typename codePathT>
struct check_local_accessor_api_dim;

using generic_path_t = sycl_cts::util::extensions::tag::generic;
using atomic64_path_t = sycl_cts::util::extensions::tag::atomic64;

/**
 *  @brief Run checks with different access modes for different dims
 *         for generic code path
 */
template <>
struct check_local_accessor_api_dim<generic_path_t> {

  /**
   *  @brief check local buffer accessor api for different modes except atomic
   */
  template <typename T, typename kernelName, int dims, typename ... argsT>
  static void run(acc_target_tag::local,
                  argsT&& ... args) {
    // Run verification for read_write access mode
    constexpr auto mode = sycl::access_mode::read_write;
    check_local_accessor_api_mode<T, kernelName, dims, mode>(
        std::forward<argsT>(args)...);
  }

  /**
   *  @brief Check local accessor api for all modes except atomic64 ones
   */
  template <typename T, typename kernelName, int dims, typename accTagT,
            typename ... argsT>
  static void run(acc_target_tag::atomic<accTagT>, argsT&& ... args) {
    // Run verification for read_write access mode
    {
      constexpr auto mode = sycl::access_mode::read_write;
      check_local_accessor_api_mode<T, kernelName, dims, mode>(
          std::forward<argsT>(args)...);
    }
    // Run verification for atomic access mode
    {
      constexpr auto mode = sycl::access_mode::atomic;
      check_local_accessor_api_mode<T, kernelName, dims, mode>(
          std::forward<argsT>(args)...);
    }
  }

  /**
   *  @brief Switch off local accessor api check of atomic64 modes for generic
   *         code path
   */
  template <typename T, typename kernelName, int dims, typename accTagT,
            typename ... argsT>
  static void run(acc_target_tag::atomic64<accTagT>,
                  util::logger &log, const std::string& typeName, argsT&& ...) {
    // Do not run atomic64 checks
#if SYCL_CTS_ENABLE_VERBOSE_LOG
    constexpr auto mode = sycl::access_mode::atomic;
    log_accessor<T, kernelName, dims, mode, target>(
        "skip_local_accessor_atomic64", typeName, log);
#else
    static_cast<void>(log);
    static_cast<void>(typeName);
#endif  // SYCL_CTS_ENABLE_VERBOSE_LOG
  }
};

/**
 *  @brief Run checks with different access modes for different dims
 *         for atomic64 code path
 */
template <>
struct check_local_accessor_api_dim<atomic64_path_t> {
  /**
   *  @brief Switch off accessor api check of any modes except the atomic64 ones
   */
  template <typename T, typename kernelName, int dims, typename ... argsT>
  static void run(acc_target_tag::generic, argsT&& ...) {
    // Run atomic64 checks only
  }
  /**
   *  @brief Run local accessor verification for atomic64 modes only
   */
  template <typename T, typename kernelName, int dims, typename accTagT,
            typename ... argsT>
  static void run(acc_target_tag::atomic64<accTagT>,
                  argsT&& ... args) {
    // Run atomic64 checks only
    {
      constexpr auto mode = sycl::access_mode::atomic;
      check_local_accessor_api_mode<T, kernelName, dims, mode>(
          std::forward<argsT>(args)...);
    }
  }
};

/** @brief Tests local accessors with different dims for all types
 *         which do not require atomic64 extension
 */
template <typename T, typename kernelName, int dims, typename ... argsT>
void check_local_accessor_api_dim_wrapper(generic_path_t, argsT&& ... args) {

  using verifier = check_local_accessor_api_dim<generic_path_t>;

  verifier::run<T, kernelName, dims>(acc_target_tag::get<T, target>(),
                                     std::forward<argsT>(args)...);
}
/** @brief Tests local accessors with different targets for all types
 *         which do require atomic64 extension
 */
template <typename T, typename kernelName, int dims, typename ... argsT>
void check_local_accessor_api_dim_wrapper(atomic64_path_t, argsT&& ... args) {

  using verifier = check_local_accessor_api_dim<atomic64_path_t>;

  verifier::run<T, kernelName, dims>(acc_target_tag::get<T, target>(),
                                     std::forward<argsT>(args)...);
}

/** @brief tests local accessors for different types
*/
template <typename T, typename extensionTagT, typename kernelName>
class check_local_accessor_api_type {
  static constexpr auto count = 8;
  static constexpr auto size = count * sizeof(T);

 public:
  void operator()(util::logger &log, sycl::queue &queue,
                  const std::string& typeName) {

    static const extensionTagT extensionTag;

    /** check buffer accessor api for 0 dimension
     */
    sycl::range<1> range0d(count);
    check_local_accessor_api_dim_wrapper<T, kernelName, 0>(
        extensionTag, log, typeName, count, size, queue, range0d);

    /** check local accessor api for 1 dimension
     */
    sycl::range<1> range1d(range0d);
    check_local_accessor_api_dim_wrapper<T, kernelName, 1>(
        extensionTag, log, typeName, count, size, queue, range1d);

    /** check local accessor api for 2 dimensions
     */
    sycl::range<2> range2d(count / 4, 4);
    check_local_accessor_api_dim_wrapper<T, kernelName, 2>(
        extensionTag, log, typeName, count, size, queue, range2d);

    /** check local accessor api for 3 dimensions
     */
    sycl::range<3> range3d(count / 8, 4, 2);
    check_local_accessor_api_dim_wrapper<T, kernelName, 3>(
        extensionTag, log, typeName, count, size, queue, range3d);
  }
};

}  // namespace

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_LOCAL_COMMON_H
