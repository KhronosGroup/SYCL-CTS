/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_LOCAL_COMMON_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_LOCAL_COMMON_H

#include "../common/common.h"
#include "./../../util/math_helper.h"
#include "accessor_utility.h"
#include "accessor_api_common_buffer_local.h"
#include "accessor_api_common_all.h"

#include <array>
#include <numeric>
#include <sstream>
#include <type_traits>

namespace {

using namespace sycl_cts;
using namespace accessor_utility;

/** unique dummy_functor per file
 */
template <typename T>
class dummy_accessor_api_local {};
template <typename T>
using dummy_functor = ::dummy_functor<dummy_accessor_api_local<T>>;

/** explicit pointer type
*/
template <typename T, cl::sycl::access::target target>
struct explicit_pointer;

/** explicit pointer type (specialization for local)
*/
template <typename T>
struct explicit_pointer<T, cl::sycl::access::target::local> {
  using type = cl::sycl::local_ptr<T>;
};

/** explicit pointer alias
 */
template <typename T, cl::sycl::access::target target>
using explicit_pointer_t = typename explicit_pointer<T, target>::type;

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

static constexpr auto mode = cl::sycl::access::mode::read_write;
static constexpr auto target = cl::sycl::access::target::local;

/** tests local accessor methods
*/
template <typename T, int dims>
class check_local_accessor_api_methods {
 public:
  size_t count;
  size_t size;

  void operator()(util::logger &log, cl::sycl::queue &queue,
                  sycl_range_t<dims> range) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target>("check_local_accessor_api_methods",
                                        log);
#endif  // VERBOSE_LOG

    queue.submit([&](cl::sycl::handler &h) {
      auto acc = make_local_accessor_generic<T, dims, mode>(range, h);
      {
        /** check get_count() method
        */
        auto accessorCount = acc.get_count();
        check_return_type<size_t>(log, accessorCount, "get_count()");
        const auto expectedCount = ((dims == 0) ? 1 : count);
        if (accessorCount != expectedCount) {
          FAIL(log, "accessor does not return the correct count");
        }
      }
      {
        /** check get_size() method
        */
        auto accessorSize = acc.get_size();
        check_return_type<size_t>(log, accessorSize, "get_size()");
        const auto expectedSize = ((dims == 0) ? sizeof(T) : size);
        if (accessorSize != expectedSize) {
          FAIL(log, "accessor does not return the correct size");
        }
      }
      {
        /** check get_pointer() method
        */
        check_return_type<explicit_pointer_t<T, target>>(log, acc.get_pointer(),
                                                         "get_pointer()");
      }

      /** check local accessor type alias
      */
      static_assert(std::is_same<
          decltype( local_accessor<T,dims>{range, h} ),
          decltype( accessor<T, dims, mode, target> {range, h} )
          >::value, "local accessor type alias check");

      /** check default value of dims parameter in local accessor type alias
      */
      if (1 == dims) {
        static_assert(std::is_same<
            decltype( local_accessor<T>{range, h} ),
            decltype( accessor<T, dims, mode, target> {range, h} )
            >::value, "default dimension local accessor type alias check");
      }

      /** dummy kernel, as no kernel is required for these checks
      */
      h.single_task(dummy_functor<T>());
    });
  }
};

/** tests local accessor reads and writes
*/
template <typename T, int dims>
class check_local_accessor_api_reads_and_writes {
 public:
  size_t count;
  size_t size;

  void operator()(util::logger &log, cl::sycl::queue &queue,
                  sycl_range_t<dims> range) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target>(
        "check_local_accessor_api_reads_and_writes", log);
#endif  // VERBOSE_LOG

    auto errors = get_error_data(4);

    static constexpr auto errorTarget = cl::sycl::access::target::global_buffer;

    {
      error_buffer_t errorBuffer(errors.data(),
                                 cl::sycl::range<1>(errors.size()));
      queue.submit([&](cl::sycl::handler &handler) {
        auto accIdSyntax =
            make_local_accessor_generic<T, dims, mode>(range, handler);
        auto accMultiDimSyntax =
            make_local_accessor_generic<T, dims, mode>(range, handler);
        auto errorAccessor =
            make_accessor<int, 1, errorMode, errorTarget>(errorBuffer, handler);
        /** check buffer accessor subscript operators for reads and writes
        */
        handler.parallel_for(
            range,
            buffer_accessor_api_rw<T, dims, mode, target, errorTarget>(
                size, accIdSyntax, accMultiDimSyntax, errorAccessor, range));
      });
    }

    if (dims == 0) {
      // Cannot check for read data
      if (errors[0] != 0) {
        FAIL(log, "operator dataT&() did not write to the correct index");
      }
    } else {
      if (errors[0] != 0) {
        FAIL(log, "operator[id<N>] did not read from the correct index");
      }
      if (errors[1] != 0) {
        FAIL(log,
             "operator[size_t][size_t][size_t] did not read from the "
             "correct index");
      }

      if (errors[2] != 0) {
        FAIL(log, "operator[id<N>] did not write to the correct index");
      }
      if (errors[3] != 0) {
        FAIL(log,
             "operator[size_t][size_t][size_t] did not write to the correct "
             "index");
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Enable tests for all combinations
////////////////////////////////////////////////////////////////////////////////

/** tests local accessors with different dimensions
*/
template <typename T, int dims>
void check_local_accessor_api_dim(util::logger &log, size_t count, size_t size,
                                  cl::sycl::queue &queue,
                                  sycl_range_t<dims> range) {
  log_accessor<T, dims, mode, target>("", log);

  /** check local accessor members
   */
  check_accessor_members<T, dims, mode, target>(log);

  /** check local accessor methods
   */
  check_local_accessor_api_methods<T, dims>{count, size}(log, queue, range);

  /** check local accessor subscript operators
   */
  check_local_accessor_api_reads_and_writes<T, dims>{count, size}(log, queue,
                                                                  range);
}

/**
*/
template <typename T>
class check_local_accessor_api_type {
  static constexpr auto count = 8;
  static constexpr auto size = count * sizeof(T);

 public:
  void operator()(util::logger &log, cl::sycl::queue &queue) {
    /** check buffer accessor api for 0 dimension
     */
    cl::sycl::range<1> range0d(count);
    check_local_accessor_api_dim<T, 0>(log, count, size, queue, range0d);

    /** check local accessor api for 1 dimension
     */
    cl::sycl::range<1> range1d(range0d);
    check_local_accessor_api_dim<T, 1>(log, count, size, queue, range1d);

    /** check local accessor api for 2 dimensions
     */
    cl::sycl::range<2> range2d(count / 4, 4);
    check_local_accessor_api_dim<T, 2>(log, count, size, queue, range2d);

    /** check local accessor api for 3 dimensions
     */
    cl::sycl::range<3> range3d(count / 8, 4, 2);
    check_local_accessor_api_dim<T, 3>(log, count, size, queue, range3d);
  }
};

}  // namespace

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_LOCAL_COMMON_H
