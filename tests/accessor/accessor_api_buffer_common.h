/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_BUFFER_COMMON_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_BUFFER_COMMON_H

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

/** tests buffer accessors methods
*/
template <typename T, typename kernelName, int dims,
          sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
class check_buffer_accessor_api_methods {
 public:
  using acc_t = sycl::accessor<T, dims, mode, target, placeholder>;

  size_t count;
  size_t size;

  void operator()(util::logger &log, sycl::queue &queue,
                  const sycl_range_t<dims> &range,
                  const std::string& typeName) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target, placeholder>(
        "check_buffer_accessor_api_methods", typeName, log);
#endif  // VERBOSE_LOG

    auto data = get_buffer_input_data<T>(count, dims);
    buffer_t<T, dims> buffer(data.data(), range);

    // Prepare access range and access offset
    const auto accessRange = range / 2;
    const size_t accessedCount = dims == 0 ? 1 : accessRange.size();
    const size_t accessedSize = accessedCount * sizeof(T);
    auto accessOffset = sycl_id_t<dims>{};
    accessOffset[0] = accessRange[0] / 2;

    check_all_methods(log, queue, accessRange, accessOffset,
                      buffer, accessedSize, accessedCount,
                      typeName, acc_type_tag::get<target, placeholder>());
  }

 private:
  template <typename expectedT, typename returnT>
  void check_acc_return_type(sycl_cts::util::logger& log, returnT returnVal,
                            const std::string& functionName,
                            const std::string& typeName) const {
    accessor_utility::check_acc_return_type<
        expectedT, T, dims, mode, target, placeholder>(
            log, returnVal, functionName, typeName);
  }

  void check_common_methods(util::logger &log, const acc_t &accessor,
                            const size_t accessedSize,
                            const size_t accessedCount,
                            const std::string& typeName) const {
    {
      /** check is_placeholder() method
       */
      auto isPlaceholder = accessor.is_placeholder();
      check_acc_return_type<bool>(log, isPlaceholder, "is_placeholder()",
                                  typeName);
      if (isPlaceholder !=
          (placeholder == sycl::access::placeholder::true_t)) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "accessor does not properly report placeholder status");
      }
    }
    {
      /** check size() method
       */
      auto accessorCount = accessor.size();
      check_acc_return_type<size_t>(log, accessorCount, "size()",
                                    typeName);
      if (accessorCount != accessedCount) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "accessor does not return the correct count");
      }
    }
    {
      /** check get_size() method
       */
      auto accessorSize = accessor.get_size();
      check_acc_return_type<size_t>(log, accessorSize, "get_size()", typeName);
      if (accessorSize != accessedSize) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "accessor does not return the correct size");
      }
    }
    {
      /** check return type for get_pointer() method
       */
      check_acc_return_type<explicit_pointer_t<T, target>>(
          log, accessor.get_pointer(), "get_pointer()", typeName);
    }
  }

  /**
   *  @brief check get_pointer() value for host accessor
   */
  void check_get_pointer(util::logger &log, const std::string& typeName,
                         const sycl_id_t<dims> &accessOffset,
                         const acc_t &accessor) const {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target, placeholder>(
        "check_buffer_accessor_api_methods::get_pointer::host", typeName, log);
#endif  // VERBOSE_LOG
    static constexpr auto errorTarget = sycl::target::host_buffer;

    auto errors = get_error_data(2);
    {
      error_buffer_t errorBuffer(errors.data(),
                                 sycl::range<1>(errors.size()));
      auto errorAccessor =
          make_accessor<int, 1, errorMode, errorTarget, acc_placeholder::error>(
              errorBuffer);

      auto verifier =
            buffer_accessor_get_pointer<T, dims, mode, target, errorTarget,
                                        placeholder>(accessor, errorAccessor,
                                                     accessOffset);

      /** check buffer accessor pointer access
       */
      verifier();
    }

    using error_code_t = buffer_accessor_api_pointer_error_code;
    if (errors[error_code_t::pointer_read_access] != 0) {
      fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
          "accessor did not read from the correct pointer");
    }
    if (errors[error_code_t::pointer_write_access] != 0) {
      fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
          "accessor did not write to the correct pointer");
    }
  }
  /**
   *  @brief check get_pointer() value for device accessors
   *  @tparam accLambdaT Lambda to initialize placeholder on non-placeholder
   *                     accessor at command group scope
   */
  template <typename accLambdaT>
  void check_get_pointer(util::logger &log, const std::string& typeName,
                         const sycl_id_t<dims> &accessOffset,
                         sycl::queue& queue,
                         accLambdaT makeAccessor) const {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target, placeholder>(
        "check_buffer_accessor_api_methods::get_pointer::device", typeName, log);
#endif  // VERBOSE_LOG
    static constexpr auto errorTarget = sycl::target::device;

    auto errors = get_error_data(2);
    {
      error_buffer_t errorBuffer(errors.data(),
                                 sycl::range<1>(errors.size()));

      queue.submit([&](sycl::handler &cgh) {
        auto accessor = makeAccessor(cgh);
        auto errorAccessor = make_accessor<int, 1, errorMode, errorTarget,
                                           acc_placeholder::error>(
            errorBuffer, cgh);

        auto verifier =
            buffer_accessor_get_pointer<T, dims, mode, target, errorTarget,
                                        placeholder>(accessor, errorAccessor,
                                                     accessOffset);
        using kernel_name =
            buffer_accessor_get_pointer_kernel<
                kernelName, dims, mode, target, placeholder>;

        cgh.single_task<kernel_name>(verifier);
      });
    }

    using error_code_t = buffer_accessor_api_pointer_error_code;
    if (errors[error_code_t::pointer_read_access] != 0) {
      fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
          "accessor did not read from the correct pointer");
    }
    if (errors[error_code_t::pointer_write_access] != 0) {
      fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
          "accessor did not write to the correct pointer");
    }
  }

  void check_range_offset(util::logger &log,
                          const sycl_range_t<dims> &accessRange,
                          const sycl_id_t<dims> &accessOffset,
                          const acc_t &accessor,
                          const std::string& typeName) const {
    {
      /** check get_range() method
       */
      auto accessorRange = accessor.get_range();
      check_acc_return_type<sycl_range_t<dims>>(log, accessor.get_range(),
                                                "get_range()", typeName);
      if (accessorRange != accessRange) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "accessor does not return the correct range");
      }
    }
    {
      /** check get_offset() method
       */
      auto accessorOffset = accessor.get_offset();
      check_acc_return_type<sycl_id_t<dims>>(log, accessor.get_offset(),
                                             "get_offset()", typeName);
      if (accessorOffset != accessOffset) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "accessor does not return the correct offset");
      }
    }
  }

  /**
   * @brief Checks member functions where (dims != 0)
   * @param log The logger object
   * @param accessRange The range used on construction
   * @param accessOffset The offset used on construction
   * @param accessor Accessor under test
   */
  void check_methods(util::logger &log, const sycl_range_t<dims> &accessRange,
                     const sycl_id_t<dims> &accessOffset, const acc_t &accessor,
                     const size_t accessedSize, const size_t accessedCount,
                     const std::string& typeName,
                     generic_dim_tag) const {
    check_common_methods(log, accessor, accessedSize, accessedCount, typeName);
    check_range_offset(log, accessRange, accessOffset, accessor, typeName);
  }

  /**
   * @brief Checks member functions where (dims == 0)
   * @param log The logger object
   * @param accessor Accessor under test
   */
  void check_methods(util::logger &log,
                     const sycl_range_t<dims> & /*accessRange*/,
                     const sycl_id_t<dims> & /*accessOffset*/,
                     const acc_t &accessor,
                     const size_t accessedSize,
                     const size_t accessedCount,
                     const std::string& typeName,
                     zero_dim_tag) const {
    check_common_methods(log, accessor, accessedSize, accessedCount, typeName);
    // Zero-dim accessors do not provide get_range() and get_offset()
  }

  /**
   * @brief Checks member functions of accessors that can be used in kernels
   * @param log The logger object
   * @param queue SYCL queue where a kernel will be executed
   * @param accessRange The range of the accessor
   * @param accessOffset The offset of the accessor
   * @param buffer SYCL buffer used for constructing the accessor
   */
  void check_all_methods(util::logger &log, sycl::queue &queue,
                         const sycl_range_t<dims> &accessRange,
                         const sycl_id_t<dims> &accessOffset,
                         buffer_t<T, dims> &buffer,
                         const size_t accessedSize,
                         const size_t accessedCount,
                         const std::string& typeName,
                         acc_type_tag::generic) const {
    static_assert(placeholder == sycl::access::placeholder::false_t,
                  "Unexpected placeholder");
    auto make_accessor = [&](sycl::handler& cgh) -> acc_t {
      return make_accessor_generic<dims, mode, target, placeholder>(
            buffer, &accessRange, &accessOffset, &cgh);
    };

    queue.submit([&](sycl::handler &cgh) {
      auto acc = make_accessor(cgh);
      check_methods(log, accessRange, accessOffset, acc, accessedSize,
          accessedCount, typeName, is_zero_dim<dims>{});
      cgh.single_task(dummy_functor<kernelName>{});
    });
    // Pointer verification requires scope out of command group
    check_get_pointer(log, typeName, accessOffset, queue, make_accessor);
  }

  /**
   * @brief Checks member functions of placeholder accessors
   * @param log The logger object
   * @param queue SYCL queue where a kernel will be executed
   * @param accessRange The range of the accessor
   * @param accessOffset The offset of the accessor
   * @param buffer SYCL buffer used for constructing the accessor
   */
  void check_all_methods(util::logger &log, sycl::queue &queue,
                         const sycl_range_t<dims> &accessRange,
                         const sycl_id_t<dims> &accessOffset,
                         buffer_t<T, dims> &buffer,
                         const size_t accessedSize,
                         const size_t accessedCount,
                         const std::string& typeName,
                         acc_type_tag::placeholder) const {
    static_assert(placeholder == sycl::access::placeholder::true_t,
                  "Unexpected placeholder");
    auto acc = make_accessor_generic<dims, mode, target, placeholder>(
        buffer, &accessRange, &accessOffset, nullptr);

    queue.submit([&](sycl::handler &cgh) {
      cgh.require(acc);
      check_methods(log, accessRange, accessOffset, acc, accessedSize,
          accessedCount, typeName, is_zero_dim<dims>{});
      cgh.single_task(dummy_functor<kernelName>{});
    });

    // Pointer verification requires scope out of command group
    check_get_pointer(log, typeName, accessOffset, queue,
                      [&](sycl::handler& cgh) -> acc_t {
        cgh.require(acc);
        return acc;
    });
  }

  /**
   * @brief Checks member functions of host accessors
   * @param log The logger object
   * @param accessRange The range of the accessor
   * @param accessOffset The offset of the accessor
   * @param buffer SYCL buffer used for constructing the accessor
   */
  void check_all_methods(util::logger &log, sycl::queue & /*queue*/,
                         const sycl_range_t<dims> &accessRange,
                         const sycl_id_t<dims> &accessOffset,
                         buffer_t<T, dims> &buffer, const size_t accessedSize,
                         size_t accessedCount,
                         const std::string& typeName,
                         acc_type_tag::host) const {
    auto acc = make_accessor_generic<dims, mode, target, placeholder>(
        buffer, &accessRange, &accessOffset, nullptr);

    check_methods(log, accessRange, accessOffset, acc,accessedSize,
        accessedCount, typeName, is_zero_dim<dims>{});

    // Pointer verification differs for host and device accessors
    check_get_pointer(log, typeName, accessOffset, acc);
  }
};

template <typename T, typename kernelName, int dims,
          sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
class check_buffer_accessor_api {
  using acc_t = sycl::accessor<T, dims, mode, target, placeholder>;

 public:
  size_t count;
  size_t size;

  /** tests buffer accessors reads
  */
  void operator()(util::logger &log, sycl::queue &queue,
                  sycl_range_t<dims> range, const std::string& typeName,
                  acc_mode_tag::read_only) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target, placeholder>(
        "check_buffer_accessor_api::reads", typeName, log);
#endif  // VERBOSE_LOG

    auto dataIdSyntax = get_buffer_input_data<T>(count, dims);
    auto dataMultiDimSyntax = get_buffer_input_data<T>(count, dims);
    auto errors = get_error_data(2);

    {
      buffer_t<T, dims> bufIdSyntax(dataIdSyntax.data(), range);
      buffer_t<T, dims> bufMultiDimSyntax(dataMultiDimSyntax.data(), range);
      error_buffer_t errorBuffer(errors.data(),
                                 sycl::range<1>(errors.size()));

      check_command_group_read_only(queue, bufIdSyntax, bufMultiDimSyntax,
                                    errorBuffer, range,
                                    acc_type_tag::get<target, placeholder>());
    }

    using error_code_t = buffer_accessor_api_subscripts_error_code;
    if (dims == 0) {
      if (errors[error_code_t::zero_dim_access] != 0) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator dataT&() did not read from the correct index");
      }
    } else {
      if (errors[error_code_t::multi_dim_read_id] != 0) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator[id<N>] did not read from the correct index");
      }
      if (errors[error_code_t::multi_dim_read_size_t] != 0) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator[size_t][size_t][size_t] did not read from the correct "
            "index");
      }
    }
  }

 private:
  /**
   * @brief Checks reading from an accessor using subscript operators.
   *        Executed only for accessor that can be used in kernels.
   * @param queue SYCL queue where a kernel will be executed
   * @param bufIdSyntax SYCL buffer used for testing reading an accessor
   *        by passing an ID to the subscript operator
   * @param bufMultiDimSyntax SYCL buffer used for testing reading an accessor
   *        by using the multidimensional subscript operators
   * @param errorBuffer Buffer where errors will be stored
   * @param range The range of the data buffers
   */
  void check_command_group_read_only(sycl::queue &queue,
                                     buffer_t<T, dims> &bufIdSyntax,
                                     buffer_t<T, dims> &bufMultiDimSyntax,
                                     error_buffer_t &errorBuffer,
                                     sycl_range_t<dims> range,
                                     acc_type_tag::generic) {
    static_assert(placeholder == sycl::access::placeholder::false_t,
                  "Unexpected placeholder");
    queue.submit([&](sycl::handler &handler) {
      auto accIdSyntax =
          make_accessor<T, dims, mode, target, placeholder>(bufIdSyntax,
                                                            handler);
      auto accMultiDimSyntax =
          make_accessor<T, dims, mode, target, placeholder>(bufMultiDimSyntax,
                                                            handler);
      static constexpr auto errorTarget =
          sycl::target::device;
      auto errorAccessor =
          make_accessor<int, 1, errorMode, errorTarget, acc_placeholder::error>(
              errorBuffer, handler);

      using kernel_name =
          buffer_accessor_api_kernel<
              kernelName, dims, mode, target, placeholder>;

      handler.parallel_for<kernel_name>(
          range,
          buffer_accessor_api_r<T, dims, mode, target, errorTarget, placeholder>(
              size, accIdSyntax, accMultiDimSyntax, errorAccessor, range));
    });
  }

  /**
   * @brief Checks reading from a host accessor using subscript operators
   * @param bufIdSyntax SYCL buffer used for testing reading an accessor
   *        by passing an ID to the subscript operator
   * @param bufMultiDimSyntax SYCL buffer used for testing reading an accessor
   *        by using the multidimensional subscript operators
   * @param errorBuffer Buffer where errors will be stored
   * @param range The range of the data buffers
   */
  void check_command_group_read_only(sycl::queue & /*queue*/,
                                     buffer_t<T, dims> &bufIdSyntax,
                                     buffer_t<T, dims> &bufMultiDimSyntax,
                                     error_buffer_t &errorBuffer,
                                     sycl_range_t<dims> range,
                                     acc_type_tag::host) {
    static_assert(placeholder == sycl::access::placeholder::false_t,
                  "Unexpected placeholder");
    auto accIdSyntax =
        make_accessor<T, dims, mode, target, placeholder>(bufIdSyntax);
    auto accMultiDimSyntax =
        make_accessor<T, dims, mode, target, placeholder>(bufMultiDimSyntax);

    static constexpr auto errorTarget = sycl::target::host_buffer;
    auto errorAccessor =
        make_accessor<int, 1, errorMode, errorTarget, acc_placeholder::error>(
            errorBuffer);

    /** check buffer accessor subscript operators for reads
    */
    auto idList = create_id_list<data_dim<dims>::value>(range);
    for (auto id : idList) {
      buffer_accessor_api_r<T, dims, mode, target, errorTarget, placeholder>(
          size, accIdSyntax, accMultiDimSyntax, errorAccessor, range)(id);
    }
  }

  /**
   * @brief Checks reading from a placeholder accessor
   *        using subscript operators
   * @param queue SYCL queue where a kernel will be executed
   * @param bufIdSyntax SYCL buffer used for testing reading an accessor
   *        by passing an ID to the subscript operator
   * @param bufMultiDimSyntax SYCL buffer used for testing reading an accessor
   *        by using the multidimensional subscript operators
   * @param errorBuffer Buffer where errors will be stored
   * @param range The range of the data buffers
   */
  void check_command_group_read_only(sycl::queue &queue,
                                     buffer_t<T, dims> &bufIdSyntax,
                                     buffer_t<T, dims> &bufMultiDimSyntax,
                                     error_buffer_t &errorBuffer,
                                     sycl_range_t<dims> range,
                                     acc_type_tag::placeholder) {
    static_assert(placeholder == sycl::access::placeholder::true_t,
                  "Unexpected placeholder");
    auto a1 =
        sycl::accessor<T, dims, mode, target,
                           sycl::access::placeholder::true_t>(bufIdSyntax);
    auto a2 = sycl::accessor<T, dims, mode, target,
                                 sycl::access::placeholder::true_t>(
        bufMultiDimSyntax);

    queue.submit([&](sycl::handler &h) {
      h.require(a1);
      h.require(a2);

      static constexpr auto errorTarget =
          sycl::target::device;

      auto errorAccessor =
          make_accessor<int, 1, errorMode, errorTarget, acc_placeholder::error>(
              errorBuffer, h);

      const auto accSize = size;

      auto reader = buffer_accessor_api_r<T, dims, mode, target, errorTarget,
                                          placeholder>{accSize, a1, a2,
                                                       errorAccessor, range};
      using kernel_name =
          buffer_accessor_api_kernel<
              kernelName, dims, mode, target, placeholder>;

      h.parallel_for<kernel_name>(range, reader);
    });
  }

 public:
  /** tests buffer accessors writes
  */
  void operator()(util::logger &log, sycl::queue &queue,
                  sycl_range_t<dims> range, const std::string& typeName,
                  acc_mode_tag::write_only) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target, placeholder>(
        "check_buffer_accessor_api::writes", typeName, log);
#endif  // VERBOSE_LOG

    static constexpr bool useIndexes = false;
    auto dataIdSyntax = get_buffer_input_data<T>(count, dims, useIndexes);
    auto dataMultiDimSyntax = get_buffer_input_data<T>(count, dims, useIndexes);

    {
      buffer_t<T, dims> bufIdSyntax(dataIdSyntax.data(), range);
      buffer_t<T, dims> bufMultiDimSyntax(dataMultiDimSyntax.data(), range);

      check_command_group_writes(queue, bufIdSyntax, bufMultiDimSyntax, range,
                                 acc_type_tag::get<target, placeholder>());
    }

    if (dims == 0) {
      const auto expected =
          buffer_accessor_expected_value<T, dims>::expected_write();
      if (!check_elems_equal(dataIdSyntax[0], expected)) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator dataT&() did not write to the correct index");
      }
    } else {
      const auto mul = buffer_accessor_expected_value<T, dims>::write_mul;
      const auto offset = buffer_accessor_expected_value<T, dims>::write_offset;

      if (!check_linear_index(log, dataIdSyntax.data(),
                              count, mul, offset)) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator[id<N>] did not assign to the correct index");
      }
      if (!check_linear_index(log, dataMultiDimSyntax.data(),
                              count, mul, offset)) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator[size_t][size_t][size_t] did not assign to the correct "
            "index");
      }
    }
  }

 private:
  /**
   * @brief Checks writing to an accessor using subscript operators.
   *        Executed only for accessor that can be used in kernels.
   * @param queue SYCL queue where a kernel will be executed
   * @param bufIdSyntax SYCL buffer used for testing writing to an accessor
   *        by passing an ID to the subscript operator
   * @param bufMultiDimSyntax SYCL buffer used for testing writing to an
   *        accessor by using the multidimensional subscript operators
   * @param range The range of the data buffers
   */
  void check_command_group_writes(sycl::queue &queue,
                                  buffer_t<T, dims> &bufIdSyntax,
                                  buffer_t<T, dims> &bufMultiDimSyntax,
                                  sycl_range_t<dims> range,
                                  acc_type_tag::generic) {
    queue.submit([&](sycl::handler &handler) {
      auto accIdSyntax = make_accessor<T, dims, mode, target, placeholder>(
          bufIdSyntax, handler);
      auto accMultiDimSyntax =
          make_accessor<T, dims, mode, target, placeholder>(bufMultiDimSyntax,
                                                            handler);
      using kernel_name =
          buffer_accessor_api_kernel<
              kernelName, dims, mode, target, placeholder>;

      /** check buffer accessor subscript operators for writes
      */
      handler.parallel_for<kernel_name>(
          range, buffer_accessor_api_w<T, dims, mode, target, placeholder>(
                     size, accIdSyntax, accMultiDimSyntax, range));
    });
  }

  /**
   * @brief Checks writing to a host accessor using subscript operators
   * @param bufIdSyntax SYCL buffer used for testing writing to an accessor
   *        by passing an ID to the subscript operator
   * @param bufMultiDimSyntax SYCL buffer used for testing writing to an
   *        accessor by using the multidimensional subscript operators
   * @param range The range of the data buffers
   */
  void check_command_group_writes(sycl::queue & /*queue*/,
                                  buffer_t<T, dims> &bufIdSyntax,
                                  buffer_t<T, dims> &bufMultiDimSyntax,
                                  sycl_range_t<dims> range,
                                  acc_type_tag::host) {
    auto accIdSyntax =
        make_accessor<T, dims, mode, target, placeholder>(bufIdSyntax);
    auto accMultiDimSyntax =
        make_accessor<T, dims, mode, target, placeholder>(bufMultiDimSyntax);

    /** check buffer accessor subscript operators for writes
    */
    auto idList = create_id_list<data_dim<dims>::value>(range);
    for (auto id : idList) {
      buffer_accessor_api_w<T, dims, mode, target, placeholder>(
          size, accIdSyntax, accMultiDimSyntax, range)(id);
    }
  }

  /**
   * @brief Checks writing to a placeholder accessor using subscript operators
   * @param queue SYCL queue where a kernel will be executed
   * @param bufIdSyntax SYCL buffer used for testing writing to an accessor
   *        by passing an ID to the subscript operator
   * @param bufMultiDimSyntax SYCL buffer used for testing writing to an
   *        accessor by using the multidimensional subscript operators
   * @param range The range of the data buffers
   */
  void check_command_group_writes(sycl::queue &queue,
                                  buffer_t<T, dims> &bufIdSyntax,
                                  buffer_t<T, dims> &bufMultiDimSyntax,
                                  sycl_range_t<dims> range,
                                  acc_type_tag::placeholder) {
    auto a1 =
        sycl::accessor<T, dims, mode, target,
                           sycl::access::placeholder::true_t>(bufIdSyntax);
    auto a2 = sycl::accessor<T, dims, mode, target,
                                 sycl::access::placeholder::true_t>(
        bufMultiDimSyntax);

    queue.submit([&](sycl::handler &h) {
      h.require(a1);
      h.require(a2);
      auto writer =
          buffer_accessor_api_w<T, dims, mode, target, placeholder>{
              size, a1, a2, range};
      using kernel_name =
          buffer_accessor_api_kernel<
              kernelName, dims, mode, target, placeholder>;

      h.parallel_for<kernel_name>(range, writer);
    });
  }

 public:
  /** tests buffer accessors reads and writes
  */
  void operator()(util::logger &log, sycl::queue &queue,
                  sycl_range_t<dims> range, const std::string& typeName,
                  acc_mode_tag::generic) {
#ifdef VERBOSE_LOG
    log_accessor<T, dims, mode, target, placeholder>(
        "check_buffer_accessor_api::reads_and_writes", typeName, log);
#endif  // VERBOSE_LOG

    // In case of dims == 0, there will be a read from dataIdSyntax
    // and a write to dataMultiDimSyntax
    static constexpr bool useIndexesWrite = (dims > 0);
    auto dataIdSyntax = get_buffer_input_data<T>(count, dims);
    auto dataMultiDimSyntax =
        get_buffer_input_data<T>(count, dims, useIndexesWrite);

    static constexpr bool isHostBuffer =
        (target == sycl::target::host_buffer);
    auto errors = get_error_data(isHostBuffer ? 2 : 4);

    {
      buffer_t<T, dims> bufIdSyntax(dataIdSyntax.data(), range);
      buffer_t<T, dims> bufMultiDimSyntax(dataMultiDimSyntax.data(), range);
      error_buffer_t errorBuffer(errors.data(),
                                 sycl::range<1>(errors.size()));

      check_command_group_reads_writes(
          queue, bufIdSyntax, bufMultiDimSyntax, errorBuffer, range,
          acc_type_tag::get<target, placeholder>());
    }

    using error_code_t = buffer_accessor_api_subscripts_error_code;
    if (dims == 0) {
      if ((mode != sycl::access_mode::discard_read_write) &&
          (errors[error_code_t::zero_dim_access] != 0)) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator dataT&() did not read from the correct index");
      }
      const auto expected =
          buffer_accessor_expected_value<T, dims>::expected_write();
      if (!check_elems_equal(dataIdSyntax[0], expected)) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator dataT&() did not write to the correct index");
      }
    } else {
      if (mode != sycl::access_mode::discard_read_write) {
        if (errors[error_code_t::multi_dim_read_id] != 0) {
          fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator[id<N>] did not read from the correct index");
        }
        if (errors[error_code_t::multi_dim_read_size_t] != 0) {
          fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
              "operator[size_t][size_t][size_t] did not read from the "
              "correct index");
        }
      }
      const auto mul = buffer_accessor_expected_value<T, dims>::write_mul;
      const auto offset = buffer_accessor_expected_value<T, dims>::write_offset;

      if (!check_linear_index(log, dataIdSyntax.data(),
                              count, mul, offset)) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator[id<N>] did not assign to the correct index");
      }
      if (!check_linear_index(log, dataMultiDimSyntax.data(),
                              count, mul, offset)) {
        fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
            "operator[size_t][size_t][size_t] did not write to the correct "
            "index");
      }
      if (!isHostBuffer) {
        if (errors[error_code_t::multi_dim_write_id] != 0) {
          fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
              "operator[id<N>] did not write to the correct index");
        }
        if (errors[error_code_t::multi_dim_write_size_t] != 0) {
          fail_for_accessor<T, dims, mode, target, placeholder>(log, typeName,
              "operator[size_t][size_t][size_t] did not write to the correct "
              "index");
        }
      }
    }
  }

 private:
  /**
   * @brief Checks reading from and writing to an accessor
   *        using subscript operators.
   *        Executed only for accessor that can be used in kernels.
   * @param queue SYCL queue where a kernel will be executed
   * @param bufIdSyntax SYCL buffer used for testing writing to an accessor
   *        by passing an ID to the subscript operator
   * @param bufMultiDimSyntax SYCL buffer used for testing writing to an
   *        accessor by using the multidimensional subscript operators
   * @param errorBuffer Buffer where errors will be stored
   * @param range The range of the data buffers
   */
  void check_command_group_reads_writes(sycl::queue &queue,
                                        buffer_t<T, dims> &bufIdSyntax,
                                        buffer_t<T, dims> &bufMultiDimSyntax,
                                        error_buffer_t &errorBuffer,
                                        sycl_range_t<dims> range,
                                        acc_type_tag::generic) {
    static_assert(placeholder == sycl::access::placeholder::false_t,
                  "Unexpected placeholder");
    queue.submit([&](sycl::handler &handler) {
      auto accIdSyntax =
          make_accessor<T, dims, mode, target, placeholder>(bufIdSyntax,
                                                            handler);
      auto accMultiDimSyntax =
          make_accessor<T, dims, mode, target, placeholder>(bufMultiDimSyntax,
                                                            handler);
      static constexpr auto errorTarget =
          sycl::target::device;
      auto errorAccessor =
          make_accessor<int, 1, errorMode, errorTarget, acc_placeholder::error>(
              errorBuffer, handler);

      using kernel_name =
          buffer_accessor_api_kernel<
              kernelName, dims, mode, target, placeholder>;
      /** check buffer accessor subscript operators for reads and writes
      */
      handler.parallel_for<kernel_name>(
          range,
          buffer_accessor_api_rw<T, dims, mode, target, errorTarget, placeholder>(
              size, accIdSyntax, accMultiDimSyntax, errorAccessor, range));
    });
  }

  /**
   * @brief Checks reading from and writing to a host accessor
   *        using subscript operators
   * @param queue SYCL queue where a kernel will be executed
   * @param bufIdSyntax SYCL buffer used for testing writing to an accessor
   *        by passing an ID to the subscript operator
   * @param bufMultiDimSyntax SYCL buffer used for testing writing to an
   *        accessor by using the multidimensional subscript operators
   * @param errorBuffer Buffer where errors will be stored
   * @param range The range of the data buffers
   */
  void check_command_group_reads_writes(sycl::queue & /*queue*/,
                                        buffer_t<T, dims> &bufIdSyntax,
                                        buffer_t<T, dims> &bufMultiDimSyntax,
                                        error_buffer_t &errorBuffer,
                                        sycl_range_t<dims> range,
                                        acc_type_tag::host) {
    static_assert(placeholder == sycl::access::placeholder::false_t,
                  "Unexpected placeholder");
    auto accIdSyntax =
        make_accessor<T, dims, mode, target, placeholder>(bufIdSyntax);
    auto accMultiDimSyntax =
        make_accessor<T, dims, mode, target, placeholder>(bufMultiDimSyntax);
    static constexpr auto errorTarget =
          sycl::target::host_buffer;
    auto errorAccessor =
        make_accessor<int, 1, errorMode, errorTarget, acc_placeholder::error>(
            errorBuffer);

    /** check buffer accessor subscript operators for reads and writes
    */
    auto idList = create_id_list<data_dim<dims>::value>(range);
    for (auto id : idList) {
      buffer_accessor_api_rw<T, dims, mode, target, errorTarget, placeholder>(
          size, accIdSyntax, accMultiDimSyntax, errorAccessor, range)(id);
    }
  }

  /**
   * @brief Checks reading from and writing to a placeholder accessor
   *        using subscript operators
   * @param queue SYCL queue where a kernel will be executed
   * @param bufIdSyntax SYCL buffer used for testing writing to an accessor
   *        by passing an ID to the subscript operator
   * @param bufMultiDimSyntax SYCL buffer used for testing writing to an
   *        accessor by using the multidimensional subscript operators
   * @param errorBuffer Buffer where errors will be stored
   * @param range The range of the data buffers
   */
  void check_command_group_reads_writes(sycl::queue &queue,
                                        buffer_t<T, dims> &bufIdSyntax,
                                        buffer_t<T, dims> &bufMultiDimSyntax,
                                        error_buffer_t &errorBuffer,
                                        sycl_range_t<dims> range,
                                        acc_type_tag::placeholder) {
    static_assert(placeholder == sycl::access::placeholder::true_t,
                  "Unexpected placeholder");
    auto a1 =
        sycl::accessor<T, dims, mode, target,
                           sycl::access::placeholder::true_t>(bufIdSyntax);
    auto a2 = sycl::accessor<T, dims, mode, target,
                                 sycl::access::placeholder::true_t>(
        bufMultiDimSyntax);

    queue.submit([&](sycl::handler &h) {
      h.require(a1);
      h.require(a2);
      static constexpr auto errorTarget =
          sycl::target::device;

      auto errorAccessor =
          make_accessor<int, 1, errorMode, errorTarget, acc_placeholder::error>(
              errorBuffer, h);

      const auto accSize = size;

      auto reader_writer = buffer_accessor_api_rw<T, dims, mode, target,
                                                  errorTarget, placeholder>{
          accSize, a1, a2, errorAccessor, range};

      using kernel_name =
          buffer_accessor_api_kernel<
              kernelName, dims, mode, target, placeholder>;

      h.parallel_for<kernel_name>(range, reader_writer);
    });
  }
};

////////////////////////////////////////////////////////////////////////////////
// Enable tests for all combinations
////////////////////////////////////////////////////////////////////////////////

/** tests buffer accessors with different modes
*/
template <typename T, typename kernelName, int dims,
          sycl::access_mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder>
void check_buffer_accessor_api_mode(util::logger &log,
                                    const std::string& typeName,
                                    size_t count, size_t size,
                                    sycl::queue &queue,
                                    sycl_range_t<dims> range) {
#ifdef VERBOSE_LOG
  log_accessor<T, dims, mode, target, placeholder>("", typeName, log);
#endif

  /** check buffer accessor members
   */
  check_accessor_members<T, dims, mode, target, placeholder>(
      log, typeName);

  /** check buffer accessor methods
   */
  using verifier_methods =
      check_buffer_accessor_api_methods<T, kernelName, dims, mode, target,
                                        placeholder>;
  verifier_methods{count, size}(log, queue, range, typeName);

  /** check buffer accessor subscript operators
   */
  using verifier_api =
      check_buffer_accessor_api<T, kernelName, dims, mode, target, placeholder>;

  verifier_api{count, size}(
      log, queue, range, typeName, acc_mode_tag::get<mode>());
}

/**
 *  @brief Run checks with different access modes for different targets and
 *         for atomic64 or generic code path
 */
template <typename codePathT>
struct check_buffer_accessor_api_target;

using generic_path_t = sycl_cts::util::extensions::tag::generic;
using atomic64_path_t = sycl_cts::util::extensions::tag::atomic64;

/**
 *  @brief Run checks with different access modes for different targets
 *         for generic code path
 */
template <>
struct check_buffer_accessor_api_target<generic_path_t> {

  /**
   *  @brief Check global buffer accessor api for different modes except atomic
   */
  template <typename T, typename kernelName, int dims,
            sycl::target target,
            sycl::access::placeholder placeholder, typename ... argsT>
  static void run(acc_target_tag::generic, argsT&& ... args) {

    {
      constexpr auto mode = sycl::access_mode::read;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
    {
      constexpr auto mode = sycl::access_mode::write;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
    {
      constexpr auto mode = sycl::access_mode::read_write;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
    {
      constexpr auto mode = sycl::access_mode::discard_write;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
    {
      constexpr auto mode = sycl::access_mode::discard_read_write;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
  }

  /**
   *  @brief Check global buffer accessor api for all modes except atomic64 ones
   */
  template <typename T, typename kernelName, int dims,
            sycl::target target,
            sycl::access::placeholder placeholder, typename accTagT,
            typename ... argsT>
  static void run(acc_target_tag::atomic<accTagT>, argsT&& ... args) {

    // Run all except atomic checks
    run<T, kernelName, dims, target, placeholder>(accTagT{},
                                                  std::forward<argsT>(args)...);

    // Run atomic checks except atomic64 ones
    {
      constexpr auto mode = sycl::access_mode::atomic;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
  }

  /**
   *  @brief Switch off global buffer accessor api check of atomic64 modes for
   *         generic code path
   */
  template <typename T, typename kernelName, int dims,
            sycl::target target,
            sycl::access::placeholder placeholder, typename accTagT,
            typename ... argsT>
  static void run(acc_target_tag::atomic64<accTagT>,
                  util::logger &log, const std::string& typeName, argsT&& ...) {
    // Do not run atomic64 checks
#ifdef VERBOSE_LOG
    constexpr auto mode = sycl::access_mode::atomic;
    log_accessor<T, dims, mode, target, placeholder>(
        "skip_buffer_accessor_atomic64", typeName, log);
#else
    static_cast<void>(log);
    static_cast<void>(typeName);
#endif  // VERBOSE_LOG
  }

  /**
   *  @brief Check constant buffer accessor api for read
   */
  template <typename T, typename kernelName, int dims,
            sycl::target target,
            sycl::access::placeholder placeholder, typename ... argsT>
  static void run(acc_target_tag::constant, argsT&& ... args) {

    check_buffer_accessor_api_mode<T, kernelName, dims, sycl::access_mode::read,
                                   target, placeholder>(
        std::forward<argsT>(args)...);
  }

  /**
   *  @brief Check host buffer accessor api for different modes
   */
  template <typename T, typename kernelName, int dims,
            sycl::target target,
            sycl::access::placeholder placeholder, typename ... argsT>
  static void run(acc_target_tag::host, argsT&& ... args) {

    {
      constexpr auto mode = sycl::access_mode::read;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
    {
      constexpr auto mode = sycl::access_mode::write;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
    {
      constexpr auto mode = sycl::access_mode::read_write;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
    {
      constexpr auto mode = sycl::access_mode::discard_write;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
    {
      constexpr auto mode = sycl::access_mode::discard_read_write;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
  }
};

/**
 *  @brief Run checks with different access modes for different targets
 *         for atomic64 code path
 */
template <>
struct check_buffer_accessor_api_target<atomic64_path_t> {
  /**
   *  @brief Switch off accessor api check of any modes except the atomic64 ones
   */
  template <typename T, typename kernelName, int dims,
            sycl::target target,
            sycl::access::placeholder placeholder,
            typename ... argsT>
  static void run(acc_target_tag::generic, argsT&& ...) {
    // Run atomic64 checks only
  }

  /**
   *  @brief Run accessor verification for atomic64 modes only
   */
  template <typename T, typename kernelName, int dims,
            sycl::target target,
            sycl::access::placeholder placeholder, typename accTagT,
            typename ... argsT>
  static void run(acc_target_tag::atomic64<accTagT>, argsT&& ... args) {
    // Run atomic64 checks only
    {
      constexpr auto mode = sycl::access_mode::atomic;
      check_buffer_accessor_api_mode<T, kernelName, dims, mode, target,
                                     placeholder>(
          std::forward<argsT>(args)...);
    }
  }
};

/** @brief Tests buffer accessors with different targets for all types
 *         which do not require atomic64 extension
 */
template <typename T, typename kernelName, int dims,
          sycl::target target,
          sycl::access::placeholder placeholder,
          typename ... argsT>
void check_buffer_accessor_api_target_wrapper(generic_path_t,
                                              argsT&& ... args) {

  using verifier = check_buffer_accessor_api_target<generic_path_t>;

  verifier::run<T, kernelName, dims, target, placeholder>(
      acc_target_tag::get<T, target>(), std::forward<argsT>(args)...);
}
/** @brief Tests buffer accessors with different targets for all types
 *         which do require atomic64 extension
 */
template <typename T, typename kernelName, int dims,
          sycl::target target,
          sycl::access::placeholder placeholder,
          typename ... argsT>
void check_buffer_accessor_api_target_wrapper(atomic64_path_t,
                                              argsT&& ... args) {

  using verifier = check_buffer_accessor_api_target<atomic64_path_t>;

  verifier::run<T, kernelName, dims, target, placeholder>(
      acc_target_tag::get<T, target>(), std::forward<argsT>(args)...);
}

/** tests buffer accessors with different placeholder values
*/
template <typename T, typename kernelName, int dims,
          sycl::target target,
          typename ... argsT>
void check_buffer_accessor_api_placeholder(argsT&& ... args) {
  check_buffer_accessor_api_target_wrapper<T, kernelName, dims, target,
                                   sycl::access::placeholder::false_t>(
      std::forward<argsT>(args)...);

  check_buffer_accessor_api_target_wrapper<T, kernelName, dims, target,
                                   sycl::access::placeholder::true_t>(
      std::forward<argsT>(args)...);
}

/** tests buffer accessors with different dimensions
*/
template <typename T, typename kernelName, int dims, typename ... argsT>
void check_buffer_accessor_api_dim(argsT&& ... args) {
  /** check buffer accessor api for device
  */
  check_buffer_accessor_api_placeholder<
      T, kernelName, dims, sycl::target::device>(
          std::forward<argsT>(args)...);

  /** check buffer accessor api for constant_buffer
  */
  check_buffer_accessor_api_placeholder<
      T, kernelName, dims, sycl::target::constant_buffer>(
          std::forward<argsT>(args)...);

  /** check buffer accessor api for host_buffer
  */
  check_buffer_accessor_api_target_wrapper<T, kernelName, dims,
                                   sycl::target::host_buffer,
                                   sycl::access::placeholder::false_t>(
      std::forward<argsT>(args)...);
}

/** tests buffer accessors with different types
*/
template <typename T, typename extensionTagT, typename kernelName>
class check_buffer_accessor_api_type {
  static constexpr auto count = 8;
  static constexpr auto size = count * sizeof(T);

 public:
  void operator()(util::logger &log, sycl::queue &queue,
                  const std::string& typeName) {

    static const extensionTagT extensionTag;

    /** check buffer accessor api for 0 dimension
     */
    sycl::range<1> range0d(count);
    check_buffer_accessor_api_dim<T, kernelName, 0>(extensionTag, log, typeName,
                                                    count, size, queue, range0d);

    /** check buffer accessor api for 1 dimension
     */
    sycl::range<1> range1d(range0d);
    check_buffer_accessor_api_dim<T, kernelName, 1>(extensionTag, log, typeName,
                                                    count, size, queue, range1d);

    /** check buffer accessor api for 2 dimension
     */
    sycl::range<2> range2d(count / 4, 4);
    check_buffer_accessor_api_dim<T, kernelName, 2>(extensionTag, log, typeName,
                                                    count, size, queue, range2d);

    /** check buffer accessor api for 3 dimension
     */
    sycl::range<3> range3d(count / 8, 4, 2);
    check_buffer_accessor_api_dim<T, kernelName, 3>(extensionTag, log, typeName,
                                                    count, size, queue, range3d);
  }
};
}  // namespace

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_API_BUFFER_COMMON_H
