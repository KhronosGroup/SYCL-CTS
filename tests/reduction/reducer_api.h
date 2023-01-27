/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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

#ifndef __SYCL_CTS_TEST_REDUCER_API_H
#define __SYCL_CTS_TEST_REDUCER_API_H

#include "../../util/usm_helper.h"
#include "../common/common.h"
#include "../common/type_coverage.h"
#include "identity_helper.h"
#include "reduction_common.h"

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

template <typename AccumulatorT, typename OperatorT>
struct kernel_name_value_ptr;

template <typename AccumulatorT, typename OperatorT>
struct kernel_name_buffer;

template <typename AccumulatorT, typename OperatorT>
struct kernel_name_span;

template <typename AccumulatorT>
struct check_reducer_subscript {
  template <typename OperatorT>
  void check_value_ptr(sycl::queue& queue) {
    auto allocated_memory =
        usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, AccumulatorT>(
            queue, 1);

    constexpr size_t result_count = 3;
    std::vector<int> results(result_count, 0);
    {
      sycl::buffer<int> buf_results(results.data(),
                                    sycl::range<1>{result_count});
      queue.submit([&](sycl::handler& cgh) {
        auto acc_results =
            buf_results.get_access<sycl::access_mode::write>(cgh);
        auto reduction = sycl::reduction(allocated_memory.get(), OperatorT{});
        cgh.parallel_for<kernel_name_value_ptr<AccumulatorT, OperatorT>>(
            sycl::range<1>{1}, reduction, [=](sycl::id<1>, auto& reducer) {
              size_t i = 0;

              {
                using red_t = decltype(reducer);
                acc_results[i++] =
                    std::is_same_v<AccumulatorT, typename red_t::value_type>;
                acc_results[i++] =
                    std::is_same_v<OperatorT, typename red_t::binary_operation>;
                acc_results[i++] = 0 == red_t::dimensions;
              }

              assert(i == result_count);
            });
      });
    }

    CHECK(std::all_of(results.begin(), results.end(),
                      [](int val) { return val; }));
  }

  template <typename OperatorT>
  void check_buffer(sycl::queue& queue) {
    AccumulatorT red_output;
    sycl::buffer<AccumulatorT> buf_output{&red_output, sycl::range<1>{1}};

    constexpr size_t result_count = 3;
    std::vector<int> results(result_count, 0);
    {
      sycl::buffer<int> buf_results(results.data(),
                                    sycl::range<1>{result_count});
      queue.submit([&](sycl::handler& cgh) {
        auto acc_results =
            buf_results.get_access<sycl::access_mode::write>(cgh);
        auto reduction = sycl::reduction(buf_output, cgh, OperatorT{});
        cgh.parallel_for<kernel_name_buffer<AccumulatorT, OperatorT>>(
            sycl::range<1>{1}, reduction, [=](sycl::id<1>, auto& reducer) {
              size_t i = 0;

              {
                using red_t = decltype(reducer);
                acc_results[i++] =
                    std::is_same_v<AccumulatorT, typename red_t::value_type>;
                acc_results[i++] =
                    std::is_same_v<OperatorT, typename red_t::binary_operation>;
                acc_results[i++] = 0 == red_t::dimensions;
              }

              assert(i == result_count);
            });
      });
    }

    CHECK(std::all_of(results.begin(), results.end(),
                      [](int val) { return val; }));
  }

  template <typename OperatorT>
  void check_span(sycl::queue& queue) {
    auto allocated_memory =
        usm_helper::allocate_usm_memory<sycl::usm::alloc::shared, AccumulatorT>(
            queue, 1);

    constexpr size_t result_count = 6;
    std::vector<int> results(result_count, 0);
    {
      sycl::buffer<int> buf_results(results.data(),
                                    sycl::range<1>{result_count});
      queue.submit([&](sycl::handler& cgh) {
        auto acc_results =
            buf_results.get_access<sycl::access_mode::write>(cgh);
        sycl::span<AccumulatorT, 1> span_output{allocated_memory.get(), 1};

        auto reduction = sycl::reduction(span_output, OperatorT{});
        cgh.parallel_for<kernel_name_span<AccumulatorT, OperatorT>>(
            sycl::range<1>{1}, reduction, [=](sycl::id<1>, auto& reducer) {
              size_t i = 0;

              {
                using red_t = decltype(reducer);
                acc_results[i++] =
                    std::is_same_v<AccumulatorT, typename red_t::value_type>;
                acc_results[i++] =
                    std::is_same_v<OperatorT, typename red_t::binary_operation>;
                acc_results[i++] = 1 == red_t::dimensions;
              }

              {
                using red_t = decltype(reducer[0]);
                acc_results[i++] =
                    std::is_same_v<AccumulatorT, typename red_t::value_type>;
                acc_results[i++] =
                    std::is_same_v<OperatorT, typename red_t::binary_operation>;
                acc_results[i++] = 0 == red_t::dimensions;
              }

              assert(i == result_count);
            });
      });
    }

    CHECK(std::all_of(results.begin(), results.end(),
                      [](int val) { return val; }));
  }

  void operator()(sycl::queue& queue, const std::string& type_name) {
    INFO("type: " << type_name);

    using OperatorT = sycl::plus<AccumulatorT>;

    bool has_aspect =
        queue.get_device().has(sycl::aspect::usm_shared_allocations);
    if (!has_aspect) {
      WARN(
          "Device does not support accessing to unified shared memory "
          "allocation. Skipping value ptr and span tests.");
    }

    if (has_aspect) check_value_ptr<OperatorT>(queue);
    check_buffer<OperatorT>(queue);
    if (has_aspect) check_span<OperatorT>(queue);
  }
};

template <typename AccumulatorT, typename OperatorT>
struct kernel_name_identity;

template <typename OperatorT, typename AccumulatorT, typename Enable = void>
struct check_reducer_identity_operator {
  void operator()(sycl::queue& queue, const std::string& op_name) {}
};

/**
 Only do check if the \p OperatorT and \p AccumulatorT type combination is
 well-defined, which is tested with <tt>is_legal_operator_v</tt>. */
template <typename OperatorT, typename AccumulatorT>
struct check_reducer_identity_operator<
    OperatorT, AccumulatorT,
    typename std::enable_if_t<is_legal_operator_v<AccumulatorT, OperatorT>>> {
  void operator()(sycl::queue& queue, const std::string& op_name) {
    INFO("operation type: " << op_name);

    // dummy output value
    AccumulatorT red_output;
    sycl::buffer<AccumulatorT> buf_output{&red_output, sycl::range<1>{1}};

    AccumulatorT result = 0;
    {
      sycl::buffer<AccumulatorT> buf_results(&result, sycl::range<1>{1});
      queue.submit([&](sycl::handler& cgh) {
        auto acc_results =
            buf_results.template get_access<sycl::access_mode::write>(cgh);
        auto reduction = sycl::reduction(buf_output, cgh, OperatorT{});
        cgh.parallel_for<kernel_name_identity<AccumulatorT, OperatorT>>(
            sycl::range<1>{1}, reduction, [=](sycl::id<1> idx, auto& reducer) {
              acc_results[0] = reducer.identity();
            });
      });
    }

    AccumulatorT expected = get_identity<AccumulatorT, OperatorT>();
    CHECK((result == expected));
  }
};

template <typename AccumulatorT>
struct check_reducer_identity {
  void operator()(sycl::queue& queue, const std::string& type_name) {
    INFO("variable type: " << type_name);
    const auto op_types = named_type_pack<
        sycl::plus<AccumulatorT>, sycl::multiplies<AccumulatorT>,
        sycl::bit_and<AccumulatorT>, sycl::bit_or<AccumulatorT>,
        sycl::bit_xor<AccumulatorT>, sycl::logical_and<AccumulatorT>,
        sycl::logical_or<AccumulatorT>, sycl::minimum<AccumulatorT>,
        sycl::maximum<AccumulatorT>>::generate("plus", "multiplies", "bit_and",
                                               "bit_or", "bit_xor",
                                               "logical_and", "logical_or",
                                               "minimum", "maximum");
    for_all_types<check_reducer_identity_operator, AccumulatorT>(op_types,
                                                                 queue);
  }
};

#endif  // __SYCL_CTS_TEST_REDUCER_API_H
