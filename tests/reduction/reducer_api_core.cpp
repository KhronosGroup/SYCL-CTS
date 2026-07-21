/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/
#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
#include "reducer_api.h"
#endif

#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

struct kernel_name;

// FIXME: re-enable when sycl::reduction is implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reducer class", "[reducer]")({
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();

  // dummy output value
  int red_output;
  sycl::buffer<int> buf_output{&red_output, sycl::range<1>{1}};

  constexpr size_t result_count = 4;
  std::vector<int> results(result_count, 0);
  {
    sycl::buffer<int> buf_results(results.data(), sycl::range<1>{result_count});
    queue.submit([&](sycl::handler& cgh) {
      auto acc_results =
          buf_results.template get_access<sycl::access_mode::write>(cgh);
      auto reduction = sycl::reduction(buf_output, cgh, sycl::plus<>{});
      cgh.parallel_for<kernel_name>(
          sycl::range<1>{1}, reduction, [=](sycl::id<1> idx, auto& reducer) {
            using reducer_t = std::remove_reference_t<decltype(reducer)>;
            size_t i = 0;
            acc_results[i++] = !std::is_copy_constructible_v<reducer_t>;
            acc_results[i++] = !std::is_move_constructible_v<reducer_t>;
            acc_results[i++] = !std::is_copy_assignable_v<reducer_t>;
            acc_results[i++] = !std::is_move_assignable_v<reducer_t>;
            assert(result_count == i);
          });
    });
  }

  // all results are expected to evaluate to true
  CHECK(
      std::all_of(results.begin(), results.end(), [](int val) { return val; }));
});

// FIXME: re-enable when reducer is fully implemented in AdaptiveCpp
DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("reducer api core", "[reducer]")({
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();

  // check subscript operator
  for_all_types<check_reducer_subscript>(reduction_common::scalar_types, queue);

  // check identity operator
  for_all_types<check_reducer_identity>(reduction_common::scalar_types, queue);
  check_reducer_identity<bool>{}(queue, "bool");
});
