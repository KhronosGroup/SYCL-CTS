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
#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
#include "reducer_api.h"
#endif

#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

struct kernel_name;

// FIXME: re-enable when sycl::reduction is implemented in hipSYCL
DISABLED_FOR_TEST_CASE(hipSYCL)
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

// FIXME: re-enable when reducer is fully implemented in hipSYCL
DISABLED_FOR_TEST_CASE(hipSYCL)
("reducer identity for bool", "[reducer]")({
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  check_reducer_identity<bool>{}(queue, "bool");
});
