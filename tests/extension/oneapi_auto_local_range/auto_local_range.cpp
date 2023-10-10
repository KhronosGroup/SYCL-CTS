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

#include "../../common/common.h"
#include <type_traits>

namespace auto_local_range::tests {

#ifdef SYCL_EXT_ONEAPI_AUTO_LOCAL_RANGE

template <int Dimensions>
static void check_auto_range_result_type() {
  static_assert(1 <= Dimensions && Dimensions <= 3);
  using F = decltype(sycl::ext::oneapi::experimental::auto_range<Dimensions>);
  static_assert(std::is_same_v<typename std::invoke_result<F>::type,
                               sycl::range<Dimensions>>);
}

template <size_t... Dims>
static void check_auto_range() {
  constexpr int Dimensions = sizeof...(Dims);
  static_assert(1 <= Dimensions && Dimensions <= 3);

  sycl::queue q;
  sycl::range<Dimensions> N{Dims...};
  sycl::buffer<int, Dimensions> input_buffer{N};
  sycl::buffer<int, Dimensions> output_buffer{N};

  {
    auto input = input_buffer.get_host_access();
    int i = 0;
    for (auto& it : input) {
      it = i + 1;
      i++;
    }
  }

  q.submit([&](sycl::handler& cgh) {
     sycl::accessor input{input_buffer, cgh, sycl::read_only};
     sycl::accessor output{output_buffer, cgh, sycl::write_only};
     sycl::range<Dimensions> auto_range =
         sycl::ext::oneapi::experimental::auto_range<Dimensions>();
     cgh.parallel_for(sycl::nd_range<Dimensions>{N, auto_range}, [=](auto it) {
       sycl::group<Dimensions> g = it.get_group();
       int local_accumulator = 0;
       for (int i = it.get_local_linear_id(); i < N.size();
            i += g.get_local_linear_range()) {
         int value = input[unlinearize(N, i)];
         local_accumulator += value;
       }
       int total =
           sycl::reduce_over_group(g, local_accumulator, sycl::plus<>());
       output[it.get_global_id()] = total;
     });
   }).wait();

  {
    const int expected_sum = (N.size() * (N.size() + 1)) / 2;
    auto output = output_buffer.get_host_access();
    for (const auto& it : output) {
      CHECK(it == expected_sum);
    }
  }
}

#endif

TEST_CASE("Test case for \"Auto Local Range\" extension",
          "[oneapi_auto_local_range]") {
#ifndef SYCL_EXT_ONEAPI_AUTO_LOCAL_RANGE
  SKIP("SYCL_EXT_ONEAPI_AUTO_LOCAL_RANGE is not defined");
#else
  check_auto_range_result_type<1>();
  check_auto_range_result_type<2>();
  check_auto_range_result_type<3>();

  check_auto_range<1>();
  check_auto_range<4>();
  check_auto_range<12>();
  check_auto_range<200>();

  check_auto_range<1, 1>();
  check_auto_range<4, 4>();
  check_auto_range<2, 10>();
  check_auto_range<10, 2>();

  check_auto_range<2, 2, 2>();
  check_auto_range<3, 2, 1>();
  check_auto_range<1, 2, 3>();
  check_auto_range<3, 1, 2>();
#endif
}

}  // namespace auto_local_range::tests
