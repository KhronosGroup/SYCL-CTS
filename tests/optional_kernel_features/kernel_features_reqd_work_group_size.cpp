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

//  Provides tests for the exception that are thrown when device
//  doesn’t support required work group size.

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"
#include "catch2/catch_template_test_macros.hpp"
#include "kernel_features_common.h"

namespace kernel_features_reqd_work_group_size {
using namespace sycl_cts;
using namespace kernel_features_common;

template <size_t N, int Dimensions>
class Functor {
 public:
  [[sycl::reqd_work_group_size(N)]] void operator()(sycl::nd_item<1>) const {}
  [[sycl::reqd_work_group_size(N)]] void operator()(sycl::group<1>) const {}

  [[sycl::reqd_work_group_size(N, N)]] void operator()(sycl::nd_item<2>) const {
  }
  [[sycl::reqd_work_group_size(N, N)]] void operator()(sycl::group<2>) const {}

  [[sycl::reqd_work_group_size(N, N, N)]] void operator()(
      sycl::nd_item<3>) const {}
  [[sycl::reqd_work_group_size(N, N, N)]] void operator()(
      sycl::group<3>) const {}
};

template <size_t N, int Dimensions>
class kernel_reqd_wg_size;

// FIXME: re-enable when max_work_item_sizes is implemented in hipsycl and
// computcpp
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL
template <size_t N, int Dimensions>
void test_size() {
  INFO("N = " + std::to_string(N));
  using kname = kernel_reqd_wg_size<N, Dimensions>;
  auto queue = util::get_cts_object::queue();
  auto max_wg_size =
      queue.get_device().get_info<sycl::info::device::max_work_group_size>();
  auto max_work_item_sizes =
      queue.get_device()
          .get_info<sycl::info::device::max_work_item_sizes<Dimensions>>();

  bool is_exception_expected = (std::pow(N, Dimensions) > max_wg_size);

  for (int i = 0; i < Dimensions; i++)
    if (max_work_item_sizes[i] < N) is_exception_expected |= true;

  // Set expected error code
  constexpr sycl::errc expected_errc = sycl::errc::kernel_not_supported;

  {
    if constexpr (Dimensions == 1) {
      const auto lambda_nd_item_arg_1D =
          [](sycl::nd_item<1>) [[sycl::reqd_work_group_size(N)]] {};
      const auto lambda_group_arg_1D = [](sycl::group<1>)
                                           [[sycl::reqd_work_group_size(N)]] {};
      run_separate_lambda_nd_range<kname, N, Dimensions>(
          is_exception_expected, expected_errc, queue, lambda_nd_item_arg_1D,
          lambda_group_arg_1D);
    } else if constexpr (Dimensions == 2) {
      const auto lambda_nd_item_arg_2D =
          [](sycl::nd_item<2>) [[sycl::reqd_work_group_size(N, N)]] {};
      const auto lambda_group_arg_2D =
          [](sycl::group<2>) [[sycl::reqd_work_group_size(N, N)]] {};
      run_separate_lambda_nd_range<kname, N, Dimensions>(
          is_exception_expected, expected_errc, queue, lambda_nd_item_arg_2D,
          lambda_group_arg_2D);
    } else {
      const auto lambda_nd_item_arg_3D =
          [](sycl::nd_item<3>) [[sycl::reqd_work_group_size(N, N, N)]] {};
      const auto lambda_group_arg_3D =
          [](sycl::group<3>) [[sycl::reqd_work_group_size(N, N, N)]] {};
      run_separate_lambda_nd_range<kname, N, Dimensions>(
          is_exception_expected, expected_errc, queue, lambda_nd_item_arg_3D,
          lambda_group_arg_3D);
    }
  }
  {
    run_functor_nd_range<Functor<N, Dimensions>, N, Dimensions>(
        is_exception_expected, expected_errc, queue);
  }
  {
    RUN_SUBMISSION_CALL_ND_RANGE(
        N, Dimensions, is_exception_expected, expected_errc, queue,
        [[sycl::reqd_work_group_size(N)]], kname, NO_KERNEL_BODY);
  }
}
#endif  // !SYCL_CTS_COMPILING_WITH_HIPSYCL

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(hipSYCL)
("Exceptions thrown by [[reqd_work_group_size(N)]] with unsupported size",
 "[kernel_features]", ((int Dimensions), Dimensions), 1, 2, 3)({
  test_size<4, Dimensions>();
  test_size<4294967295, Dimensions>();
});

}  // namespace kernel_features_reqd_work_group_size
