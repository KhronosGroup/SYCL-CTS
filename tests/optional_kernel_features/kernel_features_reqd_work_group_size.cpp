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

template <size_t N>
class Functor {
 public:
  [[sycl::reqd_work_group_size(N)]] void operator()() const {}
  [[sycl::reqd_work_group_size(N)]] void operator()(sycl::item<1>) const {}
  [[sycl::reqd_work_group_size(N)]] void operator()(sycl::group<1>) const {}
};

template <size_t N>
class kernel_reqd_wg_size;

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(ComputeCpp, hipSYCL)
("Exceptions thrown by [[reqd_work_group_size(N)]] with unsupported size",
 "[kernel_features]", ((size_t N), N), 16, 4294967295)({
  using kname = kernel_reqd_wg_size<N>;
  auto queue = util::get_cts_object::queue();
  auto max_wg_size =
      queue.get_device().get_info<sycl::info::device::max_work_group_size>();

  bool is_exception_expected = (N > max_wg_size);

  // Set expected error code
  constexpr sycl::errc expected_errc = sycl::errc::kernel_not_supported;

  {
    const auto lambda_no_arg = []() [[sycl::reqd_work_group_size(N)]] {};
    const auto lambda_item_arg = [](sycl::item<1>)
                                     [[sycl::reqd_work_group_size(N)]] {};
    const auto lambda_group_arg = [](sycl::group<1>)
                                      [[sycl::reqd_work_group_size(N)]] {};

    run_separate_lambda<kname>(is_exception_expected, expected_errc, queue,
                               lambda_no_arg, lambda_item_arg,
                               lambda_group_arg);
  }
  { run_functor<Functor<N>>(is_exception_expected, expected_errc, queue); }
  {
    RUN_SUBMISSION_CALL(is_exception_expected, expected_errc, queue,
                        [[sycl::reqd_work_group_size(N)]], kname,
                        NO_KERNEL_BODY);
  }
});

}  // namespace kernel_features_reqd_work_group_size
