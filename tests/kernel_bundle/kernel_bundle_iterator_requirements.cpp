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
//  Provides tests for kernel_bundle device_image iterators
//
*******************************************************************************/

#include "../../util/named_requirement_verification/legacy_forward_iterator.h"
#include "kernel_bundle.h"

TEST_CASE(
    "Check kernel_bundle::begin(), kernel_bundle::end() "
    "return type and LegacyForwardIterator requirement",
    "[kernel_bundle]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  using kernel_name = class simple_kernel_get_kernel_iterator;
  sycl_cts::tests::kernel_bundle::define_kernel<kernel_name>(queue);
  auto kernel_bundle =
      sycl::get_kernel_bundle<kernel_name, sycl::bundle_state::executable>(
          queue.get_context());

  // Type
  using expected_t = sycl::kernel_bundle<
      sycl::bundle_state::executable>::device_image_iterator;
  auto begin_it = kernel_bundle.begin();
  auto end_it = kernel_bundle.end();

  CHECK(std::is_same_v<expected_t, decltype(begin_it)>);
  CHECK(std::is_same_v<expected_t, decltype(end_it)>);

  // Requirement
  using namespace named_requirement_verification;

  error_messages messages;
  constexpr size_t size_of_res_array =
      legacy_forward_iterator_requirement::count_of_possible_errors;
  auto verification_result =
      legacy_forward_iterator_requirement{}.is_satisfied_for(begin_it);

  if (!verification_result.first) {
    for (int i = 0; i < size_of_res_array; ++i) {
      if (verification_result.second[i] != 0) {
        FAIL_CHECK(messages.error_message(verification_result.second[i]));
      }
    }
  }
}
