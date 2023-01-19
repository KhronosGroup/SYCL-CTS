/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
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

#include "../common/common.h"

#define TEST_NAME queue_properties

namespace queue_properties {

using namespace sycl_cts;

TEST_CASE("check property::queue::enable_profiling", "[queue]") {
  sycl::queue queue(
      util::get_cts_object::device(),
      sycl::property_list{sycl::property::queue::enable_profiling()});
  CHECK(queue.has_property<sycl::property::queue::enable_profiling>());

  auto prop = queue.get_property<sycl::property::queue::enable_profiling>();
  check_return_type<sycl::property::queue::enable_profiling>(
      prop,
      "sycl::queue::has_property<sycl::property::queue::"
      "enable_profiling>()");
}
}  // namespace queue_properties
