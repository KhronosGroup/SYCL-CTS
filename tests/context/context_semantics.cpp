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
#include "../common/semantics_reference.h"

struct storage {
  std::size_t device_count;

  explicit storage(const sycl::context& context)
      : device_count(context.get_devices().size()) {}

  bool check(const sycl::context& context) const {
    return context.get_devices().size() == device_count;
  }
};

TEST_CASE("context common reference semantics", "[context]") {
  sycl::context context_0{};
  sycl::context context_1{};

  common_reference_semantics::check_host<storage>(context_0, context_1,
                                                  "context");
}
