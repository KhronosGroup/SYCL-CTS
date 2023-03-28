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

#include "weak_object_common.h"

namespace weak_object_initialization {
using namespace weak_object_common;

TEST_CASE("weak_object init", "[weak_object]") {
#if !defined SYCL_EXT_ONEAPI_WEAK_OBJECT
  SKIP("SYCL_EXT_ONEAPI_WEAK_OBJECT is not defined");
#else
  test_weak_object_init<sycl::accessor<int> >{}("accessor");
  test_weak_object_init<sycl::host_accessor<int> >{}("host_accessor");
  test_weak_object_init<sycl::buffer<int> >{}("buffer");
  test_weak_object_init<sycl::queue>{}("queue");
  test_weak_object_init<sycl::context>{}("context");
  test_weak_object_init<sycl::event>{}("event");

  test_weak_object_init<sycl::queue>{}.test_local_types();
#endif
}
}  // namespace weak_object_initialization
