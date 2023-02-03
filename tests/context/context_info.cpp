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

TEST_CASE("context info", "[context]") {
  auto ctx = sycl_cts::util::get_cts_object::context();

  {  // check get_info for info::context::platform
    check_get_info_param<sycl::info::context::platform, sycl::platform>(ctx);
  }
  {  // check get_info for info::context::devices
    check_get_info_param<sycl::info::context::devices,
                         std::vector<sycl::device>>(ctx);
  }
  {  // check get_info for info::context::atomic_memory_order_capabilities
    check_get_info_param<sycl::info::context::atomic_memory_order_capabilities,
                         std::vector<sycl::memory_order>>(ctx);
    std::vector<sycl::memory_order> capabilities =
        ctx.get_info<sycl::info::context::atomic_memory_order_capabilities>();
    CHECK(check_contains(capabilities, sycl::memory_order::relaxed));
  }
#ifndef SYCL_CTS_COMPILING_WITH_DPCPP
  {  // check get_info for info::context::atomic_fence_order_capabilities
    check_get_info_param<sycl::info::context::atomic_fence_order_capabilities,
                         std::vector<sycl::memory_order>>(ctx);
    std::vector<sycl::memory_order> capabilities =
        ctx.get_info<sycl::info::context::atomic_fence_order_capabilities>();
    CHECK(check_contains(capabilities, sycl::memory_order::relaxed));
    CHECK(check_contains(capabilities, sycl::memory_order::acquire));
    CHECK(check_contains(capabilities, sycl::memory_order::release));
    CHECK(check_contains(capabilities, sycl::memory_order::acq_rel));
  }
#else
  WARN(
      "Implementation does not support "
      "sycl::info::context::atomic_fence_order_capabilities "
      "Skipping the test case.");
#endif
  {  // check get_info for info::context::atomic_memory_scope_capabilities
    check_get_info_param<sycl::info::context::atomic_memory_scope_capabilities,
                         std::vector<sycl::memory_scope>>(ctx);
    std::vector<sycl::memory_scope> capabilities =
        ctx.get_info<sycl::info::context::atomic_memory_scope_capabilities>();
    CHECK(check_contains(capabilities, sycl::memory_scope::work_group));
  }
#ifndef SYCL_CTS_COMPILING_WITH_DPCPP
  {  // check get_info for info::context::atomic_fence_scope_capabilities
    check_get_info_param<sycl::info::context::atomic_fence_scope_capabilities,
                         std::vector<sycl::memory_scope>>(ctx);
    std::vector<sycl::memory_scope> capabilities =
        ctx.get_info<sycl::info::context::atomic_fence_scope_capabilities>();
    CHECK(check_contains(capabilities, sycl::memory_scope::work_group));
  }
#else
  WARN(
      "Implementation does not support "
      "sycl::info::context::atomic_fence_scope_capabilities "
      "Skipping the test case.");
#endif
}
