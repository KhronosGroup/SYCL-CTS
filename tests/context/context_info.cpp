/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
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
  {  // check get_info for info::context::atomic_memory_scope_capabilities
    check_get_info_param<sycl::info::context::atomic_memory_scope_capabilities,
                         std::vector<sycl::memory_scope>>(ctx);
    std::vector<sycl::memory_scope> capabilities =
        ctx.get_info<sycl::info::context::atomic_memory_scope_capabilities>();
    CHECK(check_contains(capabilities, sycl::memory_scope::work_group));
  }
  {  // check get_info for info::context::atomic_fence_scope_capabilities
    check_get_info_param<sycl::info::context::atomic_fence_scope_capabilities,
                         std::vector<sycl::memory_scope>>(ctx);
    std::vector<sycl::memory_scope> capabilities =
        ctx.get_info<sycl::info::context::atomic_fence_scope_capabilities>();
    CHECK(check_contains(capabilities, sycl::memory_scope::work_group));
  }
}
