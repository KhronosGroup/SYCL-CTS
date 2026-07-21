/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2024 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../../common/common.h"

namespace default_context::tests {

TEST_CASE(
    "the default context extension defines the SYCL_KHR_DEFAULT_CONTEXT macro",
    "[khr_default_context]") {
#ifndef SYCL_KHR_DEFAULT_CONTEXT
  static_assert(false, "SYCL_KHR_DEFAULT_CONTEXT is not defined");
#endif
}

TEST_CASE("the default context contains all of the default platform's devices",
          "[khr_default_context]") {
  sycl::platform platform{};
  sycl::context defaultContext = platform.khr_get_default_context();
  CHECK(defaultContext.get_devices() == platform.get_devices());
}

TEST_CASE("queue constructors use the default context or the context parameter",
          "[khr_default_context]") {
  const sycl::property_list& propList = {};
  cts_async_handler asyncHandler;
  const auto& deviceSelector = sycl::default_selector_v;
  sycl::device syclDevice;
  sycl::context syclContext;
  sycl::context defaultContext = sycl::platform{}.khr_get_default_context();

  // Check that a default-constructed context is not the default context.
  CHECK(syclContext != defaultContext);

  // Default context constructors
  CHECK(defaultContext == sycl::queue{propList}.get_context());
  CHECK(defaultContext == sycl::queue{asyncHandler, propList}.get_context());
  CHECK(defaultContext == sycl::queue{deviceSelector, propList}.get_context());
  CHECK(defaultContext ==
        sycl::queue{deviceSelector, asyncHandler, propList}.get_context());
  CHECK(defaultContext == sycl::queue{syclDevice, propList}.get_context());
  CHECK(defaultContext ==
        sycl::queue{syclDevice, asyncHandler, propList}.get_context());

  // Non-default context constructors
  CHECK(syclContext ==
        sycl::queue{syclContext, deviceSelector, propList}.get_context());
  CHECK(syclContext ==
        sycl::queue{syclContext, deviceSelector, asyncHandler, propList}
            .get_context());
  CHECK(syclContext ==
        sycl::queue{syclContext, syclDevice, propList}.get_context());
  CHECK(syclContext ==
        sycl::queue{syclContext, syclDevice, asyncHandler, propList}
            .get_context());
}

}  // namespace default_context::tests
