/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2024 The Khronos Group Inc.
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

namespace default_context::tests {

#ifdef SYCL_KHR_DEFAULT_CONTEXT

static void testDefaultContext() {
  sycl::platform platform{};
  sycl::context defaultContext = platform.khr_get_default_context();
  CHECK(defaultContext.get_devices() == platform.get_devices());
}

static void testQueueConstructors() {
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

#endif

TEST_CASE("Test case for \"Default Context\" extension",
          "[khr_default_context]") {
#ifndef SYCL_KHR_DEFAULT_CONTEXT
  SKIP("SYCL_KHR_DEFAULT_CONTEXT is not defined");
#else
  testDefaultContext();
  testQueueConstructors();
#endif
}

}  // namespace default_context::tests
