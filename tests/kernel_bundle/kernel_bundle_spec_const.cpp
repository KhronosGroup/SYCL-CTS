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

#include "kernel_bundle.h"

namespace kernel_bundle_spec_const {

constexpr sycl::specialization_id<int> SpecName(5);
constexpr sycl::specialization_id<int> OtherSpecName(10);

TEST_CASE(
    "Check specialization constants functionality with empty kernel_bundle",
    "[kernel_bundle]") {
  sycl::context ctx;
  const auto always_false_selector = [](auto device_image) { return false; };
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctx, always_false_selector);
  REQUIRE(bundle.empty());

  CHECK(!bundle.contains_specialization_constants());
  CHECK(!bundle.native_specialization_constant());
  CHECK(!bundle.has_specialization_constant<SpecName>());
}

TEST_CASE(
    "Check specialization constants functionality with Kernel bundle with "
    "`kernel_handler::get_specialization_constant()` call",
    "[kernel_bundle]") {
  sycl::device device = sycl_cts::util::get_cts_object::device();
  sycl::queue queue = sycl_cts::util::get_cts_object::queue();
  sycl::context ctx = queue.get_context();

  using KernelName = class simple_kernel;

  queue.submit([&](sycl::handler& cgh) {
    cgh.single_task<KernelName>([=](sycl::kernel_handler h) {
      // just to establish `kernel_handler::get_specialization_constant()` call
      // usage of spec constants is checked in spec_constants tests
      h.get_specialization_constant<SpecName>();
    });
  });

  auto inputBundle =
      sycl::get_kernel_bundle<KernelName, sycl::bundle_state::input>(ctx,
                                                                     {device});

  constexpr int new_value = 10;

  if (!inputBundle.has_kernel(sycl::get_kernel_id<KernelName>())) {
    SKIP("kernel_bundle doesn't have required kernel. Test is skipped.");
  }

  inputBundle.set_specialization_constant<SpecName>(new_value);
  inputBundle.set_specialization_constant<OtherSpecName>(new_value);

  CHECK_NOTHROW(inputBundle.native_specialization_constant());
  CHECK(inputBundle.has_specialization_constant<SpecName>());
  CHECK(!inputBundle.has_specialization_constant<OtherSpecName>());

  {
    INFO("Check get_specialization_constant() return type");
    CHECK(std::is_same_v<
          std::remove_reference_t<decltype(SpecName)>::value_type,
          decltype(inputBundle.get_specialization_constant<SpecName>())>);
    CHECK(std::is_same_v<
          std::remove_reference_t<decltype(OtherSpecName)>::value_type,
          decltype(inputBundle.get_specialization_constant<OtherSpecName>())>);
  }

  CHECK(inputBundle.get_specialization_constant<SpecName>() == new_value);

  auto objectBundle = sycl::compile(inputBundle);

  CHECK_NOTHROW(objectBundle.native_specialization_constant());
  CHECK(objectBundle.has_specialization_constant<SpecName>());
  CHECK(!objectBundle.has_specialization_constant<OtherSpecName>());

  CHECK(objectBundle.get_specialization_constant<SpecName>() == new_value);

  auto execBundleViaBuild = sycl::build(inputBundle);

  CHECK_NOTHROW(execBundleViaBuild.native_specialization_constant());
  CHECK(execBundleViaBuild.has_specialization_constant<SpecName>());
  CHECK(!execBundleViaBuild.has_specialization_constant<OtherSpecName>());

  CHECK(execBundleViaBuild.get_specialization_constant<SpecName>() ==
        new_value);

  auto execBundleViaLink = sycl::link(objectBundle);

  CHECK_NOTHROW(execBundleViaLink.native_specialization_constant());
  CHECK(execBundleViaLink.has_specialization_constant<SpecName>());
  CHECK(!execBundleViaLink.has_specialization_constant<OtherSpecName>());

  CHECK(execBundleViaLink.get_specialization_constant<SpecName>() == new_value);
}

}  // namespace kernel_bundle_spec_const
