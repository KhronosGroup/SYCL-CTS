/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Tests for the context constructors added by the sycl_khr_properties
//  extension.
//
//  The extension does not define any property that can be used with the
//  context class, and applications cannot define their own properties, so the
//  only PropertyOrList argument that satisfies the constraints is the empty
//  properties list. Properties of other classes are used to check that the
//  constraints remove the constructors from overload resolution.
//
*******************************************************************************/

#include "khr_properties_common.h"

#include <algorithm>
#include <vector>

namespace khr_properties::tests {

template <typename PropertyOrList>
void check_context_constraints(bool expected) {
  using sycl::context;
  using dev_t = const sycl::device&;
  using plt_t = const sycl::platform&;
  using devs_t = const std::vector<sycl::device>&;
  using ah_t = sycl::async_handler;
  using p_t = PropertyOrList;

  CHECK(std::is_constructible_v<context, p_t> == expected);
  CHECK(std::is_constructible_v<context, ah_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, dev_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, dev_t, ah_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, plt_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, plt_t, ah_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, devs_t, p_t> == expected);
  CHECK(std::is_constructible_v<context, devs_t, ah_t, p_t> == expected);
}

bool contains_device(const std::vector<sycl::device>& devices,
                     const sycl::device& dev) {
  return std::find(devices.begin(), devices.end(), dev) != devices.end();
}

TEST_CASE("the new context constructors are constrained on PropertyOrList",
          "[khr_properties][context]") {
  SECTION("the empty properties list is accepted") {
    check_context_constraints<khr::empty_properties_t>(true);
  }
  SECTION("queue properties are rejected") {
    check_context_constraints<prop::in_order>(false);
    check_context_constraints<prop::enable_profiling>(false);
  }
  SECTION("reduction properties are rejected") {
    check_context_constraints<prop::initialize_to_identity>(false);
  }
  SECTION("properties lists with properties of other classes are rejected") {
    check_context_constraints<queue_props_t>(false);
    check_context_constraints<reduction_props_t>(false);
  }
  SECTION("types that are neither properties nor lists are rejected") {
    check_context_constraints<not_a_property>(false);
    check_context_constraints<key::in_order>(false);
  }
  SECTION("the new constructors are explicit") {
    STATIC_CHECK_FALSE(
        std::is_convertible_v<khr::empty_properties_t, sycl::context>);
  }
}

TEST_CASE("contexts constructed with an empty properties list",
          "[khr_properties][context]") {
  const khr::empty_properties_t props{};
  cts_async_handler async_handler;
  const sycl::device dev = sycl_cts::util::get_cts_object::device();
  const sycl::platform plt = dev.get_platform();
  const std::vector<sycl::device> dev_list{dev};

  SECTION("context(props)") {
    sycl::context ctx{props};
    CHECK_FALSE(ctx.get_devices().empty());
  }
  SECTION("context(asyncHandler, props)") {
    sycl::context ctx{async_handler, props};
    CHECK_FALSE(ctx.get_devices().empty());
  }
  SECTION("context(dev, props)") {
    sycl::context ctx{dev, props};
    CHECK(ctx.get_devices() == dev_list);
  }
  SECTION("context(dev, asyncHandler, props)") {
    sycl::context ctx{dev, async_handler, props};
    CHECK(ctx.get_devices() == dev_list);
  }
  SECTION("context(plt, props)") {
    sycl::context ctx{plt, props};
    CHECK(ctx.get_platform() == plt);
    for (const auto& d : plt.get_devices()) {
      CHECK(contains_device(ctx.get_devices(), d));
    }
  }
  SECTION("context(plt, asyncHandler, props)") {
    sycl::context ctx{plt, async_handler, props};
    CHECK(ctx.get_platform() == plt);
    for (const auto& d : plt.get_devices()) {
      CHECK(contains_device(ctx.get_devices(), d));
    }
  }
  SECTION("context(deviceList, props)") {
    sycl::context ctx{dev_list, props};
    CHECK(ctx.get_devices() == dev_list);
  }
  SECTION("context(deviceList, asyncHandler, props)") {
    sycl::context ctx{dev_list, async_handler, props};
    CHECK(ctx.get_devices() == dev_list);
  }
}

TEST_CASE("a context constructed with an empty properties list is usable",
          "[khr_properties][context]") {
  const sycl::device dev = sycl_cts::util::get_cts_object::device();
  sycl::context ctx{dev, khr::empty_properties_t{}};
  sycl::queue q{ctx, dev};
  CHECK(q.get_context() == ctx);
  sycl::buffer<int, 1> buf{sycl::range<1>{1}};
  q.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{buf, cgh, sycl::write_only};
    cgh.single_task([=] { acc[0] = 42; });
  });
  sycl::host_accessor acc{buf, sycl::read_only};
  CHECK(acc[0] == 42);
}

}  // namespace khr_properties::tests
