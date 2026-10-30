/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Tests for the queue additions of the sycl_khr_properties extension: the
//  new queue constructors, queue::is_in_order, queue::khr_is_profiling_enabled
//  and the khr::property::in_order and khr::property::enable_profiling
//  properties.
//
*******************************************************************************/

#include "khr_properties_common.h"

#include <optional>
#include <vector>

namespace khr_properties::tests {

class khr_props_queue_kernel_set;
class khr_props_queue_kernel_update;
class khr_props_queue_kernel_profiling;

using in_order_only_t = decltype(khr::properties{prop::in_order{}});
using profiling_only_t = decltype(khr::properties{prop::enable_profiling{}});

/**
 * Calls `check(q, ctx, dev)` for a queue constructed with each of the ten new
 * queue constructors, passing `props` as the PropertyOrList argument. `ctx`
 * and `dev` are the context and device that the queue is expected to use, or
 * an empty optional if the constructor does not specify them.
 */
template <typename PropertyOrList, typename Check>
void for_each_queue_ctor(PropertyOrList props, Check check) {
  cts_async_handler async_handler;
  const sycl::device dev = sycl_cts::util::get_cts_object::device();
  const sycl::context ctx{dev};

  SECTION("queue(props)") {
    sycl::queue q{props};
    check(q, std::optional<sycl::context>{}, std::optional<sycl::device>{});
  }
  SECTION("queue(asyncHandler, props)") {
    sycl::queue q{async_handler, props};
    check(q, std::optional<sycl::context>{}, std::optional<sycl::device>{});
  }
  SECTION("queue(deviceSelector, props)") {
    sycl::queue q{cts_selector, props};
    check(q, std::optional<sycl::context>{}, std::optional{dev});
  }
  SECTION("queue(deviceSelector, asyncHandler, props)") {
    sycl::queue q{cts_selector, async_handler, props};
    check(q, std::optional<sycl::context>{}, std::optional{dev});
  }
  SECTION("queue(syclDevice, props)") {
    sycl::queue q{dev, props};
    check(q, std::optional<sycl::context>{}, std::optional{dev});
  }
  SECTION("queue(syclDevice, asyncHandler, props)") {
    sycl::queue q{dev, async_handler, props};
    check(q, std::optional<sycl::context>{}, std::optional{dev});
  }
  SECTION("queue(syclContext, deviceSelector, props)") {
    sycl::queue q{ctx, cts_selector, props};
    check(q, std::optional{ctx}, std::optional{dev});
  }
  SECTION("queue(syclContext, deviceSelector, asyncHandler, props)") {
    sycl::queue q{ctx, cts_selector, async_handler, props};
    check(q, std::optional{ctx}, std::optional{dev});
  }
  SECTION("queue(syclContext, syclDevice, props)") {
    sycl::queue q{ctx, dev, props};
    check(q, std::optional{ctx}, std::optional{dev});
  }
  SECTION("queue(syclContext, syclDevice, asyncHandler, props)") {
    sycl::queue q{ctx, dev, async_handler, props};
    check(q, std::optional{ctx}, std::optional{dev});
  }
}

void check_ctx_dev(const sycl::queue& q,
                   const std::optional<sycl::context>& ctx,
                   const std::optional<sycl::device>& dev) {
  if (ctx) CHECK(q.get_context() == *ctx);
  if (dev) CHECK(q.get_device() == *dev);
}

/**
 * Submits two kernels without explicit dependencies, where the second one
 * reads the result of the first one. With an in-order queue the result is
 * deterministic.
 */
void check_in_order_execution(sycl::queue& q) {
  if (!q.get_device().has(sycl::aspect::usm_device_allocations)) return;
  int* value = sycl::malloc_device<int>(1, q);
  REQUIRE(value != nullptr);
  q.single_task<khr_props_queue_kernel_set>([=] { *value = 1; });
  q.single_task<khr_props_queue_kernel_update>(
      [=] { *value = *value * 2 + 1; });
  int result = 0;
  q.memcpy(&result, value, sizeof(int));
  q.wait_and_throw();
  CHECK(result == 3);
  sycl::free(value, q);
}

/** Returns true if the profiling info of a command submitted to q is
 * available, false if querying it throws errc::invalid. */
bool profiling_info_is_available(sycl::queue& q) {
  auto event = q.single_task<khr_props_queue_kernel_profiling>([] {});
  event.wait_and_throw();
  try {
    auto start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    CHECK(start <= end);
    return true;
  } catch (const sycl::exception& e) {
    CHECK(e.code() == sycl::errc::invalid);
    return false;
  }
}

bool device_has_queue_profiling() {
  return sycl_cts::util::get_cts_object::device().has(
      sycl::aspect::queue_profiling);
}

TEST_CASE("the new queue constructors are constrained on PropertyOrList",
          "[khr_properties][queue]") {
  using sycl::queue;
  using dev_t = const sycl::device&;
  using ctx_t = const sycl::context&;
  using ah_t = const sycl::async_handler&;
  using sel_t = decltype(sycl::default_selector_v);

  SECTION("accepted properties and properties lists") {
    STATIC_CHECK(std::is_constructible_v<queue, prop::in_order>);
    STATIC_CHECK(std::is_constructible_v<queue, prop::enable_profiling>);
    STATIC_CHECK(std::is_constructible_v<queue, khr::empty_properties_t>);
    STATIC_CHECK(std::is_constructible_v<queue, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, sel_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, sel_t, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, dev_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, dev_t, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, ctx_t, sel_t, queue_props_t>);
    STATIC_CHECK(
        std::is_constructible_v<queue, ctx_t, sel_t, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, ctx_t, dev_t, queue_props_t>);
    STATIC_CHECK(
        std::is_constructible_v<queue, ctx_t, dev_t, ah_t, queue_props_t>);
    STATIC_CHECK(std::is_constructible_v<queue, ctx_t, dev_t, ah_t,
                                         prop::enable_profiling>);
  }

  SECTION("properties that do not apply to queue are rejected") {
    using p_t = prop::initialize_to_identity;
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ah_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, sel_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, sel_t, ah_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, dev_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, dev_t, ah_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ctx_t, sel_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ctx_t, sel_t, ah_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ctx_t, dev_t, p_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, ctx_t, dev_t, ah_t, p_t>);
  }

  SECTION("properties lists that do not apply to queue are rejected") {
    using mixed_t = decltype(khr::properties{prop::in_order{},
                                             prop::initialize_to_identity{}});
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, reduction_props_t>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, mixed_t>);
    STATIC_CHECK_FALSE(
        std::is_constructible_v<queue, dev_t, reduction_props_t>);
    STATIC_CHECK_FALSE(
        std::is_constructible_v<queue, ctx_t, dev_t, ah_t, mixed_t>);
  }

  SECTION("types that are neither properties nor lists are rejected") {
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, not_a_property>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, key::in_order>);
    STATIC_CHECK_FALSE(std::is_constructible_v<queue, dev_t, not_a_property>);
  }

  SECTION("the DeviceSelector must be a device selector") {
    STATIC_CHECK_FALSE(
        std::is_constructible_v<queue, not_a_property, queue_props_t>);
    STATIC_CHECK_FALSE(
        std::is_constructible_v<queue, ctx_t, not_a_property, queue_props_t>);
  }

  SECTION("the new constructors are explicit") {
    STATIC_CHECK_FALSE(std::is_convertible_v<prop::in_order, queue>);
    STATIC_CHECK_FALSE(std::is_convertible_v<prop::enable_profiling, queue>);
    STATIC_CHECK_FALSE(std::is_convertible_v<queue_props_t, queue>);
    STATIC_CHECK_FALSE(std::is_convertible_v<khr::empty_properties_t, queue>);
  }
}

TEST_CASE(
    "queue::is_in_order and khr_is_profiling_enabled have the specified "
    "signatures",
    "[khr_properties][queue]") {
  const sycl::queue q = sycl_cts::util::get_cts_object::queue();
  STATIC_CHECK(std::is_same_v<decltype(q.is_in_order()), bool>);
  STATIC_CHECK(std::is_same_v<decltype(q.khr_is_profiling_enabled()), bool>);
}

TEST_CASE(
    "queue::is_in_order and khr_is_profiling_enabled for queues without "
    "the new properties",
    "[khr_properties][queue]") {
  const sycl::device dev = sycl_cts::util::get_cts_object::device();
  SECTION("default queue") {
    sycl::queue q{dev};
    CHECK_FALSE(q.is_in_order());
    CHECK_FALSE(q.khr_is_profiling_enabled());
  }
  SECTION("queue with property::queue::in_order") {
    sycl::queue q{dev, sycl::property::queue::in_order{}};
    CHECK(q.is_in_order());
    CHECK_FALSE(q.khr_is_profiling_enabled());
  }
  SECTION("queue with property::queue::enable_profiling") {
    if (!dev.has(sycl::aspect::queue_profiling)) {
      SKIP("Device does not have aspect::queue_profiling");
    }
    sycl::queue q{dev, sycl::property::queue::enable_profiling{}};
    CHECK_FALSE(q.is_in_order());
    CHECK(q.khr_is_profiling_enabled());
  }
  SECTION("queue with an empty properties list") {
    sycl::queue q{dev, khr::empty_properties_t{}};
    CHECK_FALSE(q.is_in_order());
    CHECK_FALSE(q.khr_is_profiling_enabled());
  }
}

TEST_CASE("queues constructed with khr::property::in_order{true} are in-order",
          "[khr_properties][queue]") {
  SECTION("as a property") {
    for_each_queue_ctor(prop::in_order{true},
                        [](sycl::queue& q, auto ctx, auto dev) {
                          check_ctx_dev(q, ctx, dev);
                          CHECK(q.is_in_order());
                          CHECK_FALSE(q.khr_is_profiling_enabled());
                          check_in_order_execution(q);
                        });
  }
  SECTION("in a properties list") {
    for_each_queue_ctor(khr::properties{prop::in_order{}},
                        [](sycl::queue& q, auto ctx, auto dev) {
                          check_ctx_dev(q, ctx, dev);
                          CHECK(q.is_in_order());
                          CHECK_FALSE(q.khr_is_profiling_enabled());
                          check_in_order_execution(q);
                        });
  }
}

TEST_CASE("khr::property::in_order{false} has no effect on the queue",
          "[khr_properties][queue]") {
  SECTION("as a property") {
    for_each_queue_ctor(prop::in_order{false},
                        [](sycl::queue& q, auto ctx, auto dev) {
                          check_ctx_dev(q, ctx, dev);
                          CHECK_FALSE(q.is_in_order());
                          CHECK_FALSE(q.khr_is_profiling_enabled());
                        });
  }
  SECTION("in a properties list") {
    for_each_queue_ctor(khr::properties{prop::in_order{false}},
                        [](sycl::queue& q, auto ctx, auto dev) {
                          check_ctx_dev(q, ctx, dev);
                          CHECK_FALSE(q.is_in_order());
                          CHECK_FALSE(q.khr_is_profiling_enabled());
                        });
  }
}

TEST_CASE(
    "queues constructed with khr::property::enable_profiling{true} have "
    "profiling enabled",
    "[khr_properties][queue]") {
  if (!device_has_queue_profiling()) {
    SKIP("Device does not have aspect::queue_profiling");
  }
  SECTION("as a property") {
    for_each_queue_ctor(prop::enable_profiling{true},
                        [](sycl::queue& q, auto ctx, auto dev) {
                          check_ctx_dev(q, ctx, dev);
                          CHECK(q.khr_is_profiling_enabled());
                          CHECK_FALSE(q.is_in_order());
                          CHECK(profiling_info_is_available(q));
                        });
  }
  SECTION("in a properties list") {
    for_each_queue_ctor(khr::properties{prop::enable_profiling{}},
                        [](sycl::queue& q, auto ctx, auto dev) {
                          check_ctx_dev(q, ctx, dev);
                          CHECK(q.khr_is_profiling_enabled());
                          CHECK_FALSE(q.is_in_order());
                          CHECK(profiling_info_is_available(q));
                        });
  }
}

TEST_CASE(
    "constructing a queue with khr::property::enable_profiling{true} for a "
    "device without aspect::queue_profiling throws",
    "[khr_properties][queue]") {
  const sycl::device dev = sycl_cts::util::get_cts_object::device();
  if (dev.has(sycl::aspect::queue_profiling)) {
    SKIP("Device has aspect::queue_profiling");
  }
  CHECK_THROWS_MATCHES(
      sycl::queue(dev, prop::enable_profiling{true}), sycl::exception,
      sycl_cts::util::equals_exception(sycl::errc::feature_not_supported));
  CHECK_NOTHROW(sycl::queue(dev, prop::enable_profiling{false}));
}

TEST_CASE("khr::property::enable_profiling{false} has no effect on the queue",
          "[khr_properties][queue]") {
  SECTION("as a property") {
    for_each_queue_ctor(prop::enable_profiling{false},
                        [](sycl::queue& q, auto ctx, auto dev) {
                          check_ctx_dev(q, ctx, dev);
                          CHECK_FALSE(q.khr_is_profiling_enabled());
                          CHECK_FALSE(profiling_info_is_available(q));
                        });
  }
  SECTION("in a properties list") {
    for_each_queue_ctor(khr::properties{prop::enable_profiling{false}},
                        [](sycl::queue& q, auto ctx, auto dev) {
                          check_ctx_dev(q, ctx, dev);
                          CHECK_FALSE(q.khr_is_profiling_enabled());
                          CHECK_FALSE(profiling_info_is_available(q));
                        });
  }
}

TEST_CASE("queues constructed with both queue properties",
          "[khr_properties][queue]") {
  if (!device_has_queue_profiling()) {
    SKIP("Device does not have aspect::queue_profiling");
  }
  const bool in_order = GENERATE(true, false);
  const bool profiling = GENERATE(true, false);
  CAPTURE(in_order, profiling);
  for_each_queue_ctor(khr::properties{prop::in_order{in_order},
                                      prop::enable_profiling{profiling}},
                      [=](sycl::queue& q, auto ctx, auto dev) {
                        check_ctx_dev(q, ctx, dev);
                        CHECK(q.is_in_order() == in_order);
                        CHECK(q.khr_is_profiling_enabled() == profiling);
                        CHECK(profiling_info_is_available(q) == profiling);
                        if (in_order) check_in_order_execution(q);
                      });
}

TEST_CASE("queues constructed with an empty properties list",
          "[khr_properties][queue]") {
  for_each_queue_ctor(khr::empty_properties_t{},
                      [](sycl::queue& q, auto ctx, auto dev) {
                        check_ctx_dev(q, ctx, dev);
                        CHECK_FALSE(q.is_in_order());
                        CHECK_FALSE(q.khr_is_profiling_enabled());
                      });
}

/**
 * Checks that `q` "contains" the old-style property OldProperty only if
 * `expected` is true. If it does not, get_property throws errc::invalid as
 * described in the core specification.
 */
template <typename OldProperty>
void check_old_style_property(const sycl::queue& q, bool expected) {
  CHECK(q.has_property<OldProperty>() == expected);
  if (expected) {
    CHECK_NOTHROW(q.get_property<OldProperty>());
    STATIC_CHECK(
        std::is_same_v<decltype(q.get_property<OldProperty>()), OldProperty>);
  } else {
    CHECK_THROWS_MATCHES(q.get_property<OldProperty>(), sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }
}

TEST_CASE(
    "queues constructed with khr::property::in_order report the old-style "
    "property::queue::in_order property",
    "[khr_properties][queue]") {
  const sycl::device dev = sycl_cts::util::get_cts_object::device();
  const bool value = GENERATE(true, false);
  CAPTURE(value);
  SECTION("as a property") {
    sycl::queue q{dev, prop::in_order{value}};
    check_old_style_property<sycl::property::queue::in_order>(q, value);
    check_old_style_property<sycl::property::queue::enable_profiling>(q, false);
  }
  SECTION("in a properties list") {
    sycl::queue q{dev, khr::properties{prop::in_order{value}}};
    check_old_style_property<sycl::property::queue::in_order>(q, value);
    check_old_style_property<sycl::property::queue::enable_profiling>(q, false);
  }
}

TEST_CASE(
    "queues constructed with khr::property::enable_profiling report the "
    "old-style property::queue::enable_profiling property",
    "[khr_properties][queue]") {
  const sycl::device dev = sycl_cts::util::get_cts_object::device();
  const bool value = GENERATE(true, false);
  CAPTURE(value);
  if (value && !dev.has(sycl::aspect::queue_profiling)) {
    SKIP("Device does not have aspect::queue_profiling");
  }
  SECTION("as a property") {
    sycl::queue q{dev, prop::enable_profiling{value}};
    check_old_style_property<sycl::property::queue::enable_profiling>(q, value);
    check_old_style_property<sycl::property::queue::in_order>(q, false);
  }
  SECTION("in a properties list") {
    sycl::queue q{dev, khr::properties{prop::enable_profiling{value}}};
    check_old_style_property<sycl::property::queue::enable_profiling>(q, value);
    check_old_style_property<sycl::property::queue::in_order>(q, false);
  }
}

TEST_CASE(
    "queues constructed with an empty properties list do not report "
    "old-style queue properties",
    "[khr_properties][queue]") {
  const sycl::device dev = sycl_cts::util::get_cts_object::device();
  sycl::queue q{dev, khr::empty_properties_t{}};
  check_old_style_property<sycl::property::queue::in_order>(q, false);
  check_old_style_property<sycl::property::queue::enable_profiling>(q, false);
}

}  // namespace khr_properties::tests
