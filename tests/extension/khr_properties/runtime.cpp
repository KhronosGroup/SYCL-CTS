/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Runtime tests for the sycl_khr_properties extension: the effects of the
//  new context, queue, USM, reduction, kernel bundle and stream APIs. The
//  compile-time requirements of these APIs are tested in the test file for
//  each header. These tests need a device, so they use the CTS helpers and
//  <sycl/sycl.hpp>.
//
//  The extension does not define any property for the context, stream, USM
//  and kernel bundle APIs, and applications cannot define their own
//  properties, so these APIs are tested with the empty properties list.
//
*******************************************************************************/

#include "../../../util/sycl_exceptions.h"
#include "../../common/common.h"

#include "khr_properties_common.h"

#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace khr_properties::tests {

// ----------------------------------------------------------------------------
// Context
// ----------------------------------------------------------------------------

bool contains_device(const std::vector<sycl::device>& devices,
                     const sycl::device& dev) {
  return std::find(devices.begin(), devices.end(), dev) != devices.end();
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

// ----------------------------------------------------------------------------
// Queue
// ----------------------------------------------------------------------------

class khr_props_queue_kernel_set;
class khr_props_queue_kernel_update;
class khr_props_queue_kernel_profiling;

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

// ----------------------------------------------------------------------------
// USM
// ----------------------------------------------------------------------------

/** Returns true if `ptr` is aligned to `alignment` bytes. */
inline bool is_aligned(const void* ptr, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

constexpr std::size_t alloc_count = 16;
constexpr std::size_t alloc_alignment = 64;

/**
 * Checks that `ptr` is a non-null USM allocation of the expected kind, aligned
 * to `alignment` (if non-zero), and frees it.
 */
void check_and_free(void* ptr, const sycl::context& ctx,
                    sycl::usm::alloc expected, std::size_t alignment = 0) {
  REQUIRE(ptr != nullptr);
  CHECK(sycl::get_pointer_type(ptr, ctx) == expected);
  if (alignment != 0) CHECK(is_aligned(ptr, alignment));
  sycl::free(ptr, ctx);
}

constexpr std::size_t num_bytes = alloc_count * sizeof(int);

TEST_CASE("the new device USM allocation functions allocate device USM",
          "[khr_properties][usm]") {
  auto q = sycl_cts::util::get_cts_object::queue();
  const auto dev = q.get_device();
  const auto ctx = q.get_context();
  if (!dev.has(sycl::aspect::usm_device_allocations)) {
    SKIP("Device does not have aspect::usm_device_allocations");
  }
  const khr::empty_properties_t props{};
  const auto kind = sycl::usm::alloc::device;
  const auto align = alloc_alignment;

  check_and_free(sycl::malloc_device(num_bytes, dev, ctx, props), ctx, kind);
  check_and_free(sycl::malloc_device<int>(alloc_count, dev, ctx, props), ctx,
                 kind);
  check_and_free(sycl::malloc_device(num_bytes, q, props), ctx, kind);
  check_and_free(sycl::malloc_device<int>(alloc_count, q, props), ctx, kind);
  check_and_free(sycl::aligned_alloc_device(align, num_bytes, dev, ctx, props),
                 ctx, kind, align);
  check_and_free(
      sycl::aligned_alloc_device<int>(align, alloc_count, dev, ctx, props), ctx,
      kind, align);
  check_and_free(sycl::aligned_alloc_device(align, num_bytes, q, props), ctx,
                 kind, align);
  check_and_free(sycl::aligned_alloc_device<int>(align, alloc_count, q, props),
                 ctx, kind, align);
}

TEST_CASE("the new host USM allocation functions allocate host USM",
          "[khr_properties][usm]") {
  auto q = sycl_cts::util::get_cts_object::queue();
  const auto dev = q.get_device();
  const auto ctx = q.get_context();
  if (!dev.has(sycl::aspect::usm_host_allocations)) {
    SKIP("Device does not have aspect::usm_host_allocations");
  }
  const khr::empty_properties_t props{};
  const auto kind = sycl::usm::alloc::host;
  const auto align = alloc_alignment;

  check_and_free(sycl::malloc_host(num_bytes, ctx, props), ctx, kind);
  check_and_free(sycl::malloc_host<int>(alloc_count, ctx, props), ctx, kind);
  check_and_free(sycl::malloc_host(num_bytes, q, props), ctx, kind);
  check_and_free(sycl::malloc_host<int>(alloc_count, q, props), ctx, kind);
  check_and_free(sycl::aligned_alloc_host(align, num_bytes, ctx, props), ctx,
                 kind, align);
  check_and_free(sycl::aligned_alloc_host<int>(align, alloc_count, ctx, props),
                 ctx, kind, align);
  check_and_free(sycl::aligned_alloc_host(align, num_bytes, q, props), ctx,
                 kind, align);
  check_and_free(sycl::aligned_alloc_host<int>(align, alloc_count, q, props),
                 ctx, kind, align);
}

TEST_CASE("the new shared USM allocation functions allocate shared USM",
          "[khr_properties][usm]") {
  auto q = sycl_cts::util::get_cts_object::queue();
  const auto dev = q.get_device();
  const auto ctx = q.get_context();
  if (!dev.has(sycl::aspect::usm_shared_allocations)) {
    SKIP("Device does not have aspect::usm_shared_allocations");
  }
  const khr::empty_properties_t props{};
  const auto kind = sycl::usm::alloc::shared;
  const auto align = alloc_alignment;

  check_and_free(sycl::malloc_shared(num_bytes, dev, ctx, props), ctx, kind);
  check_and_free(sycl::malloc_shared<int>(alloc_count, dev, ctx, props), ctx,
                 kind);
  check_and_free(sycl::malloc_shared(num_bytes, q, props), ctx, kind);
  check_and_free(sycl::malloc_shared<int>(alloc_count, q, props), ctx, kind);
  check_and_free(sycl::aligned_alloc_shared(align, num_bytes, dev, ctx, props),
                 ctx, kind, align);
  check_and_free(
      sycl::aligned_alloc_shared<int>(align, alloc_count, dev, ctx, props), ctx,
      kind, align);
  check_and_free(sycl::aligned_alloc_shared(align, num_bytes, q, props), ctx,
                 kind, align);
  check_and_free(sycl::aligned_alloc_shared<int>(align, alloc_count, q, props),
                 ctx, kind, align);
}

TEST_CASE(
    "the new parameterized USM allocation functions allocate the requested "
    "kind of USM",
    "[khr_properties][usm]") {
  auto q = sycl_cts::util::get_cts_object::queue();
  const auto dev = q.get_device();
  const auto ctx = q.get_context();
  const khr::empty_properties_t props{};
  const auto align = alloc_alignment;

  const auto kind = GENERATE(sycl::usm::alloc::device, sycl::usm::alloc::host,
                             sycl::usm::alloc::shared);
  CAPTURE(kind);
  if ((kind == sycl::usm::alloc::device &&
       !dev.has(sycl::aspect::usm_device_allocations)) ||
      (kind == sycl::usm::alloc::host &&
       !dev.has(sycl::aspect::usm_host_allocations)) ||
      (kind == sycl::usm::alloc::shared &&
       !dev.has(sycl::aspect::usm_shared_allocations))) {
    SKIP("Device does not support the requested kind of allocation");
  }

  // The empty properties list applies to all USM tags, so none of the
  // functions may throw errc::invalid.
  check_and_free(sycl::malloc(num_bytes, dev, ctx, kind, props), ctx, kind);
  check_and_free(sycl::malloc<int>(alloc_count, dev, ctx, kind, props), ctx,
                 kind);
  check_and_free(sycl::malloc(num_bytes, q, kind, props), ctx, kind);
  check_and_free(sycl::malloc<int>(alloc_count, q, kind, props), ctx, kind);
  check_and_free(sycl::aligned_alloc(align, num_bytes, dev, ctx, kind, props),
                 ctx, kind, align);
  check_and_free(
      sycl::aligned_alloc<int>(align, alloc_count, dev, ctx, kind, props), ctx,
      kind, align);
  check_and_free(sycl::aligned_alloc(align, num_bytes, q, kind, props), ctx,
                 kind, align);
  check_and_free(sycl::aligned_alloc<int>(align, alloc_count, q, kind, props),
                 ctx, kind, align);
}

template <sycl::usm::alloc Kind>
void check_usm_allocator(sycl::aspect required_aspect) {
  auto q = sycl_cts::util::get_cts_object::queue();
  const auto dev = q.get_device();
  const auto ctx = q.get_context();
  if (!dev.has(required_aspect)) {
    SKIP("Device does not support the allocation kind");
  }
  const khr::empty_properties_t props{};

  SECTION("usm_allocator(syclContext, syclDevice, props)") {
    sycl::usm_allocator<int, Kind> allocator{ctx, dev, props};
    int* ptr = allocator.allocate(alloc_count);
    REQUIRE(ptr != nullptr);
    CHECK(sycl::get_pointer_type(ptr, ctx) == Kind);
    allocator.deallocate(ptr, alloc_count);
    CHECK(allocator == sycl::usm_allocator<int, Kind>{ctx, dev});
  }
  SECTION("usm_allocator(syclQueue, props)") {
    sycl::usm_allocator<int, Kind> allocator{q, props};
    std::vector<int, sycl::usm_allocator<int, Kind>> vec(alloc_count, 7,
                                                         allocator);
    CHECK(sycl::get_pointer_type(vec.data(), ctx) == Kind);
    CHECK(vec[alloc_count - 1] == 7);
    CHECK(allocator == sycl::usm_allocator<int, Kind>{q});
  }
}

TEST_CASE("usm_allocator constructed with an empty properties list",
          "[khr_properties][usm]") {
  SECTION("host") {
    check_usm_allocator<sycl::usm::alloc::host>(
        sycl::aspect::usm_host_allocations);
  }
  SECTION("shared") {
    check_usm_allocator<sycl::usm::alloc::shared>(
        sycl::aspect::usm_shared_allocations);
  }
}

// ----------------------------------------------------------------------------
// Reduction
// ----------------------------------------------------------------------------

constexpr std::size_t num_work_items = 64;
constexpr int initial_value = 100;

using plus_t = sycl::plus<int>;
using span2_t = sycl::span<int, 2>;

/** A combiner for which no identity value is known. */
struct custom_plus {
  int operator()(const int& a, const int& b) const { return a + b; }
};

/**
 * Runs a parallel_for over num_work_items work-items in which each work-item
 * combines 1 into the scalar reduction created by `make_reduction(ptr)`,
 * where `ptr` points to a device allocation initialized to initial_value.
 * Returns the final value of the reduction variable.
 */
template <typename MakeReduction>
int run_scalar_reduction(sycl::queue& q, MakeReduction make_reduction) {
  int* ptr = sycl::malloc_device<int>(1, q);
  REQUIRE(ptr != nullptr);
  int value = initial_value;
  q.copy(&value, ptr, 1).wait_and_throw();
  q.submit([&](sycl::handler& cgh) {
     cgh.parallel_for(sycl::range<1>{num_work_items}, make_reduction(ptr),
                      [=](sycl::id<1>, auto& reducer) { reducer.combine(1); });
   }).wait_and_throw();
  q.copy(ptr, &value, 1).wait_and_throw();
  sycl::free(ptr, q);
  return value;
}

/**
 * Like run_scalar_reduction, but for a span reduction of two elements. Each
 * work-item combines 1 into element 0 and 2 into element 1.
 */
template <typename MakeReduction>
std::array<int, 2> run_span_reduction(sycl::queue& q,
                                      MakeReduction make_reduction) {
  int* ptr = sycl::malloc_device<int>(2, q);
  REQUIRE(ptr != nullptr);
  std::array<int, 2> values{initial_value, initial_value};
  q.copy(values.data(), ptr, 2).wait_and_throw();
  q.submit([&](sycl::handler& cgh) {
     cgh.parallel_for(sycl::range<1>{num_work_items},
                      make_reduction(span2_t{ptr, 2}),
                      [=](sycl::id<1>, auto& reducer) {
                        reducer[0].combine(1);
                        reducer[1].combine(2);
                      });
   }).wait_and_throw();
  q.copy(ptr, values.data(), 2).wait_and_throw();
  sycl::free(ptr, q);
  return values;
}

constexpr int n = static_cast<int>(num_work_items);

/**
 * Runs all four reduction overloads with `props` and checks that the
 * original value of the reduction variable is included in the result only
 * if `initializes` is false.
 */
template <typename PropertyOrList>
void check_reduction_overloads(PropertyOrList props, bool initializes) {
  auto q = sycl_cts::util::get_cts_object::queue();
  if (!q.get_device().has(sycl::aspect::usm_device_allocations)) {
    SKIP("Device does not have aspect::usm_device_allocations");
  }
  const int base = initializes ? 0 : initial_value;

  SECTION("reduction(var, combiner, props)") {
    CHECK(run_scalar_reduction(q, [=](int* p) {
            return sycl::reduction(p, plus_t{}, props);
          }) == base + n);
  }
  SECTION("reduction(vars, combiner, props)") {
    auto result = run_span_reduction(
        q, [=](span2_t s) { return sycl::reduction(s, plus_t{}, props); });
    CHECK(result[0] == base + n);
    CHECK(result[1] == base + 2 * n);
  }
  SECTION("reduction(var, identity, combiner, props)") {
    CHECK(run_scalar_reduction(q, [=](int* p) {
            return sycl::reduction(p, 0, plus_t{}, props);
          }) == base + n);
  }
  SECTION("reduction(vars, identity, combiner, props)") {
    auto result = run_span_reduction(
        q, [=](span2_t s) { return sycl::reduction(s, 0, plus_t{}, props); });
    CHECK(result[0] == base + n);
    CHECK(result[1] == base + 2 * n);
  }
  SECTION("reduction(var, identity, custom combiner, props)") {
    CHECK(run_scalar_reduction(q, [=](int* p) {
            return sycl::reduction(p, 0, custom_plus{}, props);
          }) == base + n);
  }
}

TEST_CASE(
    "reductions with khr::property::initialize_to_identity{true} do not "
    "include the original value",
    "[khr_properties][reduction]") {
  SECTION("as a property") {
    check_reduction_overloads(prop::initialize_to_identity{true}, true);
  }
  SECTION("default-constructed property") {
    check_reduction_overloads(prop::initialize_to_identity{}, true);
  }
  SECTION("in a properties list") {
    check_reduction_overloads(khr::properties{prop::initialize_to_identity{}},
                              true);
  }
}

TEST_CASE(
    "khr::property::initialize_to_identity{false} has no effect on the "
    "reduction",
    "[khr_properties][reduction]") {
  SECTION("as a property") {
    check_reduction_overloads(prop::initialize_to_identity{false}, false);
  }
  SECTION("in a properties list") {
    check_reduction_overloads(
        khr::properties{prop::initialize_to_identity{false}}, false);
  }
}

TEST_CASE("reductions with an empty properties list include the original value",
          "[khr_properties][reduction]") {
  check_reduction_overloads(khr::empty_properties_t{}, false);
}

TEST_CASE(
    "reductions with khr::property::initialize_to_identity{true} and no known "
    "identity throw errc::invalid",
    "[khr_properties][reduction]") {
  STATIC_CHECK_FALSE(sycl::has_known_identity_v<custom_plus, int>);
  auto q = sycl_cts::util::get_cts_object::queue();
  if (!q.get_device().has(sycl::aspect::usm_device_allocations)) {
    SKIP("Device does not have aspect::usm_device_allocations");
  }
  int* ptr = sycl::malloc_device<int>(2, q);
  REQUIRE(ptr != nullptr);

  // The specification does not state whether the exception is thrown by the
  // reduction function or by the kernel invocation, so both are covered.
  SECTION("scalar reduction") {
    CHECK_THROWS_MATCHES(
        q.submit([&](sycl::handler& cgh) {
           cgh.parallel_for(
               sycl::range<1>{num_work_items},
               sycl::reduction(ptr, custom_plus{},
                               prop::initialize_to_identity{true}),
               [=](sycl::id<1>, auto& reducer) { reducer.combine(1); });
         }).wait_and_throw(),
        sycl::exception, sycl_cts::util::equals_exception(sycl::errc::invalid));
  }
  SECTION("span reduction") {
    CHECK_THROWS_MATCHES(
        q.submit([&](sycl::handler& cgh) {
           cgh.parallel_for(
               sycl::range<1>{num_work_items},
               sycl::reduction(
                   span2_t{ptr, 2}, custom_plus{},
                   khr::properties{prop::initialize_to_identity{true}}),
               [=](sycl::id<1>, auto& reducer) { reducer[0].combine(1); });
         }).wait_and_throw(),
        sycl::exception, sycl_cts::util::equals_exception(sycl::errc::invalid));
  }
  q.wait();
  sycl::free(ptr, q);
}

TEST_CASE(
    "reductions with khr::property::initialize_to_identity{false} and no "
    "known identity include the original value",
    "[khr_properties][reduction]") {
  auto q = sycl_cts::util::get_cts_object::queue();
  if (!q.get_device().has(sycl::aspect::usm_device_allocations)) {
    SKIP("Device does not have aspect::usm_device_allocations");
  }
  CHECK(run_scalar_reduction(q, [](int* p) {
          return sycl::reduction(p, custom_plus{},
                                 prop::initialize_to_identity{false});
        }) == initial_value + n);
}

// ----------------------------------------------------------------------------
// Kernel bundle
// ----------------------------------------------------------------------------

class khr_props_kernel_bundle_kernel;

using input_bundle_t = sycl::kernel_bundle<sycl::bundle_state::input>;
using object_bundle_t = sycl::kernel_bundle<sycl::bundle_state::object>;
using exe_bundle_t = sycl::kernel_bundle<sycl::bundle_state::executable>;

/** Runs the test kernel from `bundle` on `q` and checks its result. */
void check_bundle_runs(sycl::queue& q, const exe_bundle_t& bundle) {
  const auto kid = sycl::get_kernel_id<khr_props_kernel_bundle_kernel>();
  REQUIRE(bundle.has_kernel(kid));
  sycl::buffer<int, 1> buf{sycl::range<1>{1}};
  q.submit([&](sycl::handler& cgh) {
    cgh.use_kernel_bundle(bundle);
    sycl::accessor acc{buf, cgh, sycl::write_only};
    cgh.single_task<khr_props_kernel_bundle_kernel>([=] { acc[0] = 42; });
  });
  sycl::host_accessor acc{buf, sycl::read_only};
  CHECK(acc[0] == 42);
}

TEST_CASE("compile, link and build with an empty properties list",
          "[khr_properties][kernel_bundle]") {
  auto q = sycl_cts::util::get_cts_object::queue();
  const auto dev = q.get_device();
  const auto ctx = q.get_context();
  const std::vector<sycl::device> devs{dev};
  const auto kid = sycl::get_kernel_id<khr_props_kernel_bundle_kernel>();

  if (!sycl::has_kernel_bundle<sycl::bundle_state::input>(ctx, devs, {kid})) {
    SKIP("No kernel bundle in the input state for the test kernel");
  }
  const input_bundle_t input =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx, devs, {kid});
  const khr::empty_properties_t props{};

  SECTION("compile(inputBundle, devs, props)") {
    if (!dev.has(sycl::aspect::online_compiler)) {
      SKIP("Device does not have aspect::online_compiler");
    }
    object_bundle_t obj = sycl::compile(input, devs, props);
    CHECK(obj.get_devices() == devs);
    CHECK(obj.has_kernel(kid));
  }
  SECTION("compile(inputBundle, props)") {
    if (!dev.has(sycl::aspect::online_compiler)) {
      SKIP("Device does not have aspect::online_compiler");
    }
    object_bundle_t obj = sycl::compile(input, props);
    CHECK(obj.get_devices() == input.get_devices());
    CHECK(obj.has_kernel(kid));
  }
  SECTION("link overloads") {
    if (!dev.has(sycl::aspect::online_linker)) {
      SKIP("Device does not have aspect::online_linker");
    }
    object_bundle_t obj = sycl::compile(input, devs);
    const std::vector<object_bundle_t> objs{obj};

    SECTION("link(objectBundles, devs, props)") {
      exe_bundle_t exe = sycl::link(objs, devs, props);
      CHECK(exe.get_devices() == devs);
      check_bundle_runs(q, exe);
    }
    SECTION("link(objectBundle, devs, props)") {
      exe_bundle_t exe = sycl::link(obj, devs, props);
      CHECK(exe.get_devices() == devs);
      check_bundle_runs(q, exe);
    }
    SECTION("link(objectBundles, props)") {
      exe_bundle_t exe = sycl::link(objs, props);
      check_bundle_runs(q, exe);
    }
    SECTION("link(objectBundle, props)") {
      exe_bundle_t exe = sycl::link(obj, props);
      CHECK(exe.get_devices() == obj.get_devices());
      check_bundle_runs(q, exe);
    }
  }
  SECTION("build overloads") {
    if (!dev.has(sycl::aspect::online_linker)) {
      SKIP("Device does not have aspect::online_linker");
    }
    SECTION("build(inputBundle, devs, props)") {
      exe_bundle_t exe = sycl::build(input, devs, props);
      CHECK(exe.get_devices() == devs);
      check_bundle_runs(q, exe);
    }
    SECTION("build(inputBundle, props)") {
      exe_bundle_t exe = sycl::build(input, props);
      CHECK(exe.get_devices() == input.get_devices());
      check_bundle_runs(q, exe);
    }
  }
}

// ----------------------------------------------------------------------------
// Stream
// ----------------------------------------------------------------------------

class khr_props_stream_kernel;

constexpr std::size_t total_buffer_size = 1024;
constexpr std::size_t work_item_buffer_size = 256;

#if defined(__linux__)
/**
 * Redirects stdout to a temporary file for the lifetime of the object, so
 * that the output of a sycl::stream can be inspected.
 */
class stdout_capture {
 public:
  stdout_capture() {
    std::fflush(stdout);
    m_file = std::tmpfile();
    m_saved = dup(fileno(stdout));
    if (m_file && m_saved != -1) dup2(fileno(m_file), fileno(stdout));
  }
  ~stdout_capture() { restore(); }

  /** Restores stdout and returns everything that was written to it. */
  std::string restore() {
    if (m_saved == -1 || !m_file) return {};
    std::fflush(stdout);
    dup2(m_saved, fileno(stdout));
    close(m_saved);
    m_saved = -1;
    std::string result;
    std::rewind(m_file);
    char buf[256];
    std::size_t len;
    while ((len = std::fread(buf, 1, sizeof(buf), m_file)) > 0) {
      result.append(buf, len);
    }
    std::fclose(m_file);
    m_file = nullptr;
    return result;
  }

 private:
  std::FILE* m_file = nullptr;
  int m_saved = -1;
};
#endif

TEST_CASE("a stream constructed with an empty properties list",
          "[khr_properties][stream]") {
  auto q = sycl_cts::util::get_cts_object::queue();
  std::size_t size = 0;
  std::size_t wi_size = 0;
  {
#if defined(__linux__)
    stdout_capture capture;
#endif
    q.submit([&](sycl::handler& cgh) {
      sycl::stream os{total_buffer_size, work_item_buffer_size, cgh,
                      khr::empty_properties_t{}};
      size = os.get_size();
      wi_size = os.get_work_item_buffer_size();
      cgh.single_task<khr_props_stream_kernel>(
          [=] { os << "khr_properties stream " << 42 << sycl::endl; });
    });
    q.wait_and_throw();
#if defined(__linux__)
    const std::string output = capture.restore();
    CHECK(output.find("khr_properties stream 42") != std::string::npos);
#endif
  }
  CHECK(size == total_buffer_size);
  CHECK(wi_size == work_item_buffer_size);
}

}  // namespace khr_properties::tests
