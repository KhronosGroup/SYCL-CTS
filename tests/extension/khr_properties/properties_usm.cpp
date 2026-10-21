/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Tests for the USM additions of the sycl_khr_properties extension: the
//  usm_allocator constructors and the device, host, shared and parameterized
//  allocation function overloads.
//
//  The extension does not define any USM allocation property and applications
//  cannot define their own properties, so the only PropertyOrList argument
//  that satisfies the constraints is the empty properties list. Properties of
//  other classes are used to check that the constraints remove the overloads
//  from overload resolution.
//
*******************************************************************************/

#include "khr_properties_common.h"

#include <vector>

namespace khr_properties::tests {

constexpr std::size_t alloc_count = 16;
constexpr std::size_t alloc_alignment = 64;

using q_t = const sycl::queue&;
using d_t = const sycl::device&;
using c_t = const sycl::context&;
using sz = std::size_t;
using kind_t = sycl::usm::alloc;

// Detection traits for each overload. P is the type of the PropertyOrList
// argument.
KHR_PROPERTIES_DETECTOR(can_malloc_device_dc,
                        sycl::malloc_device(sz{}, std::declval<d_t>(),
                                            std::declval<c_t>(),
                                            std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_device_dc_typed,
                        sycl::malloc_device<int>(sz{}, std::declval<d_t>(),
                                                 std::declval<c_t>(),
                                                 std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_device_q,
                        sycl::malloc_device(sz{}, std::declval<q_t>(),
                                            std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_device_q_typed,
                        sycl::malloc_device<int>(sz{}, std::declval<q_t>(),
                                                 std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_device_dc,
                        sycl::aligned_alloc_device(sz{}, sz{},
                                                   std::declval<d_t>(),
                                                   std::declval<c_t>(),
                                                   std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_device_dc_typed,
                        sycl::aligned_alloc_device<int>(sz{}, sz{},
                                                        std::declval<d_t>(),
                                                        std::declval<c_t>(),
                                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_device_q,
                        sycl::aligned_alloc_device(sz{}, sz{},
                                                   std::declval<q_t>(),
                                                   std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_device_q_typed,
                        sycl::aligned_alloc_device<int>(sz{}, sz{},
                                                        std::declval<q_t>(),
                                                        std::declval<P>()))

KHR_PROPERTIES_DETECTOR(can_malloc_host_c,
                        sycl::malloc_host(sz{}, std::declval<c_t>(),
                                          std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_host_c_typed,
                        sycl::malloc_host<int>(sz{}, std::declval<c_t>(),
                                               std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_host_q,
                        sycl::malloc_host(sz{}, std::declval<q_t>(),
                                          std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_host_q_typed,
                        sycl::malloc_host<int>(sz{}, std::declval<q_t>(),
                                               std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_host_c,
                        sycl::aligned_alloc_host(sz{}, sz{},
                                                 std::declval<c_t>(),
                                                 std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_host_c_typed,
                        sycl::aligned_alloc_host<int>(sz{}, sz{},
                                                      std::declval<c_t>(),
                                                      std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_host_q,
                        sycl::aligned_alloc_host(sz{}, sz{},
                                                 std::declval<q_t>(),
                                                 std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_host_q_typed,
                        sycl::aligned_alloc_host<int>(sz{}, sz{},
                                                      std::declval<q_t>(),
                                                      std::declval<P>()))

KHR_PROPERTIES_DETECTOR(can_malloc_shared_dc,
                        sycl::malloc_shared(sz{}, std::declval<d_t>(),
                                            std::declval<c_t>(),
                                            std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_shared_dc_typed,
                        sycl::malloc_shared<int>(sz{}, std::declval<d_t>(),
                                                 std::declval<c_t>(),
                                                 std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_shared_q,
                        sycl::malloc_shared(sz{}, std::declval<q_t>(),
                                            std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_shared_q_typed,
                        sycl::malloc_shared<int>(sz{}, std::declval<q_t>(),
                                                 std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_shared_dc,
                        sycl::aligned_alloc_shared(sz{}, sz{},
                                                   std::declval<d_t>(),
                                                   std::declval<c_t>(),
                                                   std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_shared_dc_typed,
                        sycl::aligned_alloc_shared<int>(sz{}, sz{},
                                                        std::declval<d_t>(),
                                                        std::declval<c_t>(),
                                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_shared_q,
                        sycl::aligned_alloc_shared(sz{}, sz{},
                                                   std::declval<q_t>(),
                                                   std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_shared_q_typed,
                        sycl::aligned_alloc_shared<int>(sz{}, sz{},
                                                        std::declval<q_t>(),
                                                        std::declval<P>()))

KHR_PROPERTIES_DETECTOR(can_malloc_dc,
                        sycl::malloc(sz{}, std::declval<d_t>(),
                                     std::declval<c_t>(), kind_t{},
                                     std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_dc_typed,
                        sycl::malloc<int>(sz{}, std::declval<d_t>(),
                                          std::declval<c_t>(), kind_t{},
                                          std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_q, sycl::malloc(sz{}, std::declval<q_t>(),
                                                   kind_t{}, std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_malloc_q_typed,
                        sycl::malloc<int>(sz{}, std::declval<q_t>(), kind_t{},
                                          std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_dc,
                        sycl::aligned_alloc(sz{}, sz{}, std::declval<d_t>(),
                                            std::declval<c_t>(), kind_t{},
                                            std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_dc_typed,
                        sycl::aligned_alloc<int>(sz{}, sz{},
                                                 std::declval<d_t>(),
                                                 std::declval<c_t>(), kind_t{},
                                                 std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_q,
                        sycl::aligned_alloc(sz{}, sz{}, std::declval<q_t>(),
                                            kind_t{}, std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_aligned_alloc_q_typed,
                        sycl::aligned_alloc<int>(sz{}, sz{},
                                                 std::declval<q_t>(), kind_t{},
                                                 std::declval<P>()))

template <typename P>
void check_usm_constraints(bool expected) {
  CHECK(can_malloc_device_dc_v<P> == expected);
  CHECK(can_malloc_device_dc_typed_v<P> == expected);
  CHECK(can_malloc_device_q_v<P> == expected);
  CHECK(can_malloc_device_q_typed_v<P> == expected);
  CHECK(can_aligned_alloc_device_dc_v<P> == expected);
  CHECK(can_aligned_alloc_device_dc_typed_v<P> == expected);
  CHECK(can_aligned_alloc_device_q_v<P> == expected);
  CHECK(can_aligned_alloc_device_q_typed_v<P> == expected);

  CHECK(can_malloc_host_c_v<P> == expected);
  CHECK(can_malloc_host_c_typed_v<P> == expected);
  CHECK(can_malloc_host_q_v<P> == expected);
  CHECK(can_malloc_host_q_typed_v<P> == expected);
  CHECK(can_aligned_alloc_host_c_v<P> == expected);
  CHECK(can_aligned_alloc_host_c_typed_v<P> == expected);
  CHECK(can_aligned_alloc_host_q_v<P> == expected);
  CHECK(can_aligned_alloc_host_q_typed_v<P> == expected);

  CHECK(can_malloc_shared_dc_v<P> == expected);
  CHECK(can_malloc_shared_dc_typed_v<P> == expected);
  CHECK(can_malloc_shared_q_v<P> == expected);
  CHECK(can_malloc_shared_q_typed_v<P> == expected);
  CHECK(can_aligned_alloc_shared_dc_v<P> == expected);
  CHECK(can_aligned_alloc_shared_dc_typed_v<P> == expected);
  CHECK(can_aligned_alloc_shared_q_v<P> == expected);
  CHECK(can_aligned_alloc_shared_q_typed_v<P> == expected);

  CHECK(can_malloc_dc_v<P> == expected);
  CHECK(can_malloc_dc_typed_v<P> == expected);
  CHECK(can_malloc_q_v<P> == expected);
  CHECK(can_malloc_q_typed_v<P> == expected);
  CHECK(can_aligned_alloc_dc_v<P> == expected);
  CHECK(can_aligned_alloc_dc_typed_v<P> == expected);
  CHECK(can_aligned_alloc_q_v<P> == expected);
  CHECK(can_aligned_alloc_q_typed_v<P> == expected);

  using host_alloc_t = sycl::usm_allocator<int, sycl::usm::alloc::host>;
  using shared_alloc_t = sycl::usm_allocator<int, sycl::usm::alloc::shared>;
  CHECK(std::is_constructible_v<host_alloc_t, c_t, d_t, P> == expected);
  CHECK(std::is_constructible_v<host_alloc_t, q_t, P> == expected);
  CHECK(std::is_constructible_v<shared_alloc_t, c_t, d_t, P> == expected);
  CHECK(std::is_constructible_v<shared_alloc_t, q_t, P> == expected);
}

TEST_CASE("the new USM overloads are constrained on PropertyOrList",
          "[khr_properties][usm]") {
  SECTION("the empty properties list is accepted") {
    check_usm_constraints<khr::empty_properties_t>(true);
  }
  SECTION("queue properties are rejected") {
    check_usm_constraints<prop::in_order>(false);
    check_usm_constraints<prop::enable_profiling>(false);
  }
  SECTION("reduction properties are rejected") {
    check_usm_constraints<prop::initialize_to_identity>(false);
  }
  SECTION("properties lists with properties of other classes are rejected") {
    check_usm_constraints<queue_props_t>(false);
    check_usm_constraints<reduction_props_t>(false);
  }
  SECTION("types that are neither properties nor lists are rejected") {
    check_usm_constraints<not_a_property>(false);
    check_usm_constraints<key::in_order>(false);
  }
}

TEST_CASE("the new USM allocation functions have the specified return types",
          "[khr_properties][usm]") {
  using e_t = khr::empty_properties_t;
  STATIC_CHECK(std::is_same_v<decltype(sycl::malloc_device(
                                  sz{}, std::declval<q_t>(), e_t{})),
                              void*>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::malloc_device<int>(
                                  sz{}, std::declval<q_t>(), e_t{})),
                              int*>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::aligned_alloc_host<int>(
                                  sz{}, sz{}, std::declval<c_t>(), e_t{})),
                              int*>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::malloc_shared(sz{}, std::declval<d_t>(),
                                                  std::declval<c_t>(), e_t{})),
                     void*>);
  STATIC_CHECK(std::is_same_v<decltype(sycl::malloc<int>(
                                  sz{}, std::declval<q_t>(), kind_t{}, e_t{})),
                              int*>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::aligned_alloc(
                         sz{}, sz{}, std::declval<q_t>(), kind_t{}, e_t{})),
                     void*>);
}

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

}  // namespace khr_properties::tests
