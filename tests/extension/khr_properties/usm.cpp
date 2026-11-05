/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Compile-time tests for the USM additions of the sycl_khr_properties
//  extension: the usm_allocator constructors and the device, host, shared and
//  parameterized allocation function overloads. The runtime effects are
//  tested in runtime.cpp.
//
//  The extension does not define any USM allocation property and applications
//  cannot define their own properties, so the only PropertyOrList argument
//  that satisfies the constraints is the empty properties list. Properties of
//  other classes are used to check that the constraints remove the overloads
//  from overload resolution.
//
*******************************************************************************/

#include "khr_properties_common.h"

#if SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS
#include <sycl/khr/split_headers/context.hpp>
#include <sycl/khr/split_headers/device.hpp>
#include <sycl/khr/split_headers/usm.hpp>
#endif

#include <cstddef>

namespace khr_properties::tests {

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
  SECTION("properties lists with properties of other classes are rejected") {
    check_usm_constraints<queue_props_t>(false);
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

TEST_CASE("only the empty properties list applies to the USM tags",
          "[khr_properties][usm][traits]") {
  namespace tag = sycl::khr::property::tag;
  check_class_without_properties<tag::usm_device>();
  check_class_without_properties<tag::usm_host>();
  check_class_without_properties<tag::usm_shared>();
}

}  // namespace khr_properties::tests
