/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Compile-time tests for the compile, link and build overloads added by the
//  sycl_khr_properties extension. The runtime effects are tested in
//  runtime.cpp.
//
//  The extension does not define any property that can be used with these
//  functions and applications cannot define their own properties, so the only
//  PropertyOrList argument that satisfies the constraints is the empty
//  properties list.
//
*******************************************************************************/

#include "khr_properties_common.h"

#if SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS
#include <sycl/khr/split_headers/device.hpp>
#include <sycl/khr/split_headers/kernel_bundle.hpp>
#endif

#include <vector>

namespace khr_properties::tests {

using input_bundle_t = sycl::kernel_bundle<sycl::bundle_state::input>;
using object_bundle_t = sycl::kernel_bundle<sycl::bundle_state::object>;
using exe_bundle_t = sycl::kernel_bundle<sycl::bundle_state::executable>;
using devs_t = const std::vector<sycl::device>&;

// Detection traits for each overload. P is the type of the PropertyOrList
// argument.
KHR_PROPERTIES_DETECTOR(can_compile_devs,
                        sycl::compile(std::declval<const input_bundle_t&>(),
                                      std::declval<devs_t>(),
                                      std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_compile,
                        sycl::compile(std::declval<const input_bundle_t&>(),
                                      std::declval<P>()))
KHR_PROPERTIES_DETECTOR(
    can_link_vec_devs,
    sycl::link(std::declval<const std::vector<object_bundle_t>&>(),
               std::declval<devs_t>(), std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_link_one_devs,
                        sycl::link(std::declval<const object_bundle_t&>(),
                                   std::declval<devs_t>(), std::declval<P>()))
KHR_PROPERTIES_DETECTOR(
    can_link_vec,
    sycl::link(std::declval<const std::vector<object_bundle_t>&>(),
               std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_link_one,
                        sycl::link(std::declval<const object_bundle_t&>(),
                                   std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_build_devs,
                        sycl::build(std::declval<const input_bundle_t&>(),
                                    std::declval<devs_t>(), std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_build,
                        sycl::build(std::declval<const input_bundle_t&>(),
                                    std::declval<P>()))

template <typename P>
void check_kernel_bundle_constraints(bool expected) {
  CHECK(can_compile_devs_v<P> == expected);
  CHECK(can_compile_v<P> == expected);
  CHECK(can_link_vec_devs_v<P> == expected);
  CHECK(can_link_one_devs_v<P> == expected);
  CHECK(can_link_vec_v<P> == expected);
  CHECK(can_link_one_v<P> == expected);
  CHECK(can_build_devs_v<P> == expected);
  CHECK(can_build_v<P> == expected);
}

TEST_CASE("the new compile, link and build overloads are constrained",
          "[khr_properties][kernel_bundle]") {
  SECTION("the empty properties list is accepted") {
    check_kernel_bundle_constraints<khr::empty_properties_t>(true);
  }
  SECTION("properties of other classes are rejected") {
    check_kernel_bundle_constraints<prop::in_order>(false);
    check_kernel_bundle_constraints<queue_props_t>(false);
  }
  SECTION("types that are neither properties nor lists are rejected") {
    check_kernel_bundle_constraints<not_a_property>(false);
  }
}

TEST_CASE(
    "the new compile, link and build overloads have the specified "
    "return types",
    "[khr_properties][kernel_bundle]") {
  using e_t = khr::empty_properties_t;
  STATIC_CHECK(std::is_same_v<decltype(sycl::compile(
                                  std::declval<const input_bundle_t&>(),
                                  std::declval<devs_t>(), e_t{})),
                              object_bundle_t>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::compile(
                         std::declval<const input_bundle_t&>(), e_t{})),
                     object_bundle_t>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::link(std::declval<const object_bundle_t&>(),
                                         e_t{})),
                     exe_bundle_t>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::link(
                         std::declval<const std::vector<object_bundle_t>&>(),
                         std::declval<devs_t>(), e_t{})),
                     exe_bundle_t>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::build(std::declval<const input_bundle_t&>(),
                                          e_t{})),
                     exe_bundle_t>);
  STATIC_CHECK(
      std::is_same_v<decltype(sycl::build(std::declval<const input_bundle_t&>(),
                                          std::declval<devs_t>(), e_t{})),
                     exe_bundle_t>);
}

TEST_CASE("only the empty properties list applies to the kernel bundle tags",
          "[khr_properties][kernel_bundle][traits]") {
  namespace tag = sycl::khr::property::tag;
  check_class_without_properties<tag::compile>();
  check_class_without_properties<tag::link>();
  check_class_without_properties<tag::build>();
}

}  // namespace khr_properties::tests
