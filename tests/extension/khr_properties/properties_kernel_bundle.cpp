/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Tests for the compile, link and build overloads added by the
//  sycl_khr_properties extension.
//
//  The extension does not define any property that can be used with these
//  functions and applications cannot define their own properties, so the only
//  PropertyOrList argument that satisfies the constraints is the empty
//  properties list.
//
*******************************************************************************/

#include "khr_properties_common.h"

#include <vector>

namespace khr_properties::tests {

class khr_props_kernel_bundle_kernel;

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
    check_kernel_bundle_constraints<prop::initialize_to_identity>(false);
    check_kernel_bundle_constraints<queue_props_t>(false);
    check_kernel_bundle_constraints<reduction_props_t>(false);
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

}  // namespace khr_properties::tests
