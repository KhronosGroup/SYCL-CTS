/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Tests for the property traits of the sycl_khr_properties extension and for
//  the requirements on the properties and keys defined by the extension.
//
*******************************************************************************/

#include "khr_properties_common.h"

namespace khr_properties::tests {

// A properties list with a queue property and a reduction property.
using mixed_props_t =
    decltype(khr::properties{prop::in_order{}, prop::initialize_to_identity{}});

template <template <typename> class Trait, typename T, bool Expected>
void check_unary_trait() {
  using base_t = std::integral_constant<bool, Expected>;
  STATIC_CHECK(std::is_base_of_v<base_t, Trait<T>>);
  STATIC_CHECK(Trait<T>::value == Expected);
}

template <template <typename, typename> class Trait, typename T, typename Class,
          bool Expected>
void check_binary_trait() {
  using base_t = std::integral_constant<bool, Expected>;
  STATIC_CHECK(std::is_base_of_v<base_t, Trait<T, Class>>);
  STATIC_CHECK(Trait<T, Class>::value == Expected);
}

TEST_CASE("the _v variable templates have the value of their traits",
          "[khr_properties][traits]") {
  STATIC_CHECK(khr::is_property_v<prop::in_order> ==
               khr::is_property<prop::in_order>::value);
  STATIC_CHECK(khr::is_property_v<int> == khr::is_property<int>::value);
  STATIC_CHECK(khr::is_property_key_v<key::in_order> ==
               khr::is_property_key<key::in_order>::value);
  STATIC_CHECK(khr::is_property_key_v<int> == khr::is_property_key<int>::value);
  STATIC_CHECK(khr::is_property_key_compile_time_v<key::in_order> ==
               khr::is_property_key_compile_time<key::in_order>::value);
  STATIC_CHECK(khr::is_property_for_v<prop::in_order, sycl::queue> ==
               khr::is_property_for<prop::in_order, sycl::queue>::value);
  STATIC_CHECK(khr::is_property_key_for_v<key::in_order, sycl::queue> ==
               khr::is_property_key_for<key::in_order, sycl::queue>::value);
  STATIC_CHECK(khr::is_property_list_for_v<queue_props_t, sycl::queue> ==
               khr::is_property_list_for<queue_props_t, sycl::queue>::value);
  STATIC_CHECK(
      std::is_same_v<decltype(khr::is_property_v<prop::in_order>), const bool>);
}

TEST_CASE("is_property classifies new-style properties",
          "[khr_properties][traits]") {
  check_unary_trait<khr::is_property, prop::in_order, true>();
  check_unary_trait<khr::is_property, prop::enable_profiling, true>();
  check_unary_trait<khr::is_property, prop::initialize_to_identity, true>();

  check_unary_trait<khr::is_property, key::in_order, false>();
  check_unary_trait<khr::is_property, key::enable_profiling, false>();
  check_unary_trait<khr::is_property, key::initialize_to_identity, false>();
  check_unary_trait<khr::is_property, int, false>();
  check_unary_trait<khr::is_property, not_a_property, false>();
  check_unary_trait<khr::is_property, khr::empty_properties_t, false>();
  check_unary_trait<khr::is_property, queue_props_t, false>();
  // Old-style properties are not new-style properties.
  check_unary_trait<khr::is_property, sycl::property::queue::in_order, false>();
  check_unary_trait<khr::is_property, sycl::property::queue::enable_profiling,
                    false>();
  check_unary_trait<khr::is_property,
                    sycl::property::reduction::initialize_to_identity, false>();
}

TEST_CASE("is_property_key classifies property keys",
          "[khr_properties][traits]") {
  check_unary_trait<khr::is_property_key, key::in_order, true>();
  check_unary_trait<khr::is_property_key, key::enable_profiling, true>();
  check_unary_trait<khr::is_property_key, key::initialize_to_identity, true>();

  check_unary_trait<khr::is_property_key, prop::in_order, false>();
  check_unary_trait<khr::is_property_key, prop::enable_profiling, false>();
  check_unary_trait<khr::is_property_key, prop::initialize_to_identity,
                    false>();
  check_unary_trait<khr::is_property_key, int, false>();
  check_unary_trait<khr::is_property_key, not_a_property, false>();
  check_unary_trait<khr::is_property_key, khr::empty_properties_t, false>();
}

TEST_CASE(
    "is_property_key_compile_time is false for the keys of properties with "
    "runtime values",
    "[khr_properties][traits]") {
  check_unary_trait<khr::is_property_key_compile_time, key::in_order, false>();
  check_unary_trait<khr::is_property_key_compile_time, key::enable_profiling,
                    false>();
  check_unary_trait<khr::is_property_key_compile_time,
                    key::initialize_to_identity, false>();
  check_unary_trait<khr::is_property_key_compile_time, int, false>();
}

TEST_CASE("is_property_for classifies the classes a property applies to",
          "[khr_properties][traits]") {
  check_binary_trait<khr::is_property_for, prop::in_order, sycl::queue, true>();
  check_binary_trait<khr::is_property_for, prop::enable_profiling, sycl::queue,
                     true>();
  check_binary_trait<khr::is_property_for, prop::initialize_to_identity,
                     tag::reduction, true>();

  check_binary_trait<khr::is_property_for, prop::in_order, sycl::context,
                     false>();
  check_binary_trait<khr::is_property_for, prop::in_order, sycl::stream,
                     false>();
  check_binary_trait<khr::is_property_for, prop::in_order, tag::reduction,
                     false>();
  check_binary_trait<khr::is_property_for, prop::in_order, tag::usm_device,
                     false>();
  check_binary_trait<khr::is_property_for, prop::in_order, tag::usm_host,
                     false>();
  check_binary_trait<khr::is_property_for, prop::in_order, tag::usm_shared,
                     false>();
  check_binary_trait<khr::is_property_for, prop::in_order, tag::compile,
                     false>();
  check_binary_trait<khr::is_property_for, prop::in_order, tag::link, false>();
  check_binary_trait<khr::is_property_for, prop::in_order, tag::build, false>();
  check_binary_trait<khr::is_property_for, prop::enable_profiling,
                     tag::reduction, false>();
  check_binary_trait<khr::is_property_for, prop::initialize_to_identity,
                     sycl::queue, false>();
  check_binary_trait<khr::is_property_for, prop::initialize_to_identity,
                     sycl::context, false>();
  // Keys and lists are not properties.
  check_binary_trait<khr::is_property_for, key::in_order, sycl::queue, false>();
  check_binary_trait<khr::is_property_for, queue_props_t, sycl::queue, false>();
  check_binary_trait<khr::is_property_for, khr::empty_properties_t, sycl::queue,
                     false>();
  check_binary_trait<khr::is_property_for, int, sycl::queue, false>();
  check_binary_trait<khr::is_property_for, sycl::property::queue::in_order,
                     sycl::queue, false>();
}

TEST_CASE(
    "is_property_key_for classifies the classes a property key applies to",
    "[khr_properties][traits]") {
  check_binary_trait<khr::is_property_key_for, key::in_order, sycl::queue,
                     true>();
  check_binary_trait<khr::is_property_key_for, key::enable_profiling,
                     sycl::queue, true>();
  check_binary_trait<khr::is_property_key_for, key::initialize_to_identity,
                     tag::reduction, true>();

  check_binary_trait<khr::is_property_key_for, key::in_order, sycl::context,
                     false>();
  check_binary_trait<khr::is_property_key_for, key::in_order, tag::reduction,
                     false>();
  check_binary_trait<khr::is_property_key_for, key::enable_profiling,
                     tag::usm_device, false>();
  check_binary_trait<khr::is_property_key_for, key::initialize_to_identity,
                     sycl::queue, false>();
  // Properties are not keys.
  check_binary_trait<khr::is_property_key_for, prop::in_order, sycl::queue,
                     false>();
  check_binary_trait<khr::is_property_key_for, int, sycl::queue, false>();
}

TEST_CASE(
    "is_property_list_for classifies the classes a properties list applies to",
    "[khr_properties][traits]") {
  using empty_t = khr::empty_properties_t;
  SECTION("the empty list applies to every class") {
    check_binary_trait<khr::is_property_list_for, empty_t, sycl::context,
                       true>();
    check_binary_trait<khr::is_property_list_for, empty_t, sycl::queue, true>();
    check_binary_trait<khr::is_property_list_for, empty_t, sycl::stream,
                       true>();
    check_binary_trait<khr::is_property_list_for, empty_t, tag::usm_device,
                       true>();
    check_binary_trait<khr::is_property_list_for, empty_t, tag::usm_host,
                       true>();
    check_binary_trait<khr::is_property_list_for, empty_t, tag::usm_shared,
                       true>();
    check_binary_trait<khr::is_property_list_for, empty_t, tag::reduction,
                       true>();
    check_binary_trait<khr::is_property_list_for, empty_t, tag::compile,
                       true>();
    check_binary_trait<khr::is_property_list_for, empty_t, tag::link, true>();
    check_binary_trait<khr::is_property_list_for, empty_t, tag::build, true>();
  }

  SECTION("a non-empty list applies only if all its properties apply") {
    check_binary_trait<khr::is_property_list_for, queue_props_t, sycl::queue,
                       true>();
    check_binary_trait<khr::is_property_list_for, reduction_props_t,
                       tag::reduction, true>();
    using in_order_only_t = decltype(khr::properties{prop::in_order{}});
    check_binary_trait<khr::is_property_list_for, in_order_only_t, sycl::queue,
                       true>();

    check_binary_trait<khr::is_property_list_for, queue_props_t, sycl::context,
                       false>();
    check_binary_trait<khr::is_property_list_for, queue_props_t, tag::reduction,
                       false>();
    check_binary_trait<khr::is_property_list_for, queue_props_t,
                       tag::usm_device, false>();
    check_binary_trait<khr::is_property_list_for, reduction_props_t,
                       sycl::queue, false>();
    check_binary_trait<khr::is_property_list_for, mixed_props_t, sycl::queue,
                       false>();
    check_binary_trait<khr::is_property_list_for, mixed_props_t, tag::reduction,
                       false>();
  }

  SECTION("types that are not properties lists") {
    check_binary_trait<khr::is_property_list_for, prop::in_order, sycl::queue,
                       false>();
    check_binary_trait<khr::is_property_list_for, int, sycl::queue, false>();
    check_binary_trait<khr::is_property_list_for, sycl::property_list,
                       sycl::queue, false>();
  }
}

template <typename Property, typename Key>
void check_boolean_property() {
  STATIC_CHECK(std::is_trivially_copyable_v<Property>);
  STATIC_CHECK(std::is_trivially_copyable_v<Key>);

  STATIC_CHECK(std::is_same_v<decltype(Property::value), bool>);
  STATIC_CHECK(std::is_default_constructible_v<Property>);
  STATIC_CHECK(std::is_constructible_v<Property, bool>);
  STATIC_CHECK(std::is_convertible_v<bool, Property>);
  STATIC_CHECK(std::is_default_constructible_v<Key>);
  STATIC_CHECK(std::is_empty_v<Key>);

  // The constructor is constexpr and the default value is true.
  constexpr Property default_prop{};
  constexpr Property true_prop{true};
  constexpr Property false_prop{false};
  STATIC_CHECK(default_prop.value);
  STATIC_CHECK(true_prop.value);
  STATIC_CHECK_FALSE(false_prop.value);

  Property runtime_prop{false};
  CHECK_FALSE(runtime_prop.value);
  runtime_prop.value = true;
  CHECK(runtime_prop.value);
}

TEST_CASE("the enable_profiling property and its key satisfy the requirements",
          "[khr_properties][traits][queue]") {
  check_boolean_property<prop::enable_profiling, key::enable_profiling>();
}

TEST_CASE("the in_order property and its key satisfy the requirements",
          "[khr_properties][traits][queue]") {
  check_boolean_property<prop::in_order, key::in_order>();
}

TEST_CASE(
    "the initialize_to_identity property and its key satisfy the requirements",
    "[khr_properties][traits][reduction]") {
  check_boolean_property<prop::initialize_to_identity,
                         key::initialize_to_identity>();
}

}  // namespace khr_properties::tests
