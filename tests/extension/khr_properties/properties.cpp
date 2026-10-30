/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Tests for the APIs that <sycl/khr/properties.hpp> provides: the
//  SYCL_KHR_PROPERTIES feature test macro, the sycl::khr::properties class,
//  sycl::khr::empty_properties_t and the property traits. Also tests the
//  requirements on the properties and keys defined by the extension and
//  their interaction with the old-style property traits.
//
*******************************************************************************/

#include "khr_properties_common.h"

#if SYCL_CTS_KHR_PROPERTIES_SPLIT_HEADERS
#include <sycl/khr/split_headers/property_list.hpp>
#include <sycl/khr/split_headers/reduction.hpp>
#endif

#include <catch2/generators/catch_generators.hpp>

namespace khr_properties::tests {

using reduction_props_t =
    decltype(khr::properties{prop::initialize_to_identity{}});

// A properties list with a queue property and a reduction property.
using mixed_props_t =
    decltype(khr::properties{prop::in_order{}, prop::initialize_to_identity{}});

namespace tag = sycl::khr::property::tag;

template <typename T>
struct is_khr_properties : std::false_type {};
template <typename... Ts>
struct is_khr_properties<khr::properties<Ts...>> : std::true_type {};

// Detects whether CTAD of khr::properties from a single property of type P
// is well-formed.
KHR_PROPERTIES_DETECTOR(can_deduce_properties_from,
                        khr::properties{std::declval<P>()})

// Detects whether has_property<Key>() can be called. Here P is the key type,
// and the properties list is queue_props_t.
KHR_PROPERTIES_DETECTOR(can_call_has_property,
                        queue_props_t::template has_property<P>())

// Detects whether the static get_property<Key>() (overload (1)) can be called
// without an object. Here P is the key type.
KHR_PROPERTIES_DETECTOR(can_call_static_get_property,
                        queue_props_t::template get_property<P>())

// Detects whether the non-static get_property<Key>() can be called on an
// object. Here P is the key type.
KHR_PROPERTIES_DETECTOR(can_call_get_property, std::declval<queue_props_t&>()
                                                   .template get_property<P>())

// Evaluates the constructor and get_property() in a constant expression.
constexpr bool get_in_order_value_in_constexpr(bool v) {
  khr::properties props{prop::in_order{v}};
  return props.get_property<key::in_order>().value;
}

template <template <typename> class Trait, typename T, bool Expected>
void check_unary_trait() {
  using base_t = std::integral_constant<bool, Expected>;
  STATIC_CHECK(std::is_base_of_v<base_t, Trait<T>>);
  STATIC_CHECK(Trait<T>::value == Expected);
}

TEST_CASE("the properties extension defines the SYCL_KHR_PROPERTIES macro",
          "[khr_properties][properties]") {
#ifndef SYCL_KHR_PROPERTIES
  static_assert(false, "SYCL_KHR_PROPERTIES is not defined");
#else
  STATIC_CHECK(SYCL_KHR_PROPERTIES == 1);
#endif
}

TEST_CASE("empty_properties_t is the type of an empty properties list",
          "[khr_properties][properties]") {
  STATIC_CHECK(
      std::is_same_v<khr::empty_properties_t, decltype(khr::properties{})>);
  STATIC_CHECK(is_khr_properties<khr::empty_properties_t>::value);

  khr::empty_properties_t empty{};
  STATIC_CHECK_FALSE(khr::empty_properties_t::has_property<key::in_order>());
  STATIC_CHECK_FALSE(
      khr::empty_properties_t::has_property<key::enable_profiling>());
  STATIC_CHECK_FALSE(
      khr::empty_properties_t::has_property<key::initialize_to_identity>());
  CHECK_FALSE(empty.has_property<key::in_order>());
}

TEST_CASE("properties can be constructed with CTAD from properties",
          "[khr_properties][properties]") {
  SECTION("single property") {
    khr::properties props{prop::in_order{}};
    STATIC_CHECK(is_khr_properties<decltype(props)>::value);
    STATIC_CHECK(decltype(props)::has_property<key::in_order>());
    STATIC_CHECK_FALSE(decltype(props)::has_property<key::enable_profiling>());
    CHECK(props.get_property<key::in_order>().value);
  }

  SECTION("several properties") {
    khr::properties props{prop::in_order{false}, prop::enable_profiling{true},
                          prop::initialize_to_identity{false}};
    STATIC_CHECK(is_khr_properties<decltype(props)>::value);
    STATIC_CHECK(decltype(props)::has_property<key::in_order>());
    STATIC_CHECK(decltype(props)::has_property<key::enable_profiling>());
    STATIC_CHECK(decltype(props)::has_property<key::initialize_to_identity>());
    CHECK_FALSE(props.get_property<key::in_order>().value);
    CHECK(props.get_property<key::enable_profiling>().value);
    CHECK_FALSE(props.get_property<key::initialize_to_identity>().value);
  }

  SECTION("properties with runtime values") {
    bool runtime_value = GENERATE(true, false);
    khr::properties props{prop::in_order{runtime_value},
                          prop::enable_profiling{!runtime_value}};
    CHECK(props.get_property<key::in_order>().value == runtime_value);
    CHECK(props.get_property<key::enable_profiling>().value == !runtime_value);
  }
}

TEST_CASE("the properties constructor is constexpr",
          "[khr_properties][properties]") {
  constexpr khr::properties props{prop::in_order{}, prop::enable_profiling{}};
  STATIC_CHECK(decltype(props)::has_property<key::in_order>());
  STATIC_CHECK(decltype(props)::has_property<key::enable_profiling>());
  // get_property (2) is const, so it can be called on a constexpr object.
  STATIC_CHECK(props.get_property<key::in_order>().value);
  STATIC_CHECK(props.get_property<key::enable_profiling>().value);
  STATIC_CHECK(get_in_order_value_in_constexpr(true));
  STATIC_CHECK_FALSE(get_in_order_value_in_constexpr(false));
}

TEST_CASE("properties can be deduced from and constructed with properties",
          "[khr_properties][properties]") {
  STATIC_CHECK(can_deduce_properties_from_v<prop::in_order>);
  STATIC_CHECK(can_deduce_properties_from_v<prop::enable_profiling>);
  STATIC_CHECK(can_deduce_properties_from_v<prop::initialize_to_identity>);
  STATIC_CHECK(std::is_constructible_v<queue_props_t, prop::in_order,
                                       prop::enable_profiling>);
}

// The deduction guide is unspecified, so deducing a properties type from a
// non-property is not checked. The constructor constraint is checked through
// std::is_constructible on deduced properties types instead.
TEST_CASE("the properties constructor requires each argument to be a property",
          "[khr_properties][properties]") {
  STATIC_CHECK_FALSE(std::is_constructible_v<queue_props_t, int>);
  STATIC_CHECK_FALSE(std::is_constructible_v<khr::empty_properties_t, int>);
  STATIC_CHECK_FALSE(std::is_constructible_v<queue_props_t, not_a_property>);
  STATIC_CHECK_FALSE(
      std::is_constructible_v<queue_props_t, sycl::property::queue::in_order>);
}

TEST_CASE("properties::has_property is a static constexpr member function",
          "[khr_properties][properties]") {
  queue_props_t props{prop::in_order{}, prop::enable_profiling{}};
  STATIC_CHECK(
      std::is_same_v<decltype(queue_props_t::has_property<key::in_order>()),
                     bool>);
  STATIC_CHECK(queue_props_t::has_property<key::in_order>());
  STATIC_CHECK(queue_props_t::has_property<key::enable_profiling>());
  STATIC_CHECK_FALSE(
      queue_props_t::has_property<key::initialize_to_identity>());

  // has_property can be used in "if constexpr" through an object.
  if constexpr (props.has_property<key::in_order>()) {
    SUCCEED();
  } else {
    FAIL("has_property<key::in_order>() returned false");
  }
}

TEST_CASE("properties::has_property accepts property keys",
          "[khr_properties][properties]") {
  STATIC_CHECK(can_call_has_property_v<key::in_order>);
  STATIC_CHECK(can_call_has_property_v<key::initialize_to_identity>);
}

TEST_CASE("properties::has_property requires a property key",
          "[khr_properties][properties]") {
  STATIC_CHECK_FALSE(can_call_has_property_v<int>);
  STATIC_CHECK_FALSE(can_call_has_property_v<not_a_property>);
  STATIC_CHECK_FALSE(can_call_has_property_v<prop::in_order>);
}

TEST_CASE("properties::get_property returns a copy of the property",
          "[khr_properties][properties]") {
  queue_props_t props{prop::in_order{false}, prop::enable_profiling{true}};
  STATIC_CHECK(std::is_same_v<decltype(props.get_property<key::in_order>()),
                              prop::in_order>);
  STATIC_CHECK(
      std::is_same_v<decltype(props.get_property<key::enable_profiling>()),
                     prop::enable_profiling>);

  auto in_order = props.get_property<key::in_order>();
  CHECK_FALSE(in_order.value);
  // Modifying the returned copy does not modify the list.
  in_order.value = true;
  CHECK_FALSE(props.get_property<key::in_order>().value);

  // get_property can be called through a const object.
  const queue_props_t& const_props = props;
  CHECK_FALSE(const_props.get_property<key::in_order>().value);
  CHECK(const_props.get_property<key::enable_profiling>().value);
}

TEST_CASE("properties::get_property is constrained on the property key",
          "[khr_properties][properties]") {
  // Overload (2) is available for keys of runtime properties contained in
  // the list.
  STATIC_CHECK(can_call_get_property_v<key::in_order>);
  STATIC_CHECK(can_call_get_property_v<key::enable_profiling>);
  // The static overload (1) is not available for keys whose property contains
  // runtime values.
  STATIC_CHECK_FALSE(can_call_static_get_property_v<key::in_order>);
  STATIC_CHECK_FALSE(can_call_static_get_property_v<key::enable_profiling>);
}

TEST_CASE(
    "properties::get_property is not available for a key that is not in the "
    "list",
    "[khr_properties][properties]") {
  STATIC_CHECK_FALSE(can_call_get_property_v<key::initialize_to_identity>);
  STATIC_CHECK_FALSE(
      can_call_static_get_property_v<key::initialize_to_identity>);
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

// The traits for the context, stream, USM and kernel bundle classes and tags
// are checked in the test files for those APIs.
TEST_CASE("is_property_for classifies the classes a property applies to",
          "[khr_properties][traits]") {
  check_binary_trait<khr::is_property_for, prop::in_order, sycl::queue, true>();
  check_binary_trait<khr::is_property_for, prop::enable_profiling, sycl::queue,
                     true>();
  check_binary_trait<khr::is_property_for, prop::initialize_to_identity,
                     tag::reduction, true>();

  check_binary_trait<khr::is_property_for, prop::in_order, tag::reduction,
                     false>();
  check_binary_trait<khr::is_property_for, prop::enable_profiling,
                     tag::reduction, false>();
  check_binary_trait<khr::is_property_for, prop::initialize_to_identity,
                     sycl::queue, false>();
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

  check_binary_trait<khr::is_property_key_for, key::in_order, tag::reduction,
                     false>();
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
  SECTION("the empty list applies to the queue and reduction") {
    check_binary_trait<khr::is_property_list_for, empty_t, sycl::queue, true>();
    check_binary_trait<khr::is_property_list_for, empty_t, tag::reduction,
                       true>();
  }

  SECTION("a non-empty list applies only if all its properties apply") {
    check_binary_trait<khr::is_property_list_for, queue_props_t, sycl::queue,
                       true>();
    check_binary_trait<khr::is_property_list_for, reduction_props_t,
                       tag::reduction, true>();
    using in_order_only_t = decltype(khr::properties{prop::in_order{}});
    check_binary_trait<khr::is_property_list_for, in_order_only_t, sycl::queue,
                       true>();

    check_binary_trait<khr::is_property_list_for, queue_props_t, tag::reduction,
                       false>();
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

TEST_CASE("the old-style property traits do not classify new-style properties",
          "[khr_properties][traits]") {
  check_unary_trait<sycl::is_property, prop::in_order, false>();
  check_unary_trait<sycl::is_property, prop::enable_profiling, false>();
  check_unary_trait<sycl::is_property, prop::initialize_to_identity, false>();
  check_unary_trait<sycl::is_property, key::in_order, false>();
  check_unary_trait<sycl::is_property, khr::empty_properties_t, false>();
  check_unary_trait<sycl::is_property, queue_props_t, false>();
  STATIC_CHECK_FALSE(sycl::is_property_v<prop::in_order>);
  STATIC_CHECK_FALSE(sycl::is_property_v<prop::enable_profiling>);
  STATIC_CHECK_FALSE(sycl::is_property_v<prop::initialize_to_identity>);

  check_binary_trait<sycl::is_property_of, prop::in_order, sycl::queue,
                     false>();
  check_binary_trait<sycl::is_property_of, prop::enable_profiling, sycl::queue,
                     false>();
  check_binary_trait<sycl::is_property_of, prop::initialize_to_identity,
                     sycl::queue, false>();
  STATIC_CHECK_FALSE(sycl::is_property_of_v<prop::in_order, sycl::queue>);
  STATIC_CHECK_FALSE(
      sycl::is_property_of_v<prop::enable_profiling, sycl::queue>);
}

template <typename Property, typename Key>
void check_boolean_property() {
  STATIC_CHECK(std::is_trivially_copyable_v<Property>);
  STATIC_CHECK(std::is_trivially_copyable_v<Key>);

  STATIC_CHECK(std::is_same_v<decltype(Property::value), bool>);
  STATIC_CHECK(std::is_default_constructible_v<Property>);
  STATIC_CHECK(std::is_constructible_v<Property, bool>);
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

TEST_CASE("the constructors of the boolean properties from bool are explicit",
          "[khr_properties][traits]") {
  STATIC_CHECK_FALSE(std::is_convertible_v<bool, prop::enable_profiling>);
  STATIC_CHECK_FALSE(std::is_convertible_v<bool, prop::in_order>);
  STATIC_CHECK_FALSE(std::is_convertible_v<bool, prop::initialize_to_identity>);
}

}  // namespace khr_properties::tests
