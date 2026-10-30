/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Tests for the sycl::khr::properties class, sycl::khr::empty_properties_t
//  and the SYCL_KHR_PROPERTIES feature test macro.
//
*******************************************************************************/

#include "khr_properties_common.h"

namespace khr_properties::tests {

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

TEST_CASE("the properties extension defines the SYCL_KHR_PROPERTIES macro",
          "[khr_properties]") {
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
// DPC++ does not constrain the constructor.
DISABLED_FOR_TEST_CASE(DPCPP)
("the properties constructor requires each argument to be a property",
 "[khr_properties][properties]")({
  STATIC_CHECK_FALSE(std::is_constructible_v<queue_props_t, int>);
  STATIC_CHECK_FALSE(std::is_constructible_v<khr::empty_properties_t, int>);
  STATIC_CHECK_FALSE(std::is_constructible_v<queue_props_t, not_a_property>);
  STATIC_CHECK_FALSE(
      std::is_constructible_v<queue_props_t, sycl::property::queue::in_order>);
});

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

// DPC++ does not constrain has_property on khr::is_property_key_v.
DISABLED_FOR_TEST_CASE(DPCPP)
("properties::has_property requires a property key",
 "[khr_properties][properties]")({
  STATIC_CHECK_FALSE(can_call_has_property_v<int>);
  STATIC_CHECK_FALSE(can_call_has_property_v<not_a_property>);
  STATIC_CHECK_FALSE(can_call_has_property_v<prop::in_order>);
});

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

// DPC++ rejects the following with a hard error instead of removing
// get_property from overload resolution.
DISABLED_FOR_TEST_CASE(DPCPP)
("properties::get_property is not available for a key that is not in the "
 "list",
 "[khr_properties][properties]")({
  STATIC_CHECK_FALSE(can_call_get_property_v<key::initialize_to_identity>);
  STATIC_CHECK_FALSE(
      can_call_static_get_property_v<key::initialize_to_identity>);
});

}  // namespace khr_properties::tests
