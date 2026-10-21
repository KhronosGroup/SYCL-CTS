/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Tests for the reduction additions of the sycl_khr_properties extension:
//  the new reduction overloads and the khr::property::initialize_to_identity
//  property.
//
*******************************************************************************/

#include "khr_properties_common.h"

#include <array>

namespace khr_properties::tests {

constexpr std::size_t num_work_items = 64;
constexpr int initial_value = 100;

using plus_t = sycl::plus<int>;
using span2_t = sycl::span<int, 2>;
using dyn_span_t = sycl::span<int, sycl::dynamic_extent>;

/** A combiner for which no identity value is known. */
struct custom_plus {
  int operator()(const int& a, const int& b) const { return a + b; }
};

// Detection traits for each overload. P is the type of the PropertyOrList
// argument.
KHR_PROPERTIES_DETECTOR(can_reduce_ptr,
                        sycl::reduction(std::declval<int*>(), plus_t{},
                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_span,
                        sycl::reduction(std::declval<span2_t>(), plus_t{},
                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_ptr_identity,
                        sycl::reduction(std::declval<int*>(), int{}, plus_t{},
                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_span_identity,
                        sycl::reduction(std::declval<span2_t>(), int{},
                                        plus_t{}, std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_dyn_span,
                        sycl::reduction(std::declval<dyn_span_t>(), plus_t{},
                                        std::declval<P>()))
KHR_PROPERTIES_DETECTOR(can_reduce_dyn_span_identity,
                        sycl::reduction(std::declval<dyn_span_t>(), int{},
                                        plus_t{}, std::declval<P>()))

template <typename P>
void check_reduction_constraints(bool expected) {
  CHECK(can_reduce_ptr_v<P> == expected);
  CHECK(can_reduce_span_v<P> == expected);
  CHECK(can_reduce_ptr_identity_v<P> == expected);
  CHECK(can_reduce_span_identity_v<P> == expected);
}

TEST_CASE("the new reduction overloads are constrained on PropertyOrList",
          "[khr_properties][reduction]") {
  SECTION("reduction properties and lists are accepted") {
    check_reduction_constraints<prop::initialize_to_identity>(true);
    check_reduction_constraints<reduction_props_t>(true);
    check_reduction_constraints<khr::empty_properties_t>(true);
  }
  SECTION("queue properties are rejected") {
    check_reduction_constraints<prop::in_order>(false);
    check_reduction_constraints<prop::enable_profiling>(false);
    check_reduction_constraints<queue_props_t>(false);
  }
  SECTION("lists with a property of another class are rejected") {
    using mixed_t = decltype(khr::properties{prop::in_order{},
                                             prop::initialize_to_identity{}});
    check_reduction_constraints<mixed_t>(false);
  }
  SECTION("types that are neither properties nor lists are rejected") {
    check_reduction_constraints<not_a_property>(false);
    check_reduction_constraints<key::initialize_to_identity>(false);
  }
}

TEST_CASE(
    "the new span reduction overloads are not available for "
    "sycl::dynamic_extent",
    "[khr_properties][reduction]") {
  STATIC_CHECK_FALSE(can_reduce_dyn_span_v<prop::initialize_to_identity>);
  STATIC_CHECK_FALSE(can_reduce_dyn_span_v<reduction_props_t>);
  STATIC_CHECK_FALSE(can_reduce_dyn_span_v<khr::empty_properties_t>);
  STATIC_CHECK_FALSE(
      can_reduce_dyn_span_identity_v<prop::initialize_to_identity>);
  STATIC_CHECK_FALSE(can_reduce_dyn_span_identity_v<reduction_props_t>);
  STATIC_CHECK_FALSE(can_reduce_dyn_span_identity_v<khr::empty_properties_t>);
}

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

}  // namespace khr_properties::tests
