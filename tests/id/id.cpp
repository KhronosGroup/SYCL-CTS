/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022-2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include <type_traits>
#include <utility>

#include <catch2/catch_template_test_macros.hpp>

#include "../common/common.h"
#include "../common/disabled_for_test_case.h"

using namespace sycl_cts;

template <typename T, std::size_t Line, int Dimensions = 0>
class kernel_name;

/** Variadic parameter is the kernel name. */
#define DEVICE_EVAL_T(T, expr, ...)                                 \
  ([=] {                                                            \
    sycl::buffer<std::decay_t<T>, 1> result_buf{1};                 \
    sycl_cts::util::get_cts_object::queue()                         \
        .submit([=, &result_buf](sycl::handler& cgh) {              \
          sycl::accessor result{result_buf, cgh, sycl::write_only}; \
          cgh.single_task<__VA_ARGS__>([=] { result[0] = expr; });  \
        })                                                          \
        .wait_and_throw();                                          \
    sycl::host_accessor acc{result_buf, sycl::read_only};           \
    return acc[0];                                                  \
  })()

/**
 Evaluates a given expression on the SYCL device and returns the result.
 A unique kernel name must be passed in via variadic arguments.

 Limitations:
  - Operands must exist in surrounding scope ([=] capture).
  - No lambda expressions (requires C++20). Use DEVICE_EVAL_T instead. */
#define DEVICE_EVAL(expr, ...) DEVICE_EVAL_T(decltype(expr), expr, __VA_ARGS__)

/** Define a unique kernel name for \p DEVICE_EVAL. */
#define EVAL(expr) DEVICE_EVAL(expr, kernel_name<decltype(expr), __LINE__>)

/** Define a unique kernel name for \p DEVICE_EVAL_T. */
#define EVAL_T(T, expr) DEVICE_EVAL_T(T, expr, kernel_name<T, __LINE__>)

/** Same as \p EVAL but takes parameter \p D into account for kernel name. */
#define EVAL_D(expr) DEVICE_EVAL(expr, kernel_name<decltype(expr), __LINE__, D>)

/** Same as \p EVAL_T but takes parameter \p D into account for kernel name. */
#define EVAL_T_D(T, expr) DEVICE_EVAL_T(T, expr, kernel_name<T, __LINE__, D>)

/**
 Expand a \p EVAL call outside of \p CHECK to prevent the kernel from
 being defined multiple times. */
#define KCHECK(...)            \
  {                            \
    bool result = __VA_ARGS__; \
    CHECK(result);             \
  }

/**
 Expand a \p EVAL call outside of \p CHECK_FALSE to prevent the kernel from
 being defined multiple times. */
#define KCHECK_FALSE(...)      \
  {                            \
    bool result = __VA_ARGS__; \
    CHECK_FALSE(result);       \
  }

TEST_CASE("id provides a default constructor", "[id]") {
  using sycl::id;
  STATIC_CHECK(std::is_default_constructible_v<id<1>>);
  STATIC_CHECK(std::is_default_constructible_v<id<2>>);
  STATIC_CHECK(std::is_default_constructible_v<id<3>>);

  CHECK(id<1>{} == id<1>{0});
  CHECK(id<2>{} == id<2>{0, 0});
  CHECK(id<3>{} == id<3>{0, 0, 0});

  KCHECK(EVAL(id<1>{}) == id<1>{0});
  KCHECK(EVAL(id<2>{}) == id<2>{0, 0});
  KCHECK(EVAL(id<3>{}) == id<3>{0, 0, 0});
}

TEST_CASE("id provides specialized constructors for each dimensionality",
          "[id]") {
  using sycl::id;

  STATIC_CHECK(std::is_constructible_v<id<1>, size_t>);
  STATIC_CHECK(std::is_constructible_v<id<2>, size_t, size_t>);
  STATIC_CHECK(std::is_constructible_v<id<3>, size_t, size_t, size_t>);

  const id<1> a{5};
  CHECK(a[0] == 5);

  const id<2> b{5, 8};
  CHECK(b[0] == 5);
  CHECK(b[1] == 8);

  const id<3> c{5, 8, 3};
  CHECK(c[0] == 5);
  CHECK(c[1] == 8);
  CHECK(c[2] == 3);

  KCHECK(EVAL(id<1>{5}) == id<1>{5});
  KCHECK(EVAL((id<2>{5, 8})) == id<2>{5, 8});
  KCHECK(EVAL((id<3>{5, 8, 3})) == id<3>{5, 8, 3});
}

// id h[elper] type for creating ids in templated contexts
template <int Dimensions>
using idh = util::get_cts_object::id<Dimensions>;

// TODO SPEC: Do common by-value semantics require trivially copyable?
// See also https://github.com/KhronosGroup/SYCL-Docs/issues/210
TEMPLATE_TEST_CASE_SIG("id provides common by-value semantics", "[id]",
                       ((int D), D), 1, 2, 3) {
  using sycl::id;

  SECTION("copy constructor") {
    CHECK(std::is_trivially_copy_constructible_v<id<D>>);
    KCHECK(EVAL_D(std::is_trivially_copy_constructible_v<id<D>>));

    const auto copy = [] {
      const auto a = idh<D>::get(5, 8, 3);
      id<D> b{a};
      return b;
    };
    CHECK(copy() == idh<D>::get(5, 8, 3));
    KCHECK(EVAL_T_D(id<D>, copy()) == idh<D>::get(5, 8, 3));
  }

  SECTION("copy assignment operator") {
    CHECK(std::is_trivially_copy_assignable_v<id<D>>);
    KCHECK(EVAL_D(std::is_trivially_copy_assignable_v<id<D>>));

    const auto copy = [] {
      const auto a = idh<D>::get(5, 8, 3);
      id<D> b;
      b = a;
      return b;
    };
    CHECK(copy() == idh<D>::get(5, 8, 3));
    KCHECK(EVAL_T_D(id<D>, copy()) == idh<D>::get(5, 8, 3));
  }

  SECTION("destructor") {
    CHECK(std::is_trivially_destructible_v<id<D>>);
    KCHECK(EVAL_D(std::is_trivially_destructible_v<id<D>>));
  }

  SECTION("move constructor") {
    CHECK(std::is_trivially_move_constructible_v<id<D>>);
    KCHECK(EVAL_D(std::is_trivially_move_constructible_v<id<D>>));

    const auto move = [] {
      auto a = idh<D>::get(5, 8, 3);
      id<D> b{std::move(a)};
      return b;
    };
    CHECK(move() == idh<D>::get(5, 8, 3));
    KCHECK(EVAL_T_D(id<D>, move()) == idh<D>::get(5, 8, 3));
  }

  SECTION("move assignment operator") {
    CHECK(std::is_trivially_move_assignable_v<id<D>>);
    KCHECK(EVAL_D(std::is_trivially_move_assignable_v<id<D>>));

    const auto move = [] {
      auto a = idh<D>::get(5, 8, 3);
      id<D> b;
      b = std::move(a);
      return b;
    };
    CHECK(move() == idh<D>::get(5, 8, 3));
    KCHECK(EVAL_T_D(id<D>, move()) == idh<D>::get(5, 8, 3));
  }

  SECTION("equality operators") {
    auto a1 = idh<D>::get(5, 8, 3);
    auto a2 = idh<D>::get(5, 8, 3);
    auto b1 = idh<D>::get(4, 8, 2);

    CHECK(a1 == a1);
    CHECK(a1 == a2);
    CHECK(a2 == a1);
    CHECK(b1 == b1);
    CHECK_FALSE(a1 == b1);
    CHECK_FALSE(b1 == a1);
    CHECK_FALSE(a2 == b1);

    KCHECK(EVAL_D(a1 == a1));
    KCHECK(EVAL_D(a1 == a2));
    KCHECK(EVAL_D(a2 == a1));
    KCHECK(EVAL_D(b1 == b1));
    KCHECK_FALSE(EVAL_D(a1 == b1));
    KCHECK_FALSE(EVAL_D(b1 == a1));
    KCHECK_FALSE(EVAL_D(a2 == b1));

    CHECK_FALSE(a1 != a1);
    CHECK_FALSE(a1 != a2);
    CHECK_FALSE(a2 != a1);
    CHECK_FALSE(b1 != b1);
    CHECK(a1 != b1);
    CHECK(b1 != a1);
    CHECK(a2 != b1);

    KCHECK_FALSE(EVAL_D(a1 != a1));
    KCHECK_FALSE(EVAL_D(a1 != a2));
    KCHECK_FALSE(EVAL_D(a2 != a1));
    KCHECK_FALSE(EVAL_D(b1 != b1));
    KCHECK(EVAL_D(a1 != b1));
    KCHECK(EVAL_D(b1 != a1));
    KCHECK(EVAL_D(a2 != b1));
  }
}

TEMPLATE_TEST_CASE_SIG("id can be implicitly conversion-constructed from range",
                       "[id]", ((int D), D), 1, 2, 3) {
  using sycl::id;

  const auto convert = [] {
    const auto r = util::get_cts_object::range<D>::get(5, 8, 3);
    sycl::id<D> a;
    // Use assignment operator to trigger implicit conversion
    a = r;
    return a;
  };

  CHECK(convert() == idh<D>::get(5, 8, 3));
  KCHECK(EVAL_T_D(id<D>, convert()) == idh<D>::get(5, 8, 3));
}

template <int D>
class kernel_id;

TEMPLATE_TEST_CASE_SIG("id can be implicitly conversion-constructed from item",
                       "[id]", ((int D), D), 1, 2, 3) {
  using sycl::id;

  auto q = util::get_cts_object::queue();
  sycl::buffer<id<D>, 1> result_buf{1};
  const auto r = util::get_cts_object::range<D>::get(5, 8, 3);
  q.submit([r, &result_buf](sycl::handler& cgh) {
     sycl::accessor result{result_buf, cgh, sycl::write_only};
     cgh.parallel_for<kernel_id<D>>(r, [=](sycl::item<D> itm) {
       // `ul` suffix is necessary to resolve ambiguity for id<1>
       if (itm.get_id() == id<D>{r} - 1ul) {
         // Use assignment operator to trigger implicit conversion
         result[0] = itm;
       }
     });
   }).wait_and_throw();
  sycl::host_accessor acc{result_buf, sycl::read_only};

  CHECK(acc[0] == idh<D>::get(4, 7, 2));
}

TEMPLATE_TEST_CASE_SIG("id supports get() and operator[]", "[id]", ((int D), D),
                       1, 2, 3) {
  const auto a = idh<D>::get(5, 8, 3);
  const size_t values[] = {5, 8, 3};

  for (int i = 0; i < D; ++i) {
    CHECK(a.get(i) == values[i]);
    CHECK(a[i] == values[i]);
    KCHECK(EVAL_D(a.get(i)) == values[i]);
    KCHECK(EVAL_D(a[i]) == values[i]);
  }

  const auto assign_component = [](auto x, auto c, auto v) {
    x[c] = v;
    return x;
  };

  using sycl::id;

  CHECK(assign_component(a, 0, 7) == idh<D>::get(7, 8, 3));
  KCHECK(EVAL_T_D(id<D>, assign_component(a, 0, 7)) == idh<D>::get(7, 8, 3));

  if (D >= 2) {
    CHECK(assign_component(a, 1, 9) == idh<D>::get(5, 9, 3));
    KCHECK(EVAL_T_D(id<D>, assign_component(a, 1, 9)) == idh<D>::get(5, 9, 3));
  }

  if (D == 3) {
    CHECK(assign_component(a, 2, 11) == idh<D>::get(5, 8, 11));
    KCHECK(EVAL_T_D(id<D>, assign_component(a, 2, 11)) ==
           idh<D>::get(5, 8, 11));
  }
}

DISABLED_FOR_TEST_CASE(AdaptiveCpp)
("id provides static constexpr member 'dimensions'", "[id]")({
  // Dimension arguments in this test case are expanded to avoid conflicting
  // kernel names for different instantiations of sycl::id.
  CHECK(std::is_same_v<decltype(sycl::id<1>::dimensions), const int>);
  CHECK(std::is_same_v<decltype(sycl::id<2>::dimensions), const int>);
  CHECK(std::is_same_v<decltype(sycl::id<3>::dimensions), const int>);
  KCHECK(EVAL((std::is_same_v<decltype(sycl::id<1>::dimensions), const int>)));
  KCHECK(EVAL((std::is_same_v<decltype(sycl::id<2>::dimensions), const int>)));
  KCHECK(EVAL((std::is_same_v<decltype(sycl::id<3>::dimensions), const int>)));

  CHECK(sycl::id<1>::dimensions == 1);
  CHECK(sycl::id<2>::dimensions == 2);
  CHECK(sycl::id<3>::dimensions == 3);
  KCHECK(EVAL(sycl::id<1>::dimensions) == 1);
  KCHECK(EVAL(sycl::id<2>::dimensions) == 2);
  KCHECK(EVAL(sycl::id<3>::dimensions) == 3);
});

TEST_CASE("id can be converted to size_t if Dimensions == 1", "[id]") {
  using sycl::id;
  const auto convert = [] {
    const sycl::id a{42};
    const size_t b = a;
    return b;
  };
  CHECK(convert() == 42);
  KCHECK(EVAL(convert()) == 42);
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various binary operators of the form `id OP id`", "[id]",
    ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(5, 8, 3);
  const auto b = idh<D>::get(4, 8, 2);

  CHECK(a + b == idh<D>::get(9, 16, 5));
  CHECK(a - b == idh<D>::get(1, 0, 1));
  CHECK(a * b == idh<D>::get(20, 64, 6));
  CHECK(a / b == idh<D>::get(1, 1, 1));
  CHECK(a % b == idh<D>::get(1, 0, 1));
  CHECK(a << b == idh<D>::get(80, 2048, 12));
  CHECK(a >> b == idh<D>::get(0, 0, 0));
  CHECK((a & b) == idh<D>::get(4, 8, 2));
  CHECK((a | b) == idh<D>::get(5, 8, 3));
  CHECK((a ^ b) == idh<D>::get(1, 0, 1));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(0, 0, 0));
  CHECK((a > b) == idh<D>::get(1, 0, 1));
  CHECK((a <= b) == idh<D>::get(0, 1, 0));
  CHECK((a >= b) == idh<D>::get(1, 1, 1));

  KCHECK(EVAL_D(a + b) == idh<D>::get(9, 16, 5));
  KCHECK(EVAL_D(a - b) == idh<D>::get(1, 0, 1));
  KCHECK(EVAL_D(a * b) == idh<D>::get(20, 64, 6));
  KCHECK(EVAL_D(a / b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a % b) == idh<D>::get(1, 0, 1));
  KCHECK(EVAL_D(a << b) == idh<D>::get(80, 2048, 12));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a & b) == idh<D>::get(4, 8, 2));
  KCHECK(EVAL_D(a | b) == idh<D>::get(5, 8, 3));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(1, 0, 1));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a > b) == idh<D>::get(1, 0, 1));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(0, 1, 0));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(1, 1, 1));
}

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP size_t` and "
 "`size_t OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(5, 8, 3);
  const size_t b = 3;

  CHECK(a + b == idh<D>::get(8, 11, 6));
  CHECK(b + a == idh<D>::get(8, 11, 6));
  CHECK(a - b == idh<D>::get(2, 5, 0));
  CHECK(b - a == idh<D>::get(-2, -5, 0));
  CHECK(a * b == idh<D>::get(15, 24, 9));
  CHECK(b * a == idh<D>::get(15, 24, 9));
  CHECK(a / b == idh<D>::get(1, 2, 1));
  CHECK(b / a == idh<D>::get(0, 0, 1));
  CHECK(a % b == idh<D>::get(2, 2, 0));
  CHECK(b % a == idh<D>::get(3, 3, 0));
  CHECK(a << b == idh<D>::get(40, 64, 24));
  CHECK(b << a == idh<D>::get(96, 768, 24));
  CHECK(a >> b == idh<D>::get(0, 1, 0));
  CHECK(b >> a == idh<D>::get(0, 0, 0));
  CHECK((a & b) == idh<D>::get(1, 0, 3));
  CHECK((b & a) == idh<D>::get(1, 0, 3));
  CHECK((a | b) == idh<D>::get(7, 11, 3));
  CHECK((b | a) == idh<D>::get(7, 11, 3));
  CHECK((a ^ b) == idh<D>::get(6, 11, 0));
  CHECK((b ^ a) == idh<D>::get(6, 11, 0));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(0, 0, 0));
  CHECK((b < a) == idh<D>::get(1, 1, 0));
  CHECK((a > b) == idh<D>::get(1, 1, 0));
  CHECK((b > a) == idh<D>::get(0, 0, 0));
  CHECK((a <= b) == idh<D>::get(0, 0, 1));
  CHECK((b <= a) == idh<D>::get(1, 1, 1));
  CHECK((a >= b) == idh<D>::get(1, 1, 1));
  CHECK((b >= a) == idh<D>::get(0, 0, 1));

  KCHECK(EVAL_D(a + b) == idh<D>::get(8, 11, 6));
  KCHECK(EVAL_D(b + a) == idh<D>::get(8, 11, 6));
  KCHECK(EVAL_D(a - b) == idh<D>::get(2, 5, 0));
  KCHECK(EVAL_D(b - a) == idh<D>::get(-2, -5, 0));
  KCHECK(EVAL_D(a * b) == idh<D>::get(15, 24, 9));
  KCHECK(EVAL_D(b * a) == idh<D>::get(15, 24, 9));
  KCHECK(EVAL_D(a / b) == idh<D>::get(1, 2, 1));
  KCHECK(EVAL_D(b / a) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(a % b) == idh<D>::get(2, 2, 0));
  KCHECK(EVAL_D(b % a) == idh<D>::get(3, 3, 0));
  KCHECK(EVAL_D(a << b) == idh<D>::get(40, 64, 24));
  KCHECK(EVAL_D(b << a) == idh<D>::get(96, 768, 24));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(0, 1, 0));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a & b) == idh<D>::get(1, 0, 3));
  KCHECK(EVAL_D(b & a) == idh<D>::get(1, 0, 3));
  KCHECK(EVAL_D(a | b) == idh<D>::get(7, 11, 3));
  KCHECK(EVAL_D(b | a) == idh<D>::get(7, 11, 3));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(6, 11, 0));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(6, 11, 0));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b < a) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(a > b) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(b > a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(0, 0, 1));
});

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP bool` and "
 "`bool OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(4, 9, 2);
  const bool b = 1;

  CHECK(a + b == idh<D>::get(5, 10, 3));
  CHECK(b + a == idh<D>::get(5, 10, 3));
  CHECK(a - b == idh<D>::get(3, 8, 1));
  CHECK(b - a == idh<D>::get(-3, -8, -1));
  CHECK(a * b == idh<D>::get(4, 9, 2));
  CHECK(b * a == idh<D>::get(4, 9, 2));
  CHECK(a / b == idh<D>::get(4, 9, 2));
  CHECK(b / a == idh<D>::get(0, 0, 0));
  CHECK(a % b == idh<D>::get(0, 0, 0));
  CHECK(b % a == idh<D>::get(1, 1, 1));
  CHECK(a << b == idh<D>::get(8, 18, 4));
  CHECK(b << a == idh<D>::get(16, 512, 4));
  CHECK(a >> b == idh<D>::get(2, 4, 1));
  CHECK(b >> a == idh<D>::get(0, 0, 0));
  CHECK((a & b) == idh<D>::get(0, 1, 0));
  CHECK((b & a) == idh<D>::get(0, 1, 0));
  CHECK((a | b) == idh<D>::get(5, 9, 3));
  CHECK((b | a) == idh<D>::get(5, 9, 3));
  CHECK((a ^ b) == idh<D>::get(5, 8, 3));
  CHECK((b ^ a) == idh<D>::get(5, 8, 3));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(0, 0, 0));
  CHECK((b < a) == idh<D>::get(1, 1, 1));
  CHECK((a > b) == idh<D>::get(1, 1, 1));
  CHECK((b > a) == idh<D>::get(0, 0, 0));
  CHECK((a <= b) == idh<D>::get(0, 0, 0));
  CHECK((b <= a) == idh<D>::get(1, 1, 1));
  CHECK((a >= b) == idh<D>::get(1, 1, 1));
  CHECK((b >= a) == idh<D>::get(0, 0, 0));

  KCHECK(EVAL_D(a + b) == idh<D>::get(5, 10, 3));
  KCHECK(EVAL_D(b + a) == idh<D>::get(5, 10, 3));
  KCHECK(EVAL_D(a - b) == idh<D>::get(3, 8, 1));
  KCHECK(EVAL_D(b - a) == idh<D>::get(-3, -8, -1));
  KCHECK(EVAL_D(a * b) == idh<D>::get(4, 9, 2));
  KCHECK(EVAL_D(b * a) == idh<D>::get(4, 9, 2));
  KCHECK(EVAL_D(a / b) == idh<D>::get(4, 9, 2));
  KCHECK(EVAL_D(b / a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a % b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b % a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a << b) == idh<D>::get(8, 18, 4));
  KCHECK(EVAL_D(b << a) == idh<D>::get(16, 512, 4));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(2, 4, 1));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a & b) == idh<D>::get(0, 1, 0));
  KCHECK(EVAL_D(b & a) == idh<D>::get(0, 1, 0));
  KCHECK(EVAL_D(a | b) == idh<D>::get(5, 9, 3));
  KCHECK(EVAL_D(b | a) == idh<D>::get(5, 9, 3));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(5, 8, 3));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(5, 8, 3));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b < a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a > b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b > a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(0, 0, 0));
});

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP char` and "
 "`char OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(8, 8, 6);
  const char b = 2;

  CHECK(a + b == idh<D>::get(10, 10, 8));
  CHECK(b + a == idh<D>::get(10, 10, 8));
  CHECK(a - b == idh<D>::get(6, 6, 4));
  CHECK(b - a == idh<D>::get(-6, -6, -4));
  CHECK(a * b == idh<D>::get(16, 16, 12));
  CHECK(b * a == idh<D>::get(16, 16, 12));
  CHECK(a / b == idh<D>::get(4, 4, 3));
  CHECK(b / a == idh<D>::get(0, 0, 0));
  CHECK(a % b == idh<D>::get(0, 0, 0));
  CHECK(b % a == idh<D>::get(2, 2, 2));
  CHECK(a << b == idh<D>::get(32, 32, 24));
  CHECK(b << a == idh<D>::get(512, 512, 128));
  CHECK(a >> b == idh<D>::get(2, 2, 1));
  CHECK(b >> a == idh<D>::get(0, 0, 0));
  CHECK((a & b) == idh<D>::get(0, 0, 2));
  CHECK((b & a) == idh<D>::get(0, 0, 2));
  CHECK((a | b) == idh<D>::get(10, 10, 6));
  CHECK((b | a) == idh<D>::get(10, 10, 6));
  CHECK((a ^ b) == idh<D>::get(10, 10, 4));
  CHECK((b ^ a) == idh<D>::get(10, 10, 4));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(0, 0, 0));
  CHECK((b < a) == idh<D>::get(1, 1, 1));
  CHECK((a > b) == idh<D>::get(1, 1, 1));
  CHECK((b > a) == idh<D>::get(0, 0, 0));
  CHECK((a <= b) == idh<D>::get(0, 0, 0));
  CHECK((b <= a) == idh<D>::get(1, 1, 1));
  CHECK((a >= b) == idh<D>::get(1, 1, 1));
  CHECK((b >= a) == idh<D>::get(0, 0, 0));

  KCHECK(EVAL_D(a + b) == idh<D>::get(10, 10, 8));
  KCHECK(EVAL_D(b + a) == idh<D>::get(10, 10, 8));
  KCHECK(EVAL_D(a - b) == idh<D>::get(6, 6, 4));
  KCHECK(EVAL_D(b - a) == idh<D>::get(-6, -6, -4));
  KCHECK(EVAL_D(a * b) == idh<D>::get(16, 16, 12));
  KCHECK(EVAL_D(b * a) == idh<D>::get(16, 16, 12));
  KCHECK(EVAL_D(a / b) == idh<D>::get(4, 4, 3));
  KCHECK(EVAL_D(b / a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a % b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b % a) == idh<D>::get(2, 2, 2));
  KCHECK(EVAL_D(a << b) == idh<D>::get(32, 32, 24));
  KCHECK(EVAL_D(b << a) == idh<D>::get(512, 512, 128));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(2, 2, 1));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a & b) == idh<D>::get(0, 0, 2));
  KCHECK(EVAL_D(b & a) == idh<D>::get(0, 0, 2));
  KCHECK(EVAL_D(a | b) == idh<D>::get(10, 10, 6));
  KCHECK(EVAL_D(b | a) == idh<D>::get(10, 10, 6));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(10, 10, 4));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(10, 10, 4));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b < a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a > b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b > a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(0, 0, 0));
});

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP char16_t` and "
 "`char16_t OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(1, 1, 10);
  const char16_t b = 7;

  CHECK(a + b == idh<D>::get(8, 8, 17));
  CHECK(b + a == idh<D>::get(8, 8, 17));
  CHECK(a - b == idh<D>::get(-6, -6, 3));
  CHECK(b - a == idh<D>::get(6, 6, -3));
  CHECK(a * b == idh<D>::get(7, 7, 70));
  CHECK(b * a == idh<D>::get(7, 7, 70));
  CHECK(a / b == idh<D>::get(0, 0, 1));
  CHECK(b / a == idh<D>::get(7, 7, 0));
  CHECK(a % b == idh<D>::get(1, 1, 3));
  CHECK(b % a == idh<D>::get(0, 0, 7));
  CHECK(a << b == idh<D>::get(128, 128, 1280));
  CHECK(b << a == idh<D>::get(14, 14, 7168));
  CHECK(a >> b == idh<D>::get(0, 0, 0));
  CHECK(b >> a == idh<D>::get(3, 3, 0));
  CHECK((a & b) == idh<D>::get(1, 1, 2));
  CHECK((b & a) == idh<D>::get(1, 1, 2));
  CHECK((a | b) == idh<D>::get(7, 7, 15));
  CHECK((b | a) == idh<D>::get(7, 7, 15));
  CHECK((a ^ b) == idh<D>::get(6, 6, 13));
  CHECK((b ^ a) == idh<D>::get(6, 6, 13));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(1, 1, 0));
  CHECK((b < a) == idh<D>::get(0, 0, 1));
  CHECK((a > b) == idh<D>::get(0, 0, 1));
  CHECK((b > a) == idh<D>::get(1, 1, 0));
  CHECK((a <= b) == idh<D>::get(1, 1, 0));
  CHECK((b <= a) == idh<D>::get(0, 0, 1));
  CHECK((a >= b) == idh<D>::get(0, 0, 1));
  CHECK((b >= a) == idh<D>::get(1, 1, 0));

  KCHECK(EVAL_D(a + b) == idh<D>::get(8, 8, 17));
  KCHECK(EVAL_D(b + a) == idh<D>::get(8, 8, 17));
  KCHECK(EVAL_D(a - b) == idh<D>::get(-6, -6, 3));
  KCHECK(EVAL_D(b - a) == idh<D>::get(6, 6, -3));
  KCHECK(EVAL_D(a * b) == idh<D>::get(7, 7, 70));
  KCHECK(EVAL_D(b * a) == idh<D>::get(7, 7, 70));
  KCHECK(EVAL_D(a / b) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(b / a) == idh<D>::get(7, 7, 0));
  KCHECK(EVAL_D(a % b) == idh<D>::get(1, 1, 3));
  KCHECK(EVAL_D(b % a) == idh<D>::get(0, 0, 7));
  KCHECK(EVAL_D(a << b) == idh<D>::get(128, 128, 1280));
  KCHECK(EVAL_D(b << a) == idh<D>::get(14, 14, 7168));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(3, 3, 0));
  KCHECK(EVAL_D(a & b) == idh<D>::get(1, 1, 2));
  KCHECK(EVAL_D(b & a) == idh<D>::get(1, 1, 2));
  KCHECK(EVAL_D(a | b) == idh<D>::get(7, 7, 15));
  KCHECK(EVAL_D(b | a) == idh<D>::get(7, 7, 15));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(6, 6, 13));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(6, 6, 13));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(b < a) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(a > b) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(b > a) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(1, 1, 0));
});

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP char32_t` and "
 "`char32_t OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(4, 4, 4);
  const char32_t b = 7;

  CHECK(a + b == idh<D>::get(11, 11, 11));
  CHECK(b + a == idh<D>::get(11, 11, 11));
  CHECK(a - b == idh<D>::get(-3, -3, -3));
  CHECK(b - a == idh<D>::get(3, 3, 3));
  CHECK(a * b == idh<D>::get(28, 28, 28));
  CHECK(b * a == idh<D>::get(28, 28, 28));
  CHECK(a / b == idh<D>::get(0, 0, 0));
  CHECK(b / a == idh<D>::get(1, 1, 1));
  CHECK(a % b == idh<D>::get(4, 4, 4));
  CHECK(b % a == idh<D>::get(3, 3, 3));
  CHECK(a << b == idh<D>::get(512, 512, 512));
  CHECK(b << a == idh<D>::get(112, 112, 112));
  CHECK(a >> b == idh<D>::get(0, 0, 0));
  CHECK(b >> a == idh<D>::get(0, 0, 0));
  CHECK((a & b) == idh<D>::get(4, 4, 4));
  CHECK((b & a) == idh<D>::get(4, 4, 4));
  CHECK((a | b) == idh<D>::get(7, 7, 7));
  CHECK((b | a) == idh<D>::get(7, 7, 7));
  CHECK((a ^ b) == idh<D>::get(3, 3, 3));
  CHECK((b ^ a) == idh<D>::get(3, 3, 3));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(1, 1, 1));
  CHECK((b < a) == idh<D>::get(0, 0, 0));
  CHECK((a > b) == idh<D>::get(0, 0, 0));
  CHECK((b > a) == idh<D>::get(1, 1, 1));
  CHECK((a <= b) == idh<D>::get(1, 1, 1));
  CHECK((b <= a) == idh<D>::get(0, 0, 0));
  CHECK((a >= b) == idh<D>::get(0, 0, 0));
  CHECK((b >= a) == idh<D>::get(1, 1, 1));

  KCHECK(EVAL_D(a + b) == idh<D>::get(11, 11, 11));
  KCHECK(EVAL_D(b + a) == idh<D>::get(11, 11, 11));
  KCHECK(EVAL_D(a - b) == idh<D>::get(-3, -3, -3));
  KCHECK(EVAL_D(b - a) == idh<D>::get(3, 3, 3));
  KCHECK(EVAL_D(a * b) == idh<D>::get(28, 28, 28));
  KCHECK(EVAL_D(b * a) == idh<D>::get(28, 28, 28));
  KCHECK(EVAL_D(a / b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b / a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a % b) == idh<D>::get(4, 4, 4));
  KCHECK(EVAL_D(b % a) == idh<D>::get(3, 3, 3));
  KCHECK(EVAL_D(a << b) == idh<D>::get(512, 512, 512));
  KCHECK(EVAL_D(b << a) == idh<D>::get(112, 112, 112));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a & b) == idh<D>::get(4, 4, 4));
  KCHECK(EVAL_D(b & a) == idh<D>::get(4, 4, 4));
  KCHECK(EVAL_D(a | b) == idh<D>::get(7, 7, 7));
  KCHECK(EVAL_D(b | a) == idh<D>::get(7, 7, 7));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(3, 3, 3));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(3, 3, 3));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b < a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a > b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b > a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(1, 1, 1));
});

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP wchar_t` and "
 "`wchar_t OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(4, 3, 10);
  const wchar_t b = 6;

  CHECK(a + b == idh<D>::get(10, 9, 16));
  CHECK(b + a == idh<D>::get(10, 9, 16));
  CHECK(a - b == idh<D>::get(-2, -3, 4));
  CHECK(b - a == idh<D>::get(2, 3, -4));
  CHECK(a * b == idh<D>::get(24, 18, 60));
  CHECK(b * a == idh<D>::get(24, 18, 60));
  CHECK(a / b == idh<D>::get(0, 0, 1));
  CHECK(b / a == idh<D>::get(1, 2, 0));
  CHECK(a % b == idh<D>::get(4, 3, 4));
  CHECK(b % a == idh<D>::get(2, 0, 6));
  CHECK(a << b == idh<D>::get(256, 192, 640));
  CHECK(b << a == idh<D>::get(96, 48, 6144));
  CHECK(a >> b == idh<D>::get(0, 0, 0));
  CHECK(b >> a == idh<D>::get(0, 0, 0));
  CHECK((a & b) == idh<D>::get(4, 2, 2));
  CHECK((b & a) == idh<D>::get(4, 2, 2));
  CHECK((a | b) == idh<D>::get(6, 7, 14));
  CHECK((b | a) == idh<D>::get(6, 7, 14));
  CHECK((a ^ b) == idh<D>::get(2, 5, 12));
  CHECK((b ^ a) == idh<D>::get(2, 5, 12));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(1, 1, 0));
  CHECK((b < a) == idh<D>::get(0, 0, 1));
  CHECK((a > b) == idh<D>::get(0, 0, 1));
  CHECK((b > a) == idh<D>::get(1, 1, 0));
  CHECK((a <= b) == idh<D>::get(1, 1, 0));
  CHECK((b <= a) == idh<D>::get(0, 0, 1));
  CHECK((a >= b) == idh<D>::get(0, 0, 1));
  CHECK((b >= a) == idh<D>::get(1, 1, 0));

  KCHECK(EVAL_D(a + b) == idh<D>::get(10, 9, 16));
  KCHECK(EVAL_D(b + a) == idh<D>::get(10, 9, 16));
  KCHECK(EVAL_D(a - b) == idh<D>::get(-2, -3, 4));
  KCHECK(EVAL_D(b - a) == idh<D>::get(2, 3, -4));
  KCHECK(EVAL_D(a * b) == idh<D>::get(24, 18, 60));
  KCHECK(EVAL_D(b * a) == idh<D>::get(24, 18, 60));
  KCHECK(EVAL_D(a / b) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(b / a) == idh<D>::get(1, 2, 0));
  KCHECK(EVAL_D(a % b) == idh<D>::get(4, 3, 4));
  KCHECK(EVAL_D(b % a) == idh<D>::get(2, 0, 6));
  KCHECK(EVAL_D(a << b) == idh<D>::get(256, 192, 640));
  KCHECK(EVAL_D(b << a) == idh<D>::get(96, 48, 6144));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a & b) == idh<D>::get(4, 2, 2));
  KCHECK(EVAL_D(b & a) == idh<D>::get(4, 2, 2));
  KCHECK(EVAL_D(a | b) == idh<D>::get(6, 7, 14));
  KCHECK(EVAL_D(b | a) == idh<D>::get(6, 7, 14));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(2, 5, 12));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(2, 5, 12));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(b < a) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(a > b) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(b > a) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(1, 1, 0));
});

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP short` and "
 "`short OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(3, 6, 9);
  const short b = 9;

  CHECK(a + b == idh<D>::get(12, 15, 18));
  CHECK(b + a == idh<D>::get(12, 15, 18));
  CHECK(a - b == idh<D>::get(-6, -3, 0));
  CHECK(b - a == idh<D>::get(6, 3, 0));
  CHECK(a * b == idh<D>::get(27, 54, 81));
  CHECK(b * a == idh<D>::get(27, 54, 81));
  CHECK(a / b == idh<D>::get(0, 0, 1));
  CHECK(b / a == idh<D>::get(3, 1, 1));
  CHECK(a % b == idh<D>::get(3, 6, 0));
  CHECK(b % a == idh<D>::get(0, 3, 0));
  CHECK(a << b == idh<D>::get(1536, 3072, 4608));
  CHECK(b << a == idh<D>::get(72, 576, 4608));
  CHECK(a >> b == idh<D>::get(0, 0, 0));
  CHECK(b >> a == idh<D>::get(1, 0, 0));
  CHECK((a & b) == idh<D>::get(1, 0, 9));
  CHECK((b & a) == idh<D>::get(1, 0, 9));
  CHECK((a | b) == idh<D>::get(11, 15, 9));
  CHECK((b | a) == idh<D>::get(11, 15, 9));
  CHECK((a ^ b) == idh<D>::get(10, 15, 0));
  CHECK((b ^ a) == idh<D>::get(10, 15, 0));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(1, 1, 0));
  CHECK((b < a) == idh<D>::get(0, 0, 0));
  CHECK((a > b) == idh<D>::get(0, 0, 0));
  CHECK((b > a) == idh<D>::get(1, 1, 0));
  CHECK((a <= b) == idh<D>::get(1, 1, 1));
  CHECK((b <= a) == idh<D>::get(0, 0, 1));
  CHECK((a >= b) == idh<D>::get(0, 0, 1));
  CHECK((b >= a) == idh<D>::get(1, 1, 1));

  KCHECK(EVAL_D(a + b) == idh<D>::get(12, 15, 18));
  KCHECK(EVAL_D(b + a) == idh<D>::get(12, 15, 18));
  KCHECK(EVAL_D(a - b) == idh<D>::get(-6, -3, 0));
  KCHECK(EVAL_D(b - a) == idh<D>::get(6, 3, 0));
  KCHECK(EVAL_D(a * b) == idh<D>::get(27, 54, 81));
  KCHECK(EVAL_D(b * a) == idh<D>::get(27, 54, 81));
  KCHECK(EVAL_D(a / b) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(b / a) == idh<D>::get(3, 1, 1));
  KCHECK(EVAL_D(a % b) == idh<D>::get(3, 6, 0));
  KCHECK(EVAL_D(b % a) == idh<D>::get(0, 3, 0));
  KCHECK(EVAL_D(a << b) == idh<D>::get(1536, 3072, 4608));
  KCHECK(EVAL_D(b << a) == idh<D>::get(72, 576, 4608));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(1, 0, 0));
  KCHECK(EVAL_D(a & b) == idh<D>::get(1, 0, 9));
  KCHECK(EVAL_D(b & a) == idh<D>::get(1, 0, 9));
  KCHECK(EVAL_D(a | b) == idh<D>::get(11, 15, 9));
  KCHECK(EVAL_D(b | a) == idh<D>::get(11, 15, 9));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(10, 15, 0));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(10, 15, 0));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(b < a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a > b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b > a) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(0, 0, 1));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(1, 1, 1));
});

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP int` and "
 "`int OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(7, 7, 2);
  const int b = 10;

  CHECK(a + b == idh<D>::get(17, 17, 12));
  CHECK(b + a == idh<D>::get(17, 17, 12));
  CHECK(a - b == idh<D>::get(-3, -3, -8));
  CHECK(b - a == idh<D>::get(3, 3, 8));
  CHECK(a * b == idh<D>::get(70, 70, 20));
  CHECK(b * a == idh<D>::get(70, 70, 20));
  CHECK(a / b == idh<D>::get(0, 0, 0));
  CHECK(b / a == idh<D>::get(1, 1, 5));
  CHECK(a % b == idh<D>::get(7, 7, 2));
  CHECK(b % a == idh<D>::get(3, 3, 0));
  CHECK(a << b == idh<D>::get(7168, 7168, 2048));
  CHECK(b << a == idh<D>::get(1280, 1280, 40));
  CHECK(a >> b == idh<D>::get(0, 0, 0));
  CHECK(b >> a == idh<D>::get(0, 0, 2));
  CHECK((a & b) == idh<D>::get(2, 2, 2));
  CHECK((b & a) == idh<D>::get(2, 2, 2));
  CHECK((a | b) == idh<D>::get(15, 15, 10));
  CHECK((b | a) == idh<D>::get(15, 15, 10));
  CHECK((a ^ b) == idh<D>::get(13, 13, 8));
  CHECK((b ^ a) == idh<D>::get(13, 13, 8));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(1, 1, 1));
  CHECK((b < a) == idh<D>::get(0, 0, 0));
  CHECK((a > b) == idh<D>::get(0, 0, 0));
  CHECK((b > a) == idh<D>::get(1, 1, 1));
  CHECK((a <= b) == idh<D>::get(1, 1, 1));
  CHECK((b <= a) == idh<D>::get(0, 0, 0));
  CHECK((a >= b) == idh<D>::get(0, 0, 0));
  CHECK((b >= a) == idh<D>::get(1, 1, 1));

  KCHECK(EVAL_D(a + b) == idh<D>::get(17, 17, 12));
  KCHECK(EVAL_D(b + a) == idh<D>::get(17, 17, 12));
  KCHECK(EVAL_D(a - b) == idh<D>::get(-3, -3, -8));
  KCHECK(EVAL_D(b - a) == idh<D>::get(3, 3, 8));
  KCHECK(EVAL_D(a * b) == idh<D>::get(70, 70, 20));
  KCHECK(EVAL_D(b * a) == idh<D>::get(70, 70, 20));
  KCHECK(EVAL_D(a / b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b / a) == idh<D>::get(1, 1, 5));
  KCHECK(EVAL_D(a % b) == idh<D>::get(7, 7, 2));
  KCHECK(EVAL_D(b % a) == idh<D>::get(3, 3, 0));
  KCHECK(EVAL_D(a << b) == idh<D>::get(7168, 7168, 2048));
  KCHECK(EVAL_D(b << a) == idh<D>::get(1280, 1280, 40));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(0, 0, 2));
  KCHECK(EVAL_D(a & b) == idh<D>::get(2, 2, 2));
  KCHECK(EVAL_D(b & a) == idh<D>::get(2, 2, 2));
  KCHECK(EVAL_D(a | b) == idh<D>::get(15, 15, 10));
  KCHECK(EVAL_D(b | a) == idh<D>::get(15, 15, 10));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(13, 13, 8));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(13, 13, 8));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b < a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a > b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b > a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(1, 1, 1));
});

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP long` and "
 "`long OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(8, 1, 3);
  const long b = 7;

  CHECK(a + b == idh<D>::get(15, 8, 10));
  CHECK(b + a == idh<D>::get(15, 8, 10));
  CHECK(a - b == idh<D>::get(1, -6, -4));
  CHECK(b - a == idh<D>::get(-1, 6, 4));
  CHECK(a * b == idh<D>::get(56, 7, 21));
  CHECK(b * a == idh<D>::get(56, 7, 21));
  CHECK(a / b == idh<D>::get(1, 0, 0));
  CHECK(b / a == idh<D>::get(0, 7, 2));
  CHECK(a % b == idh<D>::get(1, 1, 3));
  CHECK(b % a == idh<D>::get(7, 0, 1));
  CHECK(a << b == idh<D>::get(1024, 128, 384));
  CHECK(b << a == idh<D>::get(1792, 14, 56));
  CHECK(a >> b == idh<D>::get(0, 0, 0));
  CHECK(b >> a == idh<D>::get(0, 3, 0));
  CHECK((a & b) == idh<D>::get(0, 1, 3));
  CHECK((b & a) == idh<D>::get(0, 1, 3));
  CHECK((a | b) == idh<D>::get(15, 7, 7));
  CHECK((b | a) == idh<D>::get(15, 7, 7));
  CHECK((a ^ b) == idh<D>::get(15, 6, 4));
  CHECK((b ^ a) == idh<D>::get(15, 6, 4));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(0, 1, 1));
  CHECK((b < a) == idh<D>::get(1, 0, 0));
  CHECK((a > b) == idh<D>::get(1, 0, 0));
  CHECK((b > a) == idh<D>::get(0, 1, 1));
  CHECK((a <= b) == idh<D>::get(0, 1, 1));
  CHECK((b <= a) == idh<D>::get(1, 0, 0));
  CHECK((a >= b) == idh<D>::get(1, 0, 0));
  CHECK((b >= a) == idh<D>::get(0, 1, 1));

  KCHECK(EVAL_D(a + b) == idh<D>::get(15, 8, 10));
  KCHECK(EVAL_D(b + a) == idh<D>::get(15, 8, 10));
  KCHECK(EVAL_D(a - b) == idh<D>::get(1, -6, -4));
  KCHECK(EVAL_D(b - a) == idh<D>::get(-1, 6, 4));
  KCHECK(EVAL_D(a * b) == idh<D>::get(56, 7, 21));
  KCHECK(EVAL_D(b * a) == idh<D>::get(56, 7, 21));
  KCHECK(EVAL_D(a / b) == idh<D>::get(1, 0, 0));
  KCHECK(EVAL_D(b / a) == idh<D>::get(0, 7, 2));
  KCHECK(EVAL_D(a % b) == idh<D>::get(1, 1, 3));
  KCHECK(EVAL_D(b % a) == idh<D>::get(7, 0, 1));
  KCHECK(EVAL_D(a << b) == idh<D>::get(1024, 128, 384));
  KCHECK(EVAL_D(b << a) == idh<D>::get(1792, 14, 56));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(0, 3, 0));
  KCHECK(EVAL_D(a & b) == idh<D>::get(0, 1, 3));
  KCHECK(EVAL_D(b & a) == idh<D>::get(0, 1, 3));
  KCHECK(EVAL_D(a | b) == idh<D>::get(15, 7, 7));
  KCHECK(EVAL_D(b | a) == idh<D>::get(15, 7, 7));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(15, 6, 4));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(15, 6, 4));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(0, 1, 1));
  KCHECK(EVAL_D(b < a) == idh<D>::get(1, 0, 0));
  KCHECK(EVAL_D(a > b) == idh<D>::get(1, 0, 0));
  KCHECK(EVAL_D(b > a) == idh<D>::get(0, 1, 1));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(0, 1, 1));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(1, 0, 0));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(1, 0, 0));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(0, 1, 1));
});

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports various binary operators of the form `id OP long long` and "
 "`long long OP id`",
 "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(2, 4, 3);
  const long long b = 8;

  CHECK(a + b == idh<D>::get(10, 12, 11));
  CHECK(b + a == idh<D>::get(10, 12, 11));
  CHECK(a - b == idh<D>::get(-6, -4, -5));
  CHECK(b - a == idh<D>::get(6, 4, 5));
  CHECK(a * b == idh<D>::get(16, 32, 24));
  CHECK(b * a == idh<D>::get(16, 32, 24));
  CHECK(a / b == idh<D>::get(0, 0, 0));
  CHECK(b / a == idh<D>::get(4, 2, 2));
  CHECK(a % b == idh<D>::get(2, 4, 3));
  CHECK(b % a == idh<D>::get(0, 0, 2));
  CHECK(a << b == idh<D>::get(512, 1024, 768));
  CHECK(b << a == idh<D>::get(32, 128, 64));
  CHECK(a >> b == idh<D>::get(0, 0, 0));
  CHECK(b >> a == idh<D>::get(2, 0, 1));
  CHECK((a & b) == idh<D>::get(0, 0, 0));
  CHECK((b & a) == idh<D>::get(0, 0, 0));
  CHECK((a | b) == idh<D>::get(10, 12, 11));
  CHECK((b | a) == idh<D>::get(10, 12, 11));
  CHECK((a ^ b) == idh<D>::get(10, 12, 11));
  CHECK((b ^ a) == idh<D>::get(10, 12, 11));
  CHECK((a && b) == idh<D>::get(1, 1, 1));
  CHECK((b && a) == idh<D>::get(1, 1, 1));
  CHECK((a || b) == idh<D>::get(1, 1, 1));
  CHECK((b || a) == idh<D>::get(1, 1, 1));
  CHECK((a < b) == idh<D>::get(1, 1, 1));
  CHECK((b < a) == idh<D>::get(0, 0, 0));
  CHECK((a > b) == idh<D>::get(0, 0, 0));
  CHECK((b > a) == idh<D>::get(1, 1, 1));
  CHECK((a <= b) == idh<D>::get(1, 1, 1));
  CHECK((b <= a) == idh<D>::get(0, 0, 0));
  CHECK((a >= b) == idh<D>::get(0, 0, 0));
  CHECK((b >= a) == idh<D>::get(1, 1, 1));

  KCHECK(EVAL_D(a + b) == idh<D>::get(10, 12, 11));
  KCHECK(EVAL_D(b + a) == idh<D>::get(10, 12, 11));
  KCHECK(EVAL_D(a - b) == idh<D>::get(-6, -4, -5));
  KCHECK(EVAL_D(b - a) == idh<D>::get(6, 4, 5));
  KCHECK(EVAL_D(a * b) == idh<D>::get(16, 32, 24));
  KCHECK(EVAL_D(b * a) == idh<D>::get(16, 32, 24));
  KCHECK(EVAL_D(a / b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b / a) == idh<D>::get(4, 2, 2));
  KCHECK(EVAL_D(a % b) == idh<D>::get(2, 4, 3));
  KCHECK(EVAL_D(b % a) == idh<D>::get(0, 0, 2));
  KCHECK(EVAL_D(a << b) == idh<D>::get(512, 1024, 768));
  KCHECK(EVAL_D(b << a) == idh<D>::get(32, 128, 64));
  KCHECK(EVAL_D(a >> b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >> a) == idh<D>::get(2, 0, 1));
  KCHECK(EVAL_D(a & b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b & a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a | b) == idh<D>::get(10, 12, 11));
  KCHECK(EVAL_D(b | a) == idh<D>::get(10, 12, 11));
  KCHECK(EVAL_D(a ^ b) == idh<D>::get(10, 12, 11));
  KCHECK(EVAL_D(b ^ a) == idh<D>::get(10, 12, 11));
  KCHECK(EVAL_D(a && b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b && a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a || b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b || a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a < b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b < a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a > b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b > a) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(a <= b) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_D(b <= a) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(a >= b) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_D(b >= a) == idh<D>::get(1, 1, 1));
});

#define COMPOUND_OP(operand_value, expr) \
  ([=](auto x) { return expr, x; })(operand_value)

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= id`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(5, 8, 3);
  const auto b = idh<D>::get(4, 8, 2);

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(9, 16, 5));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(1, 0, 1));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(20, 64, 6));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(1, 1, 1));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(1, 0, 1));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(80, 2048, 12));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(4, 8, 2));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(5, 8, 3));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(1, 0, 1));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(9, 16, 5));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(1, 0, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(20, 64, 6));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(1, 0, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) == idh<D>::get(80, 2048, 12));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(4, 8, 2));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(5, 8, 3));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(1, 0, 1));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= size_t`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(5, 8, 3);
  const size_t b = 3;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(8, 11, 6));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(2, 5, 0));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(15, 24, 9));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(1, 2, 1));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(2, 2, 0));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(40, 64, 24));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(0, 1, 0));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(1, 0, 3));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(7, 11, 3));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(6, 11, 0));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(8, 11, 6));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(2, 5, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(15, 24, 9));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(1, 2, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(2, 2, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) == idh<D>::get(40, 64, 24));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 1, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(1, 0, 3));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(7, 11, 3));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(6, 11, 0));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= bool`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(4, 3, 8);
  const bool b = 1;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(5, 4, 9));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(3, 2, 7));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(4, 3, 8));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(4, 3, 8));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(8, 6, 16));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(2, 1, 4));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(0, 1, 0));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(5, 3, 9));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(5, 2, 9));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(5, 4, 9));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(3, 2, 7));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(4, 3, 8));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(4, 3, 8));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) == idh<D>::get(8, 6, 16));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(2, 1, 4));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(0, 1, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(5, 3, 9));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(5, 2, 9));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= char`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(5, 6, 5);
  const char b = 1;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(6, 7, 6));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(4, 5, 4));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(5, 6, 5));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(5, 6, 5));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(10, 12, 10));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(2, 3, 2));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(1, 0, 1));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(5, 7, 5));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(4, 7, 4));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(6, 7, 6));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(4, 5, 4));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(5, 6, 5));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(5, 6, 5));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) == idh<D>::get(10, 12, 10));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(2, 3, 2));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(1, 0, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(5, 7, 5));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(4, 7, 4));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= "
    "char16_t`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(1, 4, 7);
  const char16_t b = 3;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(4, 7, 10));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(-2, 1, 4));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(3, 12, 21));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(0, 1, 2));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(1, 1, 1));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(8, 32, 56));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(1, 0, 3));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(3, 7, 7));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(2, 7, 4));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(4, 7, 10));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(-2, 1, 4));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(3, 12, 21));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(0, 1, 2));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(1, 1, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) == idh<D>::get(8, 32, 56));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(1, 0, 3));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(3, 7, 7));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(2, 7, 4));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= "
    "char32_t`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(5, 6, 7);
  const char32_t b = 6;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(11, 12, 13));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(-1, 0, 1));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(30, 36, 42));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(0, 1, 1));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(5, 0, 1));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(320, 384, 448));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(4, 6, 6));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(7, 6, 7));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(3, 0, 1));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(11, 12, 13));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(-1, 0, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(30, 36, 42));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(0, 1, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(5, 0, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) ==
         idh<D>::get(320, 384, 448));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(4, 6, 6));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(7, 6, 7));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(3, 0, 1));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= "
    "wchar_t`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(9, 2, 9);
  const wchar_t b = 1;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(10, 3, 10));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(8, 1, 8));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(9, 2, 9));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(9, 2, 9));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(18, 4, 18));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(4, 1, 4));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(1, 0, 1));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(9, 3, 9));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(8, 3, 8));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(10, 3, 10));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(8, 1, 8));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(9, 2, 9));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(9, 2, 9));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) == idh<D>::get(18, 4, 18));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(4, 1, 4));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(1, 0, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(9, 3, 9));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(8, 3, 8));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= short`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(3, 4, 5);
  const short b = 6;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(9, 10, 11));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(-3, -2, -1));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(18, 24, 30));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(3, 4, 5));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(192, 256, 320));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(2, 4, 4));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(7, 6, 7));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(5, 2, 3));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(9, 10, 11));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(-3, -2, -1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(18, 24, 30));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(3, 4, 5));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) ==
         idh<D>::get(192, 256, 320));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(2, 4, 4));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(7, 6, 7));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(5, 2, 3));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= int`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(8, 6, 2);
  const int b = 10;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(18, 16, 12));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(-2, -4, -8));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(80, 60, 20));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(8, 6, 2));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(8192, 6144, 2048));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(8, 2, 2));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(10, 14, 10));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(2, 12, 8));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(18, 16, 12));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(-2, -4, -8));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(80, 60, 20));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(8, 6, 2));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) ==
         idh<D>::get(8192, 6144, 2048));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(8, 2, 2));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(10, 14, 10));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(2, 12, 8));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= long`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(10, 8, 2);
  const long b = 7;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(17, 15, 9));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(3, 1, -5));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(70, 56, 14));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(1, 1, 0));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(3, 1, 2));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(1280, 1024, 256));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(2, 0, 2));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(15, 15, 7));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(13, 15, 5));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(17, 15, 9));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(3, 1, -5));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(70, 56, 14));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(1, 1, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(3, 1, 2));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) ==
         idh<D>::get(1280, 1024, 256));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(2, 0, 2));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(15, 15, 7));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(13, 15, 5));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various compound binary operators of the form `id OP= long "
    "long`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(1, 2, 1);
  const long long b = 2;

  CHECK(COMPOUND_OP(a, x += b) == idh<D>::get(3, 4, 3));
  CHECK(COMPOUND_OP(a, x -= b) == idh<D>::get(-1, 0, -1));
  CHECK(COMPOUND_OP(a, x *= b) == idh<D>::get(2, 4, 2));
  CHECK(COMPOUND_OP(a, x /= b) == idh<D>::get(0, 1, 0));
  CHECK(COMPOUND_OP(a, x %= b) == idh<D>::get(1, 0, 1));
  CHECK(COMPOUND_OP(a, x <<= b) == idh<D>::get(4, 8, 4));
  CHECK(COMPOUND_OP(a, x >>= b) == idh<D>::get(0, 0, 0));
  CHECK(COMPOUND_OP(a, x &= b) == idh<D>::get(0, 2, 0));
  CHECK(COMPOUND_OP(a, x |= b) == idh<D>::get(3, 2, 3));
  CHECK(COMPOUND_OP(a, x ^= b) == idh<D>::get(3, 0, 3));

  using sycl::id;
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(3, 4, 3));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(-1, 0, -1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(2, 4, 2));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(0, 1, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(1, 0, 1));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x <<= b)) == idh<D>::get(4, 8, 4));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 0, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(0, 2, 0));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(3, 2, 3));
  KCHECK(EVAL_T_D(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(3, 0, 3));
}

#undef COMPOUND_OP

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports unary +/- operators", "[id]", ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(5, 8, 3);
  const auto b = idh<D>::get(-5, -8, -3);
  CHECK(+a == a);
  CHECK(-a == b);
  CHECK(+b == b);
  CHECK(-b == a);

  KCHECK(EVAL(+a) == a);
  KCHECK(EVAL(-a) == b);
  KCHECK(EVAL(+b) == b);
  KCHECK(EVAL(-b) == a);
});

#define INC_DEC_OP(operand_value, expr) \
  ([=](auto x) { return std::pair{expr, x}; })(operand_value)

DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(AdaptiveCpp)
("id supports pre- and postfix increment/decrement operators", "[id]",
 ((int D), D), 1, 2, 3)({
  const auto a = idh<D>::get(5, 8, 3);
  const auto b = idh<D>::get(6, 9, 4);
  const auto c = idh<D>::get(4, 7, 2);

  CHECK(INC_DEC_OP(a, ++x) == std::pair{b, b});
  CHECK(INC_DEC_OP(a, --x) == std::pair{c, c});
  CHECK(INC_DEC_OP(a, x++) == std::pair{a, b});
  CHECK(INC_DEC_OP(a, x--) == std::pair{a, c});

  using id_pair = std::pair<sycl::id<D>, sycl::id<D>>;

  KCHECK(EVAL_T_D(id_pair, INC_DEC_OP(a, ++x)) == std::pair{b, b});
  KCHECK(EVAL_T_D(id_pair, INC_DEC_OP(a, --x)) == std::pair{c, c});
  KCHECK(EVAL_T_D(id_pair, INC_DEC_OP(a, x++)) == std::pair{a, b});
  KCHECK(EVAL_T_D(id_pair, INC_DEC_OP(a, x--)) == std::pair{a, c});
});

#undef INC_DEC_OP

TEST_CASE("id can deduce dimensionality from constructor parameters", "[id]") {
  using sycl::id;
  CHECK(std::is_same_v<decltype(id{5}), id<1>>);
  CHECK(std::is_same_v<decltype(id{5, 8}), id<2>>);
  CHECK(std::is_same_v<decltype(id{5, 8, 3}), id<3>>);
  KCHECK(EVAL((std::is_same_v<decltype(id{5}), id<1>>)));
  KCHECK(EVAL((std::is_same_v<decltype(id{5, 8}), id<2>>)));
  KCHECK(EVAL((std::is_same_v<decltype(id{5, 8, 3}), id<3>>)));
}
