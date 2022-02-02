/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
*******************************************************************************/

#include <type_traits>
#include <utility>

#include <catch2/catch_template_test_macros.hpp>

#include "../common/common.h"
#include "../common/device_eval.h"

using namespace sycl_cts;

TEST_CASE("id provides a default constructor", "[id]") {
  using sycl::id;
  STATIC_CHECK(std::is_default_constructible_v<id<1>>);
  STATIC_CHECK(std::is_default_constructible_v<id<2>>);
  STATIC_CHECK(std::is_default_constructible_v<id<3>>);

  CHECK(id<1>{} == id<1>{0});
  CHECK(id<2>{} == id<2>{0, 0});
  CHECK(id<3>{} == id<3>{0, 0, 0});

  CHECK(DEVICE_EVAL(id<1>{}) == id<1>{0});
  CHECK(DEVICE_EVAL(id<2>{}) == id<2>{0, 0});
  CHECK(DEVICE_EVAL(id<3>{}) == id<3>{0, 0, 0});
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

  CHECK(DEVICE_EVAL(id<1>{5}) == id<1>{5});
  CHECK(DEVICE_EVAL((id<2>{5, 8})) == id<2>{5, 8});
  CHECK(DEVICE_EVAL((id<3>{5, 8, 3})) == id<3>{5, 8, 3});
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
    CHECK(DEVICE_EVAL(std::is_trivially_copy_constructible_v<id<D>>));

    const auto copy = [] {
      const auto a = idh<D>::get(5, 8, 3);
      id<D> b{a};
      return b;
    };
    CHECK(copy() == idh<D>::get(5, 8, 3));
    CHECK(DEVICE_EVAL_T(id<D>, copy()) == idh<D>::get(5, 8, 3));
  }

  SECTION("copy assignment operator") {
    CHECK(std::is_trivially_copy_assignable_v<id<D>>);
    CHECK(DEVICE_EVAL(std::is_trivially_copy_assignable_v<id<D>>));

    const auto copy = [] {
      const auto a = idh<D>::get(5, 8, 3);
      id<D> b;
      b = a;
      return b;
    };
    CHECK(copy() == idh<D>::get(5, 8, 3));
    CHECK(DEVICE_EVAL_T(id<D>, copy()) == idh<D>::get(5, 8, 3));
  }

  SECTION("destructor") {
    CHECK(std::is_trivially_destructible_v<id<D>>);
    CHECK(DEVICE_EVAL(std::is_trivially_destructible_v<id<D>>));
  }

  SECTION("move constructor") {
    CHECK(std::is_trivially_move_constructible_v<id<D>>);
    CHECK(DEVICE_EVAL(std::is_trivially_move_constructible_v<id<D>>));

    const auto move = [] {
      auto a = idh<D>::get(5, 8, 3);
      id<D> b{std::move(a)};
      return b;
    };
    CHECK(move() == idh<D>::get(5, 8, 3));
    CHECK(DEVICE_EVAL_T(id<D>, move()) == idh<D>::get(5, 8, 3));
  }

  SECTION("move assignment operator") {
    CHECK(std::is_trivially_move_assignable_v<id<D>>);
    CHECK(DEVICE_EVAL(std::is_trivially_move_assignable_v<id<D>>));

    const auto move = [] {
      auto a = idh<D>::get(5, 8, 3);
      id<D> b;
      b = std::move(a);
      return b;
    };
    CHECK(move() == idh<D>::get(5, 8, 3));
    CHECK(DEVICE_EVAL_T(id<D>, move()) == idh<D>::get(5, 8, 3));
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

    CHECK(DEVICE_EVAL(a1 == a1));
    CHECK(DEVICE_EVAL(a1 == a2));
    CHECK(DEVICE_EVAL(a2 == a1));
    CHECK(DEVICE_EVAL(b1 == b1));
    CHECK_FALSE(DEVICE_EVAL(a1 == b1));
    CHECK_FALSE(DEVICE_EVAL(b1 == a1));
    CHECK_FALSE(DEVICE_EVAL(a2 == b1));

    CHECK_FALSE(a1 != a1);
    CHECK_FALSE(a1 != a2);
    CHECK_FALSE(a2 != a1);
    CHECK_FALSE(b1 != b1);
    CHECK(a1 != b1);
    CHECK(b1 != a1);
    CHECK(a2 != b1);

    CHECK_FALSE(DEVICE_EVAL(a1 != a1));
    CHECK_FALSE(DEVICE_EVAL(a1 != a2));
    CHECK_FALSE(DEVICE_EVAL(a2 != a1));
    CHECK_FALSE(DEVICE_EVAL(b1 != b1));
    CHECK(DEVICE_EVAL(a1 != b1));
    CHECK(DEVICE_EVAL(b1 != a1));
    CHECK(DEVICE_EVAL(a2 != b1));
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
  CHECK(DEVICE_EVAL_T(id<D>, convert()) == idh<D>::get(5, 8, 3));
}

TEMPLATE_TEST_CASE_SIG("id can be implicitly conversion-constructed from item",
                       "[id]", ((int D), D), 1, 2, 3) {
  using sycl::id;

  auto q = util::get_cts_object::queue();
  sycl::buffer<id<D>, 1> result_buf{1};
  const auto r = util::get_cts_object::range<D>::get(5, 8, 3);
  q.submit([r, &result_buf](sycl::handler& cgh) {
     sycl::accessor result{result_buf, cgh, sycl::write_only};
     cgh.parallel_for(r, [=](sycl::item<D> itm) {
       if (itm.get_id() == id<D>{r} - 1) {
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
    CHECK(DEVICE_EVAL(a.get(i)) == values[i]);
    CHECK(DEVICE_EVAL(a[i]) == values[i]);
  }

  const auto assign_component = [](auto x, auto c, auto v) {
    x[c] = v;
    return x;
  };

  using sycl::id;

  CHECK(assign_component(a, 0, 7) == idh<D>::get(7, 8, 3));
  CHECK(DEVICE_EVAL_T(id<D>, assign_component(a, 0, 7)) ==
        idh<D>::get(7, 8, 3));

  if (D >= 2) {
    CHECK(assign_component(a, 1, 9) == idh<D>::get(5, 9, 3));
    CHECK(DEVICE_EVAL_T(id<D>, assign_component(a, 1, 9)) ==
          idh<D>::get(5, 9, 3));
  }

  if (D == 3) {
    CHECK(assign_component(a, 2, 11) == idh<D>::get(5, 8, 11));
    CHECK(DEVICE_EVAL_T(id<D>, assign_component(a, 2, 11)) ==
          idh<D>::get(5, 8, 11));
  }
}

TEST_CASE("id can be converted to size_t if Dimensions == 1", "[id]") {
  using sycl::id;
  const auto convert = [] {
    const sycl::id a{42};
    const size_t b = a;
    return b;
  };
  CHECK(convert() == 42);
  CHECK(DEVICE_EVAL(convert()) == 42);
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

  CHECK(DEVICE_EVAL(a + b) == idh<D>::get(9, 16, 5));
  CHECK(DEVICE_EVAL(a - b) == idh<D>::get(1, 0, 1));
  CHECK(DEVICE_EVAL(a * b) == idh<D>::get(20, 64, 6));
  CHECK(DEVICE_EVAL(a / b) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL(a % b) == idh<D>::get(1, 0, 1));
  CHECK(DEVICE_EVAL(a << b) == idh<D>::get(80, 2048, 12));
  CHECK(DEVICE_EVAL(a >> b) == idh<D>::get(0, 0, 0));
  CHECK(DEVICE_EVAL(a & b) == idh<D>::get(4, 8, 2));
  CHECK(DEVICE_EVAL(a | b) == idh<D>::get(5, 8, 3));
  CHECK(DEVICE_EVAL(a ^ b) == idh<D>::get(1, 0, 1));
  CHECK(DEVICE_EVAL(a && b) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL(a || b) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL(a < b) == idh<D>::get(0, 0, 0));
  CHECK(DEVICE_EVAL(a > b) == idh<D>::get(1, 0, 1));
  CHECK(DEVICE_EVAL(a <= b) == idh<D>::get(0, 1, 0));
  CHECK(DEVICE_EVAL(a >= b) == idh<D>::get(1, 1, 1));
}

TEMPLATE_TEST_CASE_SIG(
    "id supports various binary operators of the form `id OP size_t` and "
    "`size_t OP id`",
    "[id]", ((int D), D), 1, 2, 3) {
  const auto a = idh<D>::get(5, 8, 3);
  const size_t b = 3;

  CHECK(a + b == idh<D>::get(8, 11, 6));
  CHECK(b + a == idh<D>::get(8, 11, 6));
  CHECK(a - b == idh<D>::get(2, 5, 0));
  CHECK(b - a == idh<D>::get(-2ul, -5ul, 0));
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

  CHECK(DEVICE_EVAL(a + b) == idh<D>::get(8, 11, 6));
  CHECK(DEVICE_EVAL(b + a) == idh<D>::get(8, 11, 6));
  CHECK(DEVICE_EVAL(a - b) == idh<D>::get(2, 5, 0));
  CHECK(DEVICE_EVAL(b - a) == idh<D>::get(-2ul, -5ul, 0));
  CHECK(DEVICE_EVAL(a * b) == idh<D>::get(15, 24, 9));
  CHECK(DEVICE_EVAL(b * a) == idh<D>::get(15, 24, 9));
  CHECK(DEVICE_EVAL(a / b) == idh<D>::get(1, 2, 1));
  CHECK(DEVICE_EVAL(b / a) == idh<D>::get(0, 0, 1));
  CHECK(DEVICE_EVAL(a % b) == idh<D>::get(2, 2, 0));
  CHECK(DEVICE_EVAL(b % a) == idh<D>::get(3, 3, 0));
  CHECK(DEVICE_EVAL(a << b) == idh<D>::get(40, 64, 24));
  CHECK(DEVICE_EVAL(b << a) == idh<D>::get(96, 768, 24));
  CHECK(DEVICE_EVAL(a >> b) == idh<D>::get(0, 1, 0));
  CHECK(DEVICE_EVAL(b >> a) == idh<D>::get(0, 0, 0));
  CHECK(DEVICE_EVAL(a & b) == idh<D>::get(1, 0, 3));
  CHECK(DEVICE_EVAL(b & a) == idh<D>::get(1, 0, 3));
  CHECK(DEVICE_EVAL(a | b) == idh<D>::get(7, 11, 3));
  CHECK(DEVICE_EVAL(b | a) == idh<D>::get(7, 11, 3));
  CHECK(DEVICE_EVAL(a ^ b) == idh<D>::get(6, 11, 0));
  CHECK(DEVICE_EVAL(b ^ a) == idh<D>::get(6, 11, 0));
  CHECK(DEVICE_EVAL(a && b) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL(b && a) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL(a || b) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL(b || a) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL(a < b) == idh<D>::get(0, 0, 0));
  CHECK(DEVICE_EVAL(b < a) == idh<D>::get(1, 1, 0));
  CHECK(DEVICE_EVAL(a > b) == idh<D>::get(1, 1, 0));
  CHECK(DEVICE_EVAL(b > a) == idh<D>::get(0, 0, 0));
  CHECK(DEVICE_EVAL(a <= b) == idh<D>::get(0, 0, 1));
  CHECK(DEVICE_EVAL(b <= a) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL(a >= b) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL(b >= a) == idh<D>::get(0, 0, 1));
}

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
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(9, 16, 5));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(1, 0, 1));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(20, 64, 6));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(1, 1, 1));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(1, 0, 1));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x <<= b)) ==
        idh<D>::get(80, 2048, 12));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 0, 0));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(4, 8, 2));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(5, 8, 3));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(1, 0, 1));
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
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x += b)) == idh<D>::get(8, 11, 6));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x -= b)) == idh<D>::get(2, 5, 0));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x *= b)) == idh<D>::get(15, 24, 9));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x /= b)) == idh<D>::get(1, 2, 1));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x %= b)) == idh<D>::get(2, 2, 0));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x <<= b)) ==
        idh<D>::get(40, 64, 24));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x >>= b)) == idh<D>::get(0, 1, 0));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x &= b)) == idh<D>::get(1, 0, 3));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x |= b)) == idh<D>::get(7, 11, 3));
  CHECK(DEVICE_EVAL_T(id<D>, COMPOUND_OP(a, x ^= b)) == idh<D>::get(6, 11, 0));
}

#undef COMPOUND_OP

TEMPLATE_TEST_CASE_SIG("id supports unary +/- operators", "[id]", ((int D), D),
                       1, 2, 3) {
  const auto a = idh<D>::get(5, 8, 3);
  const auto b = idh<D>::get(-5, -8, -3);
  CHECK(+a == a);
  CHECK(-a == b);
  CHECK(+b == b);
  CHECK(-b == a);

  CHECK(DEVICE_EVAL(+a) == a);
  CHECK(DEVICE_EVAL(-a) == b);
  CHECK(DEVICE_EVAL(+b) == b);
  CHECK(DEVICE_EVAL(-b) == a);
}

TEMPLATE_TEST_CASE_SIG(
    "id supports pre- and postfix increment/decrement operators", "[id]",
    ((int D), D), 1, 2, 3) {
#define INC_DEC_OP(operand_value, expr) \
  ([=](auto x) { return std::pair{expr, x}; })(operand_value)

  const auto a = idh<D>::get(5, 8, 3);
  const auto b = idh<D>::get(6, 9, 4);
  const auto c = idh<D>::get(4, 7, 2);

  CHECK(INC_DEC_OP(a, ++x) == std::pair{b, b});
  CHECK(INC_DEC_OP(a, --x) == std::pair{c, c});
  CHECK(INC_DEC_OP(a, x++) == std::pair{a, b});
  CHECK(INC_DEC_OP(a, x--) == std::pair{a, c});

  using id_pair = std::pair<sycl::id<D>, sycl::id<D>>;

  CHECK(DEVICE_EVAL_T(id_pair, INC_DEC_OP(a, ++x)) == std::pair{b, b});
  CHECK(DEVICE_EVAL_T(id_pair, INC_DEC_OP(a, --x)) == std::pair{c, c});
  CHECK(DEVICE_EVAL_T(id_pair, INC_DEC_OP(a, x++)) == std::pair{a, b});
  CHECK(DEVICE_EVAL_T(id_pair, INC_DEC_OP(a, x--)) == std::pair{a, c});

#undef INC_DEC_OP
}

TEST_CASE("id can deduce dimensionality from constructor parameters", "[id]") {
  using sycl::id;
  CHECK(std::is_same_v<decltype(id{5}), id<1>>);
  CHECK(std::is_same_v<decltype(id{5, 8}), id<2>>);
  CHECK(std::is_same_v<decltype(id{5, 8, 3}), id<3>>);
  CHECK(DEVICE_EVAL((std::is_same_v<decltype(id{5}), id<1>>)));
  CHECK(DEVICE_EVAL((std::is_same_v<decltype(id{5, 8}), id<2>>)));
  CHECK(DEVICE_EVAL((std::is_same_v<decltype(id{5, 8, 3}), id<3>>)));
}
