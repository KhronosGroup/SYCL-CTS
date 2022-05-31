/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This file contains tests for functions presented in value_operations.h
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_TESTS_H
#define __SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_TESTS_H

#include "../common/common.h"
#include "../common/value_operations.h"
#include <array>
#include <optional>
#include <tuple>
#include <variant>

TEST_CASE("Assign int to array", "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 42;
  const int i2 = 2;
  std::array ar = {i1, i1, i1};
  value_operations::assign(ar, i2);
  CHECK(std::get<0>(ar) == i2);
  CHECK(std::get<1>(ar) == i2);
  CHECK(std::get<2>(ar) == i2);
}

TEST_CASE("Assign array to array", "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 42;
  const int i2 = 2;
  std::array ar1 = {i1, i1, i1};
  std::array ar2 = {i2, i2, i2};
  value_operations::assign(ar1, ar2);
  CHECK(std::get<0>(ar1) == i2);
  CHECK(std::get<1>(ar1) == i2);
  CHECK(std::get<2>(ar1) == i2);
}

TEST_CASE("Assign int to array", "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 42;
  const int i2 = 2;
  std::array ar = {i1, i1, i1};
  value_operations::assign(ar, i2);
  CHECK(std::get<0>(ar) == i2);
  CHECK(std::get<1>(ar) == i2);
  CHECK(std::get<2>(ar) == i2);
}

TEST_CASE("Assign int to int", "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 2;
  const int i2 = 42;
  value_operations::assign(i1, i2);
  CHECK(i1 == i2);
}

TEST_CASE("Assign int to tuple", "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 42;
  const int i2 = 2;
  std::tuple t = {i1, i1, i1};
  value_operations::assign(t, i2);
  CHECK(std::get<0>(t) == i2);
  CHECK(std::get<1>(t) == i2);
  CHECK(std::get<2>(t) == i2);
}

TEST_CASE("Assign tuple to tuple", "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 42;
  const int i2 = 2;
  std::tuple t1 = {i1, i1, i1};
  std::tuple t2 = {i2, i2, i2};
  value_operations::assign(t1, t2);
  CHECK(std::get<0>(t1) == i2);
  CHECK(std::get<1>(t1) == i2);
  CHECK(std::get<2>(t1) == i2);
}

TEST_CASE("Assign int to pair", "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 42;
  const int i2 = 2;
  std::pair p = {i1, i1};
  value_operations::assign(p, i2);
  CHECK(p.first == i2);
  CHECK(p.second == i2);
}

TEST_CASE("Assign pair to pair", "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 42;
  const int i2 = 2;
  std::pair p1 = {i1, i1};
  std::pair p2 = {i2, i2};
  value_operations::assign(p1, p2);
  CHECK(p1.first == i2);
  CHECK(p1.second == i2);
}

TEST_CASE("Assign int to optional",
          "[cts_unit_tests][value_operations_tests]") {
  std::optional<int> opt = 1;
  value_operations::assign(opt, 42);
  CHECK(opt == 42);
}

TEST_CASE("Assign optional to optional",
          "[cts_unit_tests][value_operations_tests]") {
  std::optional<int> opt1;
  std::optional<int> opt2 = 42;
  value_operations::assign(opt1, opt2);
  CHECK(opt1 == 42);
}

TEST_CASE("Assign int to variant", "[cts_unit_tests][value_operations_tests]") {
  std::variant<int, bool, double> var;
  value_operations::assign(var, 42);
  CHECK(std::get<int>(var) == 42);
}

TEST_CASE("Assign variant to variant",
          "[cts_unit_tests][value_operations_tests]") {
  const double d1 = 42.0;
  std::variant<int, bool, double> var1;
  std::variant<int, bool, double> var2 = d1;
  value_operations::assign(var1, var2);
  CHECK(std::get<double>(var1) == d1);
}

TEST_CASE("Compare array with int",
          "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 42;
  const int i2 = 2;
  std::array ar = {i1, i1, i1};
  CHECK_FALSE(value_operations::are_equal(ar, i2));
  CHECK(value_operations::are_equal(ar, i1));
}

TEST_CASE("Compare array with array",
          "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 42;
  const int i2 = 2;
  std::array ar1 = {i1, i1, i1};
  std::array ar2 = {i2, i2, i2};
  std::array ar3 = {i1, i1, i1};
  CHECK_FALSE(value_operations::are_equal(ar1, ar2));
  CHECK(value_operations::are_equal(ar1, ar3));
}

TEST_CASE("Compare int with tuple",
          "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 1;
  const int i2 = 42;
  std::tuple t = {i1, i1, i1};
  CHECK_FALSE(value_operations::are_equal(t, i2));
  CHECK(value_operations::are_equal(t, i1));
}

TEST_CASE("Compare tuple with tuple",
          "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 1;
  const int i2 = 42;
  std::tuple t1 = {i1, i1, i1};
  std::tuple t2 = {i2, i2, i2};
  std::tuple t3 = {i1, i1, i1};
  CHECK_FALSE(value_operations::are_equal(t1, t2));
  CHECK(value_operations::are_equal(t1, t3));
}

TEST_CASE("Compare int with pair", "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 1;
  const int i2 = 42;
  std::pair p = {i1, i1};
  CHECK_FALSE(value_operations::are_equal(p, i2));
  CHECK(value_operations::are_equal(p, i1));
}

TEST_CASE("Compare pair with pair",
          "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 1;
  const int i2 = 42;
  std::pair p1 = {i1, i1};
  std::pair p2 = {i2, i2};
  std::pair p3 = {i1, i1};
  CHECK_FALSE(value_operations::are_equal(p1, p2));
  CHECK(value_operations::are_equal(p1, p3));
}

TEST_CASE("Compare int with variant",
          "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 1;
  const int i2 = 42;
  std::variant<int, bool, double> v = i1;
  CHECK_FALSE(value_operations::are_equal(v, i2));
  CHECK(value_operations::are_equal(v, i1));
}

TEST_CASE("Compare variant with pair",
          "[cts_unit_tests][value_operations_tests]") {
  const int i1 = 1;
  const int i2 = 42;
  std::variant<int, bool, double> v1 = i1;
  std::variant<int, bool, double> v2 = i2;
  std::variant<int, bool, double> v3 = i1;
  CHECK_FALSE(value_operations::are_equal(v1, v2));
  CHECK(value_operations::are_equal(v1, v3));
}
#endif  //__SYCLCTS_TESTS_COMMON_VALUE_OPERATIONS_TESTS_H
