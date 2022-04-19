/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  This file provides functions for tests on accessor default values.
//
*******************************************************************************/
#ifndef SYCL_CTS_ACCESSOR_DEFAULT_VALUES_H
#define SYCL_CTS_ACCESSOR_DEFAULT_VALUES_H

#include "accessor_common.h"
#include "catch2/catch_test_macros.hpp"

#include <type_traits>

namespace accessor_default_values_test {

constexpr int expected_dims = 1;
constexpr sycl::target expected_target = sycl::target::device;
constexpr sycl::access::placeholder expected_placeholder =
    sycl::access::placeholder::false_t;

/**
 * @brief Provides functor that lets verify that local_accessor's template
 *        parameters have expected default values
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T>
void test_exception_for_local_acc(const std::string& type_name) {
  auto section_name =
      "Verify default value for dimensions template parameter for " +
      type_name + " data type. [local_accessor]";
  SECTION(section_name) {
    REQUIRE(std::is_same_v<sycl::local_accessor<T>,
                           sycl::local_accessor<T, expected_dims>>);
  }
}

/**
 * @brief Provides functor that lets verify that host_accessor's template
 *        parameters have expected default values
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T>
void test_for_host_acc(const std::string& type_name) {
  auto section_name =
      "Verify default value for dimensions template parameter for " +
      type_name + " data type. [host_accessor]";
  SECTION(section_name) {
    REQUIRE(std::is_same_v<sycl::host_accessor<T>,
                           sycl::host_accessor<T, expected_dims>>);
  }
  section_name = "Verify default value for accessMode template parameter for " +
                 type_name + " data type. [host_accessor]";
  SECTION(section_name) {
    REQUIRE(std::is_same_v<
            sycl::host_accessor<T, expected_dims>,
            sycl::host_accessor<T, expected_dims,
                                std::is_const_v<T>
                                    ? sycl::access_mode::read
                                    : sycl::access_mode::read_write>>);
  }
}

/**
 * @brief Provides functor that lets verify that generic accessor's template
 *        parameters have expected default values
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T>
void test_for_generic_acc(const std::string& type_name) {
  auto section_name =
      "Verify default value for dimensions template parameter for " +
      type_name + " data type. [generic accessor]";
  SECTION(section_name) {
    REQUIRE(
        std::is_same_v<sycl::accessor<T>, sycl::accessor<T, expected_dims>>);
  }
  section_name = "Verify default value for accessMode template parameter for " +
                 type_name + " data type. [generic accessor]";
  SECTION(section_name) {
    REQUIRE(
        std::is_same_v<sycl::accessor<T, expected_dims>,
                       sycl::accessor<T, expected_dims,
                                      std::is_const_v<T>
                                          ? sycl::access_mode::read
                                          : sycl::access_mode::read_write>>);
  }
  section_name = "Verify default value for accessMode template parameter for " +
                 type_name + " data type. [generic accessor]";
  SECTION(section_name) {
    REQUIRE(
        std::is_same_v<sycl::accessor<T, expected_dims>,
                       sycl::accessor<T, expected_dims,
                                      std::is_const_v<T>
                                          ? sycl::access_mode::read
                                          : sycl::access_mode::read_write>>);
  }
  section_name =
      "Verify default value for accessTarget template parameter for " +
      type_name + " data type. [generic accessor]";
  SECTION(section_name) {
    REQUIRE(std::is_same_v<
            sycl::accessor<T, expected_dims,
                           std::is_const_v<T> ? sycl::access_mode::read
                                              : sycl::access_mode::read_write>,
            sycl::accessor<T, expected_dims,
                           std::is_const_v<T> ? sycl::access_mode::read
                                              : sycl::access_mode::read_write,
                           expected_target>>);
  }
  section_name =
      "Verify default value for isPlaceholder template parameter for " +
      type_name + " data type. [generic accessor]";
  SECTION(section_name) {
    REQUIRE(std::is_same_v<
            sycl::accessor<T, expected_dims,
                           std::is_const_v<T> ? sycl::access_mode::read
                                              : sycl::access_mode::read_write,
                           expected_target>,
            sycl::accessor<T, expected_dims,
                           std::is_const_v<T> ? sycl::access_mode::read
                                              : sycl::access_mode::read_write,
                           expected_target, expected_placeholder>>);
  }
}

/**
 * @brief Struct with functor that will be used in type coverage to call all
 *        verification functions.
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T>
class run_tests {
 public:
  void operator()(const std::string& type_name) {
    test_for_generic_acc<T>(type_name);
    test_for_host_acc<T>(type_name);
    test_exception_for_local_acc<T>(type_name);
  }
};

}  // namespace accessor_default_values_test

#endif  // SYCL_CTS_ACCESSOR_DEFAULT_VALUES_H
