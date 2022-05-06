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

#include <type_traits>

namespace accessor_default_values_test {

namespace detail {
constexpr int expected_dims = 1;
constexpr sycl::target expected_target = sycl::target::device;
constexpr sycl::access::placeholder expected_placeholder =
    sycl::access::placeholder::false_t;
template <typename T>
constexpr auto expected_mode =
    std::is_const_v<T> ? sycl::access_mode::read : sycl::access_mode::read_write
}  // namespace detail

/**
 * @brief Provides functor that lets verify that local_accessor's template
 *        parameters have expected default values
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T>
class test_for_local_acc {
 public:
  void operator()(const std::string& type_name) {
    auto section_name =
        "Verify default value for dimensions template parameter for " +
        type_name + " data type. [local_accessor]";
    SECTION(section_name) {
      REQUIRE(std::is_same_v<sycl::local_accessor<T>,
                             sycl::local_accessor<T, detail::expected_dims>>);
    }
  }
};

/**
 * @brief Provides functor that lets verify that host_accessor's template
 *        parameters have expected default values
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */

template <typename T, typename AccessModeT, typename DimensionT,
          typename TargetT>
class test_for_host_acc {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    auto section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Verify default value for dimensions template parameter.");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<sycl::host_accessor<T>,
                             sycl::host_accessor<T, detail::expected_dims>>);
    }
    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Verify default value for accessMode template parameter.");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<
              sycl::host_accessor<T, Dimension>,
              sycl::host_accessor<T, Dimension, detail::expected_mode<T>>>);
    }
  }
};

/**
 * @brief Provides functor that lets verify that generic accessor's template
 *        parameters have expected default values
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T, typename AccessModeT, typename DimensionT,
          typename TargetT>
class test_for_generic_acc {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    auto section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Verify default value for dimensions template parameter.");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<sycl::accessor<T>,
                             sycl::accessor<T, detail::expected_dims>>);
    }
    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Verify default value for accessMode template parameter.");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<
              sycl::accessor<T, Dimension>,
              sycl::accessor<T, Dimension, detail::expected_mode<T>>>);
    }
    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Verify default value for accessTarget template parameter.");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<sycl::accessor<T, Dimension, AccessMode>,
                             sycl::accessor<T, Dimension, AccessMode,
                                            detail::expected_target>>);
    }
    section_name = get_section_name<Dimension>(
        type_name, access_mode_name, target_name,
        "Verify default value for isPlaceholder template parameter.");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<sycl::accessor<T, Dimension, AccessMode, Target>,
                             sycl::accessor<T, Dimension, AccessMode, Target,
                                            detail::expected_placeholder>>);
    }
  }
};

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
    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = get_access_modes();
    const auto dimensions = get_all_dimensions();
    const auto targets = get_targets();

    // Run test with non const data type
    for_all_combinations<test_for_generic_acc, T>(access_modes, dimensions,
                                                  targets, type_name);
    for_all_combinations<test_for_host_acc, T>(access_modes, dimensions,
                                               targets, type_name);
    test_for_local_acc<T>(type_name);
    // Run test with const data type
    const auto const_type_name = "const " + type_name;
    for_all_combinations<test_for_generic_acc, const T>(
        access_modes, dimensions, targets, const_type_name);
    for_all_combinations<test_for_host_acc, const T>(access_modes, dimensions,
                                                     targets, const_type_name);
    test_for_local_acc<const T>(const_type_name);
  }
};

}  // namespace accessor_default_values_test

#endif  // SYCL_CTS_ACCESSOR_DEFAULT_VALUES_H
