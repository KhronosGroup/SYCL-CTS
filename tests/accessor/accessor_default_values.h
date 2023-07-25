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
using namespace accessor_tests_common;

namespace detail {

constexpr int expected_dims = 1;
constexpr auto expected_target = sycl::target::device;
constexpr auto expected_placeholder = sycl::access::placeholder::false_t;
template <typename T>
constexpr auto expected_mode =
    std::is_const_v<T> ? sycl::access_mode::read
                       : sycl::access_mode::read_write;

}  // namespace detail

/**
 * @brief Provides verification that local_accessor's dimension default value
 *        is equal to expected one
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T>
void verify_local_acc_dim_value(const std::string& type_name) {
  auto section_name = get_section_name<detail::expected_dims>(
      type_name,
      "Verify default value for dimensions "
      "template parameter. [local_accessor]");
  SECTION(section_name) {
    REQUIRE(std::is_same_v<sycl::local_accessor<T>,
                           sycl::local_accessor<T, detail::expected_dims>>);
  }
}

/**
 * @brief Provides verification that host_accessor's dimension default value
 *        is equal to expected one
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T>
void verify_host_acc_dim_value(const std::string& type_name) {
  auto section_name = get_section_name<detail::expected_dims>(
      type_name,
      "Verify default value for dimensions "
      "template parameter. [host_accessor]");
  SECTION(section_name) {
    REQUIRE(std::is_same_v<sycl::host_accessor<T>,
                           sycl::host_accessor<T, detail::expected_dims>>);
  }
}

/**
 * @brief Provides verification that host_accessor's access_mode default
 * value is equal to expected one
 * @tparam T Current data type
 * @tparam DimensionT Current dimension size
 * @param type_name Current data type string representation
 */
template <typename T, typename DimensionT>
class test_for_host_acc_access_mode_val_verification {
  static constexpr int Dimension = DimensionT::value;

 public:
  void operator()(const std::string& type_name) {
    auto section_name =
        get_section_name<Dimension>(type_name,
                                    "Verify default value for accessMode "
                                    "template parameter. [host_accessor]");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<
              sycl::host_accessor<T, Dimension>,
              sycl::host_accessor<T, Dimension, detail::expected_mode<T>>>);
    }
  }
};

/**
 * @brief Provides verification that generic accessor's dimension default value
 *        is equal to expected one
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T>
void verify_generic_acc_dim_value(const std::string& type_name) {
  auto section_name = get_section_name<detail::expected_dims>(
      type_name,
      "Verify default value for dimensions "
      "template parameter. [generic accessor]");
  SECTION(section_name) {
    REQUIRE(std::is_same_v<sycl::accessor<T>,
                           sycl::accessor<T, detail::expected_dims>>);
  }
}

/**
 * @brief Provides verification that generic accessor's access_mode default
 * value is equal to expected one
 * @tparam T Current data type
 * @param type_name Current data type string representation
 */
template <typename T, typename DimensionT>
class test_for_generic_acc_acc_mode_val_verification {
  static constexpr int Dimension = DimensionT::value;

 public:
  void operator()(const std::string& type_name) {
    auto section_name =
        get_section_name<Dimension>(type_name,
                                    "Verify default value for accessMode "
                                    "template parameter. [generic accessor]");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<
              sycl::accessor<T, Dimension>,
              sycl::accessor<T, Dimension, detail::expected_mode<T>>>);
    }
  }
};

/**
 * @brief Provides verification that generic accessor's sycl::target default
 *        value is equal to expected one
 * @tparam T Current data type
 * @tparam AccessModeT sycl::access_mode enumeration's field
 * @tparam DimensionT Current dimension size
 * @param type_name Current data type string representation
 * @param access_mode_name Current sycl::access_mode string representation
 */
template <typename T, typename AccessModeT, typename DimensionT>
class test_for_generic_acc_target_val_verification {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name) {
    auto section_name =
        get_section_name<Dimension>(type_name, access_mode_name,
                                    "Verify default value for accessTarget "
                                    "template parameter. [generic accessor]");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<sycl::accessor<T, Dimension, AccessMode>,
                             sycl::accessor<T, Dimension, AccessMode,
                                            detail::expected_target>>);
    }
  }
};

/**
 * @brief Provides verification that generic accessor's
 *        sycl::access::placeholder default value is equal to expected one
 * @tparam T Current data type
 * @tparam AccessModeT sycl::access_mode enumeration's field
 * @tparam DimensionT Current dimension size
 * @tparam TargetT sycl::target enumeration's field
 * @param type_name Current data type string representation
 * @param access_mode_name Current sycl::access_mode string representation
 * @param target_name Current sycl::target string representation
 */
template <typename T, typename AccessModeT, typename DimensionT,
          typename TargetT>
class test_for_generic_acc_placeholder_val_verification {
  static constexpr sycl::access_mode AccessMode = AccessModeT::value;
  static constexpr int Dimension = DimensionT::value;
  static constexpr sycl::target Target = TargetT::value;

 public:
  void operator()(const std::string& type_name,
                  const std::string& access_mode_name,
                  const std::string& target_name) {
    auto section_name =
        get_section_name<Dimension>(type_name, access_mode_name, target_name,
                                    "Verify default value for isPlaceholder "
                                    "template parameter. [generic accessor]");
    SECTION(section_name) {
      REQUIRE(std::is_same_v<sycl::accessor<T, Dimension, AccessMode, Target>,
                             sycl::accessor<T, Dimension, AccessMode, Target,
                                            detail::expected_placeholder>>);
    }
  }
};

using test_combinations =
    typename get_combinations<access_modes_pack, all_dimensions_pack,
                              targets_pack>::type;

/**
 * @brief Struct with functor that will be used in type coverage to call all
 *        verification functions.
 * @tparam T Current data type
 * @tparam ArgCombination A tuple containing the packs representing the current
 *         test configuration. The packs appear in the following order:
 *         access_mode, dimension, target
 * @param type_name Current data type string representation
 */
template <typename T, typename ArgCombination>
class run_tests {
 public:
  void operator()(const std::string& type_name) {
    // Get the packs from the test combination type.
    using AccessModePack = std::tuple_element_t<0, ArgCombination>;
    using DimensionsPack = std::tuple_element_t<1, ArgCombination>;
    using TargetsPack = std::tuple_element_t<2, ArgCombination>;

    // Type packs instances have to be const, otherwise for_all_combination
    // will not compile
    const auto access_modes = AccessModePack::generate_named();
    const auto dimensions = DimensionsPack::generate_unnamed();
    const auto targets = TargetsPack::generate_named();

    // Run test with non const data type
    // Run test for local_accessor
    verify_local_acc_dim_value<T>(type_name);
    // Run test for host_accessor
    verify_host_acc_dim_value<T>(type_name);
    for_all_combinations<test_for_host_acc_access_mode_val_verification, T>(
        dimensions, type_name);
    // Run test for generic accessor
    verify_generic_acc_dim_value<T>(type_name);
    for_all_combinations<test_for_generic_acc_acc_mode_val_verification, T>(
        dimensions, type_name);
    for_all_combinations<test_for_generic_acc_target_val_verification, T>(
        access_modes, dimensions, type_name);
    for_all_combinations<test_for_generic_acc_placeholder_val_verification, T>(
        access_modes, dimensions, targets, type_name);

    // Run test with const data type
    const auto const_type_name = "const " + type_name;
    using const_T = const T;
    // Run test for local_accessor
    verify_local_acc_dim_value<const_T>(type_name);
    // Run test for host_accessor
    verify_host_acc_dim_value<const_T>(type_name);
    for_all_combinations<test_for_host_acc_access_mode_val_verification,
                         const_T>(dimensions, type_name);
    // Run test for generic accessor
    verify_generic_acc_dim_value<const_T>(type_name);
    for_all_combinations<test_for_generic_acc_acc_mode_val_verification,
                         const_T>(dimensions, type_name);
    // const T can be only with access_mode::read
    const auto read_only_acc_mode =
        value_pack<sycl::access_mode, sycl::access_mode::read>::generate_named(
            "access_mode::read");
    for_all_combinations<test_for_generic_acc_target_val_verification, const_T>(
        read_only_acc_mode, dimensions, type_name);
    for_all_combinations<test_for_generic_acc_placeholder_val_verification,
                         const_T>(read_only_acc_mode, dimensions, targets,
                                  type_name);
  }
};

}  // namespace accessor_default_values_test

#endif  // SYCL_CTS_ACCESSOR_DEFAULT_VALUES_H
