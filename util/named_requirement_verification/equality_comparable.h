/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides class to verify conformity with named requirement
//  EqualityComparable
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_EQUALITY_COMPARABLE_H
#define __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_EQUALITY_COMPARABLE_H

#include "common.h"

namespace named_requirement_verification {

/**
 * @brief Class helps to verify conformity to EqualityComparable named
 * requirement. Safe to use inside kernel with SYCL 2020.
 *
 */
class equality_comparable_requirement {
 public:
  // Will be used as size of container for error messages
  // Value should be equal to the number of add_error invocations
  // Don't forget to update this value if there is any changes in class
  static constexpr size_t count_of_possible_errors = 5;

 private:
  error_messages_container<count_of_possible_errors> m_errors;

 public:
  /**
   * @brief Member function preform different checks for the requirement
   * verification
   *
   * @tparam T Type for verification
   * @return std::pair<bool,array<string_view>> First represents
   * satisfaction of the requirement. Second contains error messages
   */
  template <typename T>
  auto is_satisfied_for() {
    if (!type_traits::has_comparison::is_equal_v<T>) {
      m_errors.add_error("Doesn't have implemented operator==()");
    }

    // It will delete branch from code in compile time to not fail a compilation
    if constexpr (type_traits::has_comparison::is_equal_v<T>) {
      T a;
      T b;
      T c;
      const T const_a;
      const T const_b;
      const T const_c;

      if (!(a == b) || !(b == a) || !(b == c) || !(a == c)) {
        m_errors.add_error(
            "Non-const copies of one object doesn't equal to each other equal "
            "during comparing");
      }

      if ((!std::is_convertible_v<decltype((a == b)), bool>) ||
          (!std::is_convertible_v<decltype((b == a)), bool>) ||
          (!std::is_convertible_v<decltype((b == c)), bool>) ||
          (!std::is_convertible_v<decltype((a == c)), bool>)) {
        m_errors.add_error(
            "Non-const copies of one object doesn't return convertible to bool "
            "value after comparing");
      }

      if (!(const_a == const_b) || !(const_b == const_a) ||
          !(const_b == const_c) || !(const_a == const_c)) {
        m_errors.add_error(
            "Const copies of one object doesn't equal to each other equal "
            "during comparing");
      }

      if ((!std::is_convertible_v<decltype((const_a == const_b)), bool>) ||
          (!std::is_convertible_v<decltype((const_b == const_c)), bool>) ||
          (!std::is_convertible_v<decltype((const_a == const_c)), bool>) ||
          (!std::is_convertible_v<decltype((const_b == const_a)), bool>)) {
        m_errors.add_error(
            "Const copies of one object doesn't return convertible to bool "
            "value after comparing");
      }
    }

    const bool is_satisfied = !m_errors.has_errors();
    return std::make_pair(is_satisfied, m_errors.get_array());
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_EQUALITY_COMPARABLE_H
