/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides class to verify conformity with named requirement LegacyIterator
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_ITERATOR_H
#define __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_ITERATOR_H

#include "common.h"

namespace named_requirement_verification {

/**
 * @brief Class helps to verify conformity to LegacyIterator named
 * requirement. Safe to use inside kernel with SYCL 2020.
 *
 */
class legacy_iterator_requirement {
 public:
  // Will be used as size of container for error messages.
  // Value should be equal to the number of add_error invocations.
  static constexpr size_t count_of_possible_errors = 12;

 private:
  error_codes_container<count_of_possible_errors> m_test_error_codes;

 public:
  /**
   * @brief Member function performs different checks for the requirement
   * verification
   *
   * @tparam It Type of iterator for verification
   * @return std::pair<bool,array<int>> First represents
   * satisfaction of the requirement. Second contains error messages
   */
  template <typename It>
  std::pair<bool, std::array<int, count_of_possible_errors>>
  is_satisfied_for() {
    if (!std::is_copy_constructible_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_0);
    }

    if (!std::is_copy_assignable_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_1);
    }

    if (!std::is_destructible_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_2);
    }

    if (!std::is_swappable_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_3);
    }

    if (!type_traits::has_field::value_type_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_4);
    }

    if (!type_traits::has_field::difference_type_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_5);
    }

    if (!type_traits::has_field::reference_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_6);
    }

    if (!type_traits::has_field::pointer_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_7);
    }

    if (!type_traits::has_field::iterator_category_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_8);
    }

    if (!type_traits::has_arithmetic::pre_increment_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_9);
    }

    if constexpr (type_traits::has_arithmetic::pre_increment_v<It>) {
      if (!std::is_same_v<decltype(++std::declval<It&>()), It&>) {
        m_test_error_codes.add_error(legacy_iterator::error_code_10);
      }
    }

    if (!is_dereferenceable_v<It>) {
      m_test_error_codes.add_error(legacy_iterator::error_code_11);
    }

    const bool is_satisfied = !m_test_error_codes.has_errors();
    // According to spec std::pair with device_copyable types(in this case:
    // bool, int) can be used on device side
    return std::make_pair(is_satisfied, m_test_error_codes.get_array());
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_ITERATOR_H
