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
  error_messages_container<count_of_possible_errors> m_test_error_messages;

 public:
  /**
   * @brief Member function performs different checks for the requirement
   * verification
   *
   * @tparam It Type of iterator for verification
   * @return std::pair<bool,array<string_view>> First represents
   * satisfaction of the requirement. Second contains error messages
   */
  template <typename It>
  std::pair<bool, std::array<string_view, count_of_possible_errors>>
  is_satisfied_for() {
    if (!std::is_copy_constructible_v<It>) {
      m_test_error_messages.add_error("Iterator must be copy constructable.");
    }

    if (!std::is_copy_assignable_v<It>) {
      m_test_error_messages.add_error("Iterator must be copy assignable.");
    }

    if (!std::is_destructible_v<It>) {
      m_test_error_messages.add_error("Iterator must be destructible.");
    }

    if (!std::is_swappable_v<It>) {
      m_test_error_messages.add_error("Iterator must be swappable.");
    }

    if (!type_traits::has_field::value_type_v<It>) {
      m_test_error_messages.add_error(
          "Iterator must have value_type member typedef.");
    }

    if (!type_traits::has_field::difference_type_v<It>) {
      m_test_error_messages.add_error(
          "Iterator must have difference_type member typedef.");
    }

    if (!type_traits::has_field::reference_v<It>) {
      m_test_error_messages.add_error(
          "Iterator must have reference member typedef.");
    }

    if (!type_traits::has_field::pointer_v<It>) {
      m_test_error_messages.add_error(
          "Iterator must have pointer member typedef.");
    }

    if (!type_traits::has_field::iterator_category_v<It>) {
      m_test_error_messages.add_error(
          "Iterator must have iterator_category member typedef.");
    }

    if (!type_traits::has_arithmetic::pre_increment_v<It>) {
      m_test_error_messages.add_error("Iterator must have operator++().");
    }

    if constexpr (type_traits::has_arithmetic::pre_increment_v<It>) {
      if (!std::is_same_v<decltype(++std::declval<It&>()), It&>) {
        m_test_error_messages.add_error(
            "Iterator must return It& after usage of operator++().");
      }
    }

    if (!is_dereferenceable_v<It>) {
      m_test_error_messages.add_error("Iterator must have operator*().");
    }

    const bool is_satisfied = !m_test_error_messages.has_errors();
    // According to spec std::pair with device_copyable types(in this case:
    // bool, string_view) can be used on device side
    return std::make_pair(is_satisfied, m_test_error_messages.get_array());
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_ITERATOR_H
