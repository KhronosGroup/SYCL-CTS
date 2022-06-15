/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides class to verify conformity with named requirement
//  LegacyOutputIterator
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_OUTPUT_ITERATOR_H
#define __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_OUTPUT_ITERATOR_H

#include "common.h"
#include "legacy_iterator.h"

namespace named_requirement_verification {

/**
 * @brief Class helps to verify conformity to LegacyOutputIterator named
 * requirement. Safe to use inside kernel with SYCL 2020.
 *
 */
class legacy_output_iterator_requirement {
 public:
  // Will be used as size of container for error messages
  // Value should be equal to the number of add_error invocations
  // Don't forget to update this value if there is any changes in class
  // As we also verify other requirements, we have to keep in mind that result
  // of those verifications also increase size of container with messages
  static constexpr size_t count_of_possible_errors =
      legacy_iterator_requirement::count_of_possible_errors + 8;

 private:
  error_messages_container<count_of_possible_errors> m_errors;

 public:
  /**
   * @brief Member function preform different checks for the requirement
   * verification
   *
   * @tparam It Type of iterator for verification
   * @return std::pair<bool,array<string_view>> First represents
   * satisfaction of the requirement. Second contains error messages
   */
  template <typename It>
  auto is_satisfied_for() {
    auto legacy_iterator_res =
        legacy_iterator_requirement{}.is_satisfied_for<It>();
    if (legacy_iterator_res.first == false) {
      m_errors.add_errors(legacy_iterator_res.second);
    }

    constexpr bool is_dereferenceable = is_dereferenceable_v<It>;
    constexpr bool can_pre_increment =
        type_traits::has_arithmetic::pre_increment_v<It>;
    constexpr bool can_post_increment =
        type_traits::has_arithmetic::post_increment_v<It>;
    constexpr bool has_value_type_member =
        type_traits::has_field::value_type_v<It>;

    if (!is_dereferenceable) {
      m_errors.add_error("Iterator have to implement operator*()");
    }

    if (!can_pre_increment || !can_post_increment) {
      m_errors.add_error(
          "Iterator have to implement operator++() and operator++(int)");
    }

    if (!has_value_type_member) {
      m_errors.add_error(
          "Iterator have to implement iterator_traits::value_type");
    }

    using it_traits = std::iterator_traits<It>;

    if constexpr (has_value_type_member && is_dereferenceable) {
      if (std::is_assignable_v<decltype(*std::declval<It>()),
                               typename it_traits::value_type> == false)
        m_errors.add_error(
            "Iterator have to return iterator_traits::value_type from "
            "operator*()");
    }

    if constexpr (can_pre_increment) {
      if (std::is_same_v<decltype(++std::declval<It&>()), It&> == false) {
        m_errors.add_error("Iterator have to return It& from operator++()");
      }
      if (std::is_convertible_v<decltype(++std::declval<It&>()), const It> ==
          false) {
        m_errors.add_error(
            "Iterator have to return convertble to const It from operator++()");
      }
    }

    if constexpr (can_post_increment && is_dereferenceable &&
                  has_value_type_member) {
      if (std::is_assignable_v<decltype(*(std::declval<It&>()++)),
                               typename it_traits::value_type> == false) {
        m_errors.add_error(
            "Iterator have to be assignable with iterator_traits::value_type "
            "after useage of operator++() and operator*()");
      }
    }

    if constexpr (is_dereferenceable && has_value_type_member) {
      if (std::is_assignable_v<decltype(*std::declval<It>()),
                               typename it_traits::value_type> == false) {
        m_errors.add_error(
            "Iterator have to be assignable with iterator_traits::value_type "
            "after useage of operator*()");
      }
    }

    const bool is_satisfied = !m_errors.has_errors();
    // According to spec std::pair with device_copyable types(in this case:
    // bool, string_view) can be used on device side
    return std::make_pair(is_satisfied, m_errors.get_array());
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_OUTPUT_ITERATOR_H
