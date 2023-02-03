/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides class to verify conformity with named requirement
//  LegacyInputIterator
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_INPUT_ITERATOR_H
#define __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_INPUT_ITERATOR_H

#include "common.h"
#include "equality_comparable.h"
#include "legacy_iterator.h"

namespace named_requirement_verification {

/**
 * @brief Class helps to verify conformity to LegacyInputIterator named
 * requirement. Safe to use inside kernel with SYCL 2020.
 *
 */
class legacy_input_iterator_requirement {
 public:
  // Will be used as size of container for error messages.
  // Value should be equal to the number of add_error invocations.
  // Don't forget to update this value if there is any changes in class.
  // As we also verify other requirements, we have to keep in mind that result
  // of those verifications also increase size of container with messages.
  static constexpr size_t count_of_possible_errors =
      legacy_iterator_requirement::count_of_possible_errors +
      equality_comparable_requirement::count_of_possible_errors + 11;

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
  std::pair<bool, std::array<int, count_of_possible_errors>> is_satisfied_for(
      It valid_iterator) {
    auto legacy_iterator_res =
        legacy_iterator_requirement{}.is_satisfied_for<It>();
    if (!legacy_iterator_res.first) {
      m_test_error_codes.add_errors(legacy_iterator_res.second);
    }

    auto equal_comparable_res =
        equality_comparable_requirement{}.is_satisfied_for<It>(valid_iterator);
    if (!legacy_iterator_res.first) {
      m_test_error_codes.add_errors(legacy_iterator_res.second);
    }

    constexpr bool is_dereferenceable = is_dereferenceable_v<It>;
    constexpr bool can_pre_increment =
        type_traits::has_arithmetic::pre_increment_v<It>;
    constexpr bool can_post_increment =
        type_traits::has_arithmetic::post_increment_v<It>;
    constexpr bool has_reference_member =
        type_traits::has_field::reference_v<It>;
    constexpr bool has_value_type_member =
        type_traits::has_field::value_type_v<It>;
    constexpr bool has_equal_operator =
        type_traits::has_comparison::is_equal_v<It>;
    constexpr bool has_not_equal_operator =
        type_traits::has_comparison::not_equal_v<It>;

    if (!has_equal_operator) {
      m_test_error_codes.add_error(legacy_input_iterator::error_code_0);
    }

    if (!has_not_equal_operator) {
      m_test_error_codes.add_error(legacy_input_iterator::error_code_1);
    }

    if (!can_post_increment) {
      m_test_error_codes.add_error(legacy_input_iterator::error_code_2);
    }

    if constexpr (can_pre_increment && has_not_equal_operator) {
      It j{};
      It i{};
      // As "Legacy Iterator" implements increment operator we can get two not
      // equal iterators
      ++i;
      if (i == j) {
        m_test_error_codes.add_error(legacy_input_iterator::error_code_3);
      }

      if (!std::is_convertible_v<decltype((i != j)), bool>) {
        m_test_error_codes.add_error(legacy_input_iterator::error_code_4);
      }
    }

    if constexpr (can_pre_increment) {
      if (!std::is_same_v<decltype(++std::declval<It&>()), It&>) {
        m_test_error_codes.add_error(legacy_input_iterator::error_code_5);
      }
    }

    using it_traits = std::iterator_traits<It>;

    if constexpr (is_dereferenceable && can_post_increment &&
                  has_value_type_member) {
      if (!std::is_convertible_v<decltype(*(std::declval<It&>()++)),
                                 typename it_traits::value_type>) {
        m_test_error_codes.add_error(legacy_input_iterator::error_code_6);
      }
    }

    if constexpr (is_dereferenceable && has_reference_member) {
      if (!std::is_same_v<decltype(*std::declval<It>()),
                          typename it_traits::reference>) {
        m_test_error_codes.add_error(legacy_input_iterator::error_code_7);
      }
    }

    if constexpr (is_dereferenceable && has_value_type_member) {
      if (!std::is_convertible_v<decltype(*std::declval<It>()),
                                 typename it_traits::value_type>)
        m_test_error_codes.add_error(legacy_input_iterator::error_code_8);
    }

    const bool is_satisfied = !m_test_error_codes.has_errors();
    // According to spec std::pair with device_copyable types(in this case:
    // bool, int) can be used on device side
    return std::make_pair(is_satisfied, m_test_error_codes.get_array());
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_INPUT_ITERATOR_H
