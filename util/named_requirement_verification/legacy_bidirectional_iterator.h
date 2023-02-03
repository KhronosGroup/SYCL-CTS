/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides class to verify conformity with named requirement
//  LegacyBidirectionalIterator
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_BIDIRECTIONAL_ITERATOR_H
#define __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_BIDIRECTIONAL_ITERATOR_H

#include "common.h"
#include "legacy_forward_iterator.h"

namespace named_requirement_verification {

/**
 * @brief Class helps to verify conformity to LegacyBidirectionalIterator named
 * requirement. Safe to use inside kernel with SYCL 2020.
 *
 */
class legacy_bidirectional_iterator_requirement {
 public:
  // Will be used as size of container for error messages.
  // Value should be equal to the number of add_error invocations.
  // Don't forget to update this value if there is any changes in class.
  // As we also verify other requirements, we have to keep in mind that result
  // of those verifications also increase size of container with messages.
  static constexpr int count_of_possible_errors =
      legacy_forward_iterator_requirement::count_of_possible_errors + 9;

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
    auto legacy_forward_iterator_res =
        legacy_forward_iterator_requirement{}.is_satisfied_for<It>(
            valid_iterator);

    if (!legacy_forward_iterator_res.first) {
      m_test_error_codes.add_errors(legacy_forward_iterator_res.second);
    }

    using it_traits = std::iterator_traits<It>;

    constexpr bool is_dereferenceable = is_dereferenceable_v<It>;
    constexpr bool can_pre_increment =
        type_traits::has_arithmetic::pre_increment_v<It>;
    constexpr bool can_post_increment =
        type_traits::has_arithmetic::post_increment_v<It>;
    constexpr bool can_pre_decrement =
        type_traits::has_arithmetic::pre_decrement_v<It>;
    constexpr bool can_post_decrement =
        type_traits::has_arithmetic::post_decrement_v<It>;
    constexpr bool has_reference_member =
        type_traits::has_field::reference_v<It>;
    constexpr bool has_value_type_member =
        type_traits::has_field::value_type_v<It>;

    if constexpr (can_pre_increment && can_pre_decrement) {
      if (!std::is_same_v<decltype(--(++std::declval<It&>())), It&>) {
        m_test_error_codes.add_error(
            legacy_bidirectional_iterator::error_code_0);
      }
    }

    if (!can_post_decrement) {
      m_test_error_codes.add_error(legacy_bidirectional_iterator::error_code_1);
    }

    if constexpr (can_pre_decrement && can_pre_increment &&
                  is_dereferenceable) {
      {
        It a = valid_iterator;
        It saved_a = a;
        ++a;
        --a;
        if (a != saved_a) {
          m_test_error_codes.add_error(
              legacy_bidirectional_iterator::error_code_2);
        }
      }
      {
        It a = valid_iterator;
        It b = a;
        ++a;
        ++b;
        if (--a == --b) {
          if (a != b) {
            m_test_error_codes.add_error(
                legacy_bidirectional_iterator::error_code_3);
          }
        } else {
          m_test_error_codes.add_error(
              legacy_bidirectional_iterator::error_code_4);
        }
      }
    }

    if constexpr (can_pre_decrement) {
      if (!std::is_same_v<decltype(--(std::declval<It&>())), It&>) {
        m_test_error_codes.add_error(
            legacy_bidirectional_iterator::error_code_5);
      }
    } else {
      m_test_error_codes.add_error(legacy_bidirectional_iterator::error_code_6);
    }

    if constexpr (can_post_increment && can_post_decrement &&
                  has_value_type_member) {
      if (!std::is_convertible_v<decltype((++std::declval<It&>())--),
                                 const It&>) {
        m_test_error_codes.add_error(
            legacy_bidirectional_iterator::error_code_7);
      }
    }

    if constexpr (can_post_decrement && is_dereferenceable &&
                  has_reference_member) {
      if (!std::is_same_v<decltype(*(std::declval<It&>()--)),
                          typename it_traits::reference>) {
        m_test_error_codes.add_error(
            legacy_bidirectional_iterator::error_code_8);
      }
    }

    const bool is_satisfied = !m_test_error_codes.has_errors();
    // According to spec std::pair with device_copyable types(in this case:
    // bool, int) can be used on device side
    return std::make_pair(is_satisfied, m_test_error_codes.get_array());
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_BIDIRECTIONAL_ITERATOR_H
