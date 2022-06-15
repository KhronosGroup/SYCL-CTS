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
  // Will be used as size of container for error messages
  // Value should be equal to the number of add_error invocations
  // Don't forget to update this value if there is any changes in class
  // As we also verify other requirements, we have to keep in mind that result
  // of those verifications also increase size of container with messages
  static constexpr int count_of_possible_errors =
      legacy_forward_iterator_requirement::count_of_possible_errors + 8;

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
  auto is_satisfied_for(It valid_iterator, const size_t container_size) {
    auto legacy_forward_iterator_res =
        legacy_forward_iterator_requirement{}.is_satisfied_for<It>(
            valid_iterator, container_size);

    if (legacy_forward_iterator_res.first == false) {
      m_errors.add_errors(legacy_forward_iterator_res.second);
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
      if (std::is_same_v<decltype(--(++std::declval<It&>())), It&> == false) {
        m_errors.add_error(
            "Iterator expression --(++i) have to return It& type");
      }
    }

    if (container_size == 0) {
      m_errors.add_error(
          "Some of the test requires container size more than 0. These tests "
          "have been skipped");
    } else {
      if constexpr (can_pre_decrement && can_pre_increment &&
                    is_dereferenceable) {
        {
          It a = valid_iterator;
          It saved_a = a;
          ++a;
          --a;
          if ((a == saved_a) == false) {
            m_errors.add_error(
                "Iterator expression --(++i) have to be equal to i");
          }
        }
        {
          It a = valid_iterator;
          It b = a;
          ++a;
          ++b;
          if (--a == --b) {
            if ((a == b) == false) {
              m_errors.add_error("If --a == --b then a == b have to be true");
            }
          } else {
            m_errors.add_error(
                "--a have to be equal to --b, if they are copy of same object");
          }
        }
      }
    }

    if constexpr (can_pre_decrement) {
      if (std::is_same_v<decltype(--(std::declval<It&>())), It&> == false) {
        m_errors.add_error(
            "Iterator expression --i have to return It& data type");
      }
    }

    if constexpr (can_post_increment && can_post_decrement &&
                  has_value_type_member) {
      if (std::is_convertible_v<decltype((++std::declval<It&>())--),
                                const It&> == false) {
        m_errors.add_error(
            "Iterator expression (i++)-- have to be convertible to const It&");
      }
    }

    if constexpr (can_post_decrement && is_dereferenceable &&
                  has_reference_member) {
      if (std::is_same_v<decltype(*(std::declval<It&>()--)),
                         typename it_traits::reference> == false) {
        m_errors.add_error(
            "Iterator expression *i-- have to return reference data type");
      }
    }

    const bool is_satisfied = !m_errors.has_errors();
    // According to spec std::pair with device_copyable types(in this case:
    // bool, string_view) can be used on device side
    return std::make_pair(is_satisfied, m_errors.get_array());
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_BIDIRECTIONAL_ITERATOR_H
