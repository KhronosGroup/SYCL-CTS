/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides class to verify conformity with named requirement
//  LegacyForwardIterator
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_FORWARD_ITERATOR_H
#define __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_FORWARD_ITERATOR_H

#include "common.h"
#include "legacy_input_iterator.h"
#include "legacy_output_iterator.h"

namespace named_requirement_verification {

/**
 * @brief Class helps to verify conformity to LegacyForwardIterator named
 * requirement. Safe to use inside kernel with SYCL 2020.
 *
 */
class legacy_forward_iterator_requirement {
 public:
  // Will be used as size of container for error messages.
  // Value should be equal to the number of add_error invocations.
  // Don't forget to update this value if there is any changes in class.
  // As we also verify other requirements, we have to keep in mind that result
  // of those verifications also increase size of container with messages.
  static constexpr size_t count_of_possible_errors =
      legacy_input_iterator_requirement::count_of_possible_errors +
      legacy_output_iterator_requirement::count_of_possible_errors + 10;

 private:
  error_messages_container<count_of_possible_errors> m_test_error_messages;

 public:
  /**
   * @brief Member function preforms different checks for the requirement
   * verification
   *
   * @tparam It Type of iterator for verification
   * @return std::pair<bool,array<string_view>> First represents
   * satisfaction of the requirement. Second contains error messages
   */
  template <typename It>
  std::pair<bool, std::array<string_view, count_of_possible_errors>>
  is_satisfied_for(It valid_iterator, const size_t container_size) {
    auto legacy_input_iterator_res =
        legacy_input_iterator_requirement{}.is_satisfied_for<It>();
    if (!legacy_input_iterator_res.first) {
      m_test_error_messages.add_errors(legacy_input_iterator_res.second);
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
    constexpr bool is_def_constructable = std::is_default_constructible_v<It>;

    if (!is_def_constructable) {
      m_test_error_messages.add_error(
          "Iterator should be default constructible.");
    }

    using it_traits = std::iterator_traits<It>;

    // Allows us to use reference_t from iterator_traits
    if constexpr (has_reference_member) {
      const bool is_output_iterator_req_satisfied =
          legacy_output_iterator_requirement{}.is_satisfied_for<It>().first;

      if (is_output_iterator_req_satisfied) {
        if (std::is_const_v<typename it_traits::reference>) {
          m_test_error_messages.add_error(
              "Provided iterator satisfy to LegacyOutputIterator requirement. "
              "iterator_traits::reference should be non const.");
        }
      } else {
        if (!std::is_const_v<typename it_traits::reference>) {
          m_test_error_messages.add_error(
              "Provided iterator not satisfy to LegacyOutputIterator "
              "requirement. iterator_traits::reference should be const.");
        }
      }
    }

    if constexpr (can_post_increment) {
      if (!std::is_same_v<decltype(std::declval<It&>()++), It>) {
        m_test_error_messages.add_error("operator++(int) should return It.");
      }
    } else {
      m_errors.add_error("Iterator doesn't have implemented operator++(int)");
    }

    if constexpr (can_post_increment && is_dereferenceable &&
                  has_reference_member) {
      if (!std::is_convertible_v<decltype(*(std::declval<It&>()++)),
                                 typename it_traits::reference>) {
        m_test_error_messages.add_error(
            "Expression *i++ should be convertible to "
            "iterator_traits::reference.");
      }
    }

    if (container_size == 0) {
      m_test_error_messages.add_error(
          "Some of the test requires container size more than 0. These tests "
          "have been skipped.");
    } else {
      // Verify multipass guarantee
      if constexpr (has_equal_operator && is_dereferenceable &&
                    can_pre_increment) {
        {
          It a = valid_iterator;
          It b = valid_iterator;
          if (*a != *b) {
            m_test_error_messages.add_error(
                "If a and b compare equal (a == b) then *a and *b "
                "are references bound to the same object.");
          }

          if (a != b) {
            m_test_error_messages.add_error(
                "If *a and *b refer to the same object, then a == b equals "
                "true.");
          }

          if (++a != ++b) {
            m_test_error_messages.add_error(
                "If a == b equals true then ++a == ++b also equals true.");
          }
        }

        if constexpr (has_value_type_member) {
          // Allows us to compare values without compilation error
          constexpr bool is_value_type_comparable =
              type_traits::has_comparison::is_equal_v<
                  typename it_traits::value_type>;

          if constexpr (is_dereferenceable && can_pre_increment &&
                        is_value_type_comparable) {
            const auto zero_pos_value = *valid_iterator;
            It zero_pos_it = valid_iterator;
            ++zero_pos_it;
            if (zero_pos_value != *valid_iterator) {
              m_test_error_messages.add_error(
                  "Incrementing copy of iterator instance should not affect "
                  "on the value read from original object.");
            }
          }
        }
      }
    }

    const bool is_satisfied = !m_test_error_messages.has_errors();
    // According to spec std::pair with device_copyable types(in this case:
    // bool, string_view) can be used on device side
    return std::make_pair(is_satisfied, m_test_error_messages.get_array());
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_LEGACY_FORWARD_ITERATOR_H
