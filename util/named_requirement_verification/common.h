/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common code for iterator requirements verifications
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_COMMON_H
#define __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_COMMON_H

#include "../type_traits.h"

#include <array>
#include <map>
#include <string_view>
#include <utility>

namespace named_requirement_verification {

namespace common {
static const int error_code_0 = -1;
static const int error_code_1 = -2;
}  // namespace common

namespace equality_comparable {
static const int error_code_0 = 1;
static const int error_code_1 = 2;
static const int error_code_2 = 3;
static const int error_code_3 = 4;
static const int error_code_4 = 5;
}  // namespace equality_comparable

namespace legacy_bidirectional_iterator {
static const int error_code_0 = 6;
static const int error_code_1 = 7;
static const int error_code_2 = 8;
static const int error_code_3 = 9;
static const int error_code_4 = 10;
static const int error_code_5 = 11;
static const int error_code_6 = 12;
static const int error_code_7 = 13;
static const int error_code_8 = 14;
}  // namespace legacy_bidirectional_iterator

namespace legacy_forward_iterator {
static const int error_code_0 = 15;
static const int error_code_1 = 16;
static const int error_code_2 = 17;
static const int error_code_3 = 18;
static const int error_code_4 = 19;
static const int error_code_5 = 20;
static const int error_code_6 = 21;
static const int error_code_7 = 22;
static const int error_code_8 = 23;
static const int error_code_9 = 24;
}  // namespace legacy_forward_iterator

namespace legacy_input_iterator {
static const int error_code_0 = 25;
static const int error_code_1 = 26;
static const int error_code_2 = 27;
static const int error_code_3 = 28;
static const int error_code_4 = 29;
static const int error_code_5 = 30;
static const int error_code_6 = 31;
static const int error_code_7 = 32;
static const int error_code_8 = 33;
static const int error_code_9 = 34;
static const int error_code_10 = 35;
}  // namespace legacy_input_iterator

namespace legacy_iterator {
static const int error_code_0 = 36;
static const int error_code_1 = 37;
static const int error_code_2 = 38;
static const int error_code_3 = 39;
static const int error_code_4 = 40;
static const int error_code_5 = 41;
static const int error_code_6 = 42;
static const int error_code_7 = 43;
static const int error_code_8 = 44;
static const int error_code_9 = 45;
static const int error_code_10 = 46;
static const int error_code_11 = 47;
}  // namespace legacy_iterator

namespace legacy_output_iterator {
static const int error_code_0 = 48;
static const int error_code_1 = 49;
static const int error_code_2 = 50;
static const int error_code_3 = 51;
static const int error_code_4 = 52;
static const int error_code_5 = 53;
static const int error_code_6 = 54;
static const int error_code_7 = 55;
}  // namespace legacy_output_iterator

namespace legacy_random_access_iterator {
static const int error_code_0 = 56;
static const int error_code_1 = 57;
static const int error_code_2 = 58;
static const int error_code_3 = 59;
static const int error_code_4 = 60;
static const int error_code_5 = 61;
static const int error_code_6 = 62;
static const int error_code_7 = 63;
static const int error_code_8 = 64;
static const int error_code_9 = 65;
static const int error_code_10 = 66;
static const int error_code_11 = 67;
static const int error_code_12 = 68;
static const int error_code_13 = 69;
static const int error_code_14 = 70;
static const int error_code_15 = 71;
static const int error_code_16 = 72;
static const int error_code_17 = 73;
static const int error_code_18 = 74;
static const int error_code_19 = 75;
static const int error_code_20 = 76;
static const int error_code_21 = 77;
}  // namespace legacy_random_access_iterator

using string_view = std::basic_string_view<char>;

class error_messages {
 public:
  error_messages() { init(); }

  const string_view& error_message(int error_code) const {
    auto msg_it = m_error_map.find(error_code);
    if (msg_it == m_error_map.end()) {
      return m_error_map.at(common::error_code_1);
    }
    return msg_it->second;
  }

 private:
  std::map<int, string_view> m_error_map;

  void init() {
    init_common();
    init_equality_comparable();
    init_legacy_bidirectional_iterator();
    init_legacy_forward_iterator();
    init_legacy_input_iterator();
    init_legacy_iterator();
    init_legacy_output_iterator();
    init_legacy_random_access_iterator();
  }

  void init_common() {
    m_error_map.insert(
        {{common::error_code_0, "Size for error_codes_container setted wrong!"},
         {common::error_code_1, "Wrong error code!"}});
  }
  void init_equality_comparable() {
    m_error_map.insert({
        {equality_comparable::error_code_0,
         "Non-const copies of one object doesn't equal to each other equal "
         "during comparing."},
        {equality_comparable::error_code_1,
         "Non-const copies of one object doesn't return convertible to bool "
         "after comparing."},
        {equality_comparable::error_code_2,
         "Const copies of one object doesn't equal to each other equal "
         "during comparing."},
        {equality_comparable::error_code_3,
         "Const copies of one object doesn't return convertible to bool "
         "after comparing."},
        {equality_comparable::error_code_4, "Iterator must have operator==()."},
    });
  }

  void init_legacy_bidirectional_iterator() {
    m_error_map.insert({
        {legacy_bidirectional_iterator::error_code_0,
         "Iterator expression --(++i) must return It& type."},
        {legacy_bidirectional_iterator::error_code_1,
         "Iterator doesn't have implemented operator--(int)"},
        {legacy_bidirectional_iterator::error_code_2,
         "Iterator expression --(++i) must be equal to i."},
        {legacy_bidirectional_iterator::error_code_3,
         "If --a == --b then a == b must be true."},
        {legacy_bidirectional_iterator::error_code_4,
         "--a must be equal to --b, if they are copy of same object."},
        {legacy_bidirectional_iterator::error_code_5,
         "Iterator expression --i must return It& data type."},
        {legacy_bidirectional_iterator::error_code_6,
         "Iterator must have operator--()."},
        {legacy_bidirectional_iterator::error_code_7,
         "Iterator expression (i++)-- must be convertible to const It&."},
        {legacy_bidirectional_iterator::error_code_8,
         "Iterator expression *i-- must return reference data type."},
    });
  }

  void init_legacy_forward_iterator() {
    m_error_map.insert({
        {legacy_forward_iterator::error_code_0,
         "Iterator must be default constructible."},
        {legacy_forward_iterator::error_code_1,
         "Provided iterator satisfy to LegacyOutputIterator requirement. "
         "iterator_traits::reference must be T& or T&&."},
        {legacy_forward_iterator::error_code_2,
         "Provided iterator not satisfy to LegacyOutputIterator "
         "requirement. iterator_traits::reference must be const T& or "
         "const T&&."},
        {legacy_forward_iterator::error_code_3,
         "operator++(int) must return It."},
        {legacy_forward_iterator::error_code_4,
         "Iterator doesn't have implemented operator++(int)"},
        {legacy_forward_iterator::error_code_5,
         "Expression *i++ must be convertible to "
         "iterator_traits::reference."},
        {legacy_forward_iterator::error_code_6,
         "If a and b compare equal (a == b) then *a and *b "
         "are references bound to the same object."},
        {legacy_forward_iterator::error_code_7,
         "If *a and *b refer to the same object, then a == b equals "
         "true."},
        {legacy_forward_iterator::error_code_8,
         "If a == b equals true then ++a == ++b also equals true."},
        {legacy_forward_iterator::error_code_9,
         "Incrementing copy of iterator instance must not affect "
         "on the value read from original object."},
    });
  }

  void init_legacy_input_iterator() {
    m_error_map.insert({
        {legacy_input_iterator::error_code_0,
         "Iterator must have implemented operator==()."},
        {legacy_input_iterator::error_code_1,
         "Iterator must have implemented operator!=()."},
        {legacy_input_iterator::error_code_2,
         "Iterator must have implemented operator++(int)."},
        {legacy_input_iterator::error_code_3,
         "Two not equal iterators returns true with NOT EQUAL operator."},
        {legacy_input_iterator::error_code_4,
         "Two not equal iterators must return implicit convertible to "
         "bool value with NOT EQUAL operator."},
        {legacy_input_iterator::error_code_5,
         "Iterator must return It& from operator++()."},
        {legacy_input_iterator::error_code_6,
         "Iterator expression *i++ must be convertible to "
         "iterator_traits::value_type."},
        {legacy_input_iterator::error_code_7,
         "Iterator must return iterator_traits::reference from "
         "operator*()."},
        {legacy_input_iterator::error_code_8,
         "operator*() result must be convertible to "
         "iterator_traits::value_type."},
        {legacy_input_iterator::error_code_9, ""},
        {legacy_input_iterator::error_code_10, ""},
    });
  }

  void init_legacy_iterator() {
    m_error_map.insert({
        {legacy_iterator::error_code_0, "Iterator must be copy constructable."},
        {legacy_iterator::error_code_1, "Iterator must be copy assignable."},
        {legacy_iterator::error_code_2, "Iterator must be destructible."},
        {legacy_iterator::error_code_3, "Iterator must be swappable."},
        {legacy_iterator::error_code_4,
         "Iterator must have value_type member typedef."},
        {legacy_iterator::error_code_5,
         "Iterator must have difference_type member typedef."},
        {legacy_iterator::error_code_6,
         "Iterator must have reference member typedef."},
        {legacy_iterator::error_code_7,
         "Iterator must have pointer member typedef."},
        {legacy_iterator::error_code_8,
         "Iterator must have iterator_category member typedef."},
        {legacy_iterator::error_code_9, "Iterator must have operator++()."},
        {legacy_iterator::error_code_10,
         "Iterator must return It& after usage of operator++()."},
        {legacy_iterator::error_code_11, "Iterator must have operator*()."},
    });
  }

  void init_legacy_output_iterator() {
    m_error_map.insert({
        {legacy_output_iterator::error_code_0,
         "Iterator must return iterator_traits::value_type from "
         "operator*()."},
        {legacy_output_iterator::error_code_1,
         "Iterator must return It& from operator++()."},
        {legacy_output_iterator::error_code_2,
         "Iterator must return convertible to const It from "
         "operator++()."},
        {legacy_output_iterator::error_code_3,
         "Iterator must be assignable with iterator_traits::value_type "
         "after usage of operator++() and operator*()."},
        {legacy_output_iterator::error_code_4,
         "Iterator must be assignable with iterator_traits::value_type "
         "after usage of operator*()."},
        {legacy_output_iterator::error_code_5, ""},
        {legacy_output_iterator::error_code_6, ""},
        {legacy_output_iterator::error_code_7, ""},
    });
  }

  void init_legacy_random_access_iterator() {
    m_error_map.insert({
        {legacy_random_access_iterator::error_code_0,
         "Iterator must have difference_type member typedef."},
        {legacy_random_access_iterator::error_code_1,
         "Iterator must have operator+=() between Iterator instance and "
         "iterator_traits::difference_type."},
        {legacy_random_access_iterator::error_code_2,
         "Iterator must have operator-=() between Iterator instance and "
         "iterator_traits::difference_type."},
        {legacy_random_access_iterator::error_code_3,
         "Iterator must have subtraction between iterators."},
        {legacy_random_access_iterator::error_code_4,
         "Iterator must have operator+() iterator_traits::difference_type "
         "plus Iterator object with operator."},
        {legacy_random_access_iterator::error_code_5,
         "Iterator must have operator[]."},
        {legacy_random_access_iterator::error_code_6,
         "operator+=() and operator-=() must return It& for positive "
         "and negative vales."},
        {legacy_random_access_iterator::error_code_7,
         "operator+() and operator-() must return It& for positive "
         "and negative vales."},
        {legacy_random_access_iterator::error_code_8,
         "Iterator operator+=() must be commutative."},
        {legacy_random_access_iterator::error_code_9,
         "Iterator must have operator+() with "
         "iterator_traits::difference_type operator."},
        {legacy_random_access_iterator::error_code_10,
         "Iterator object minus iterator_traits::difference_type must "
         "return Iterator instance."},
        {legacy_random_access_iterator::error_code_11,
         "Iterator must have operator-() with "
         "iterator_traits::difference_type operator."},
        {legacy_random_access_iterator::error_code_12,
         "operator-() of It instances must return "
         "iterator_traits::difference_type."},
        {legacy_random_access_iterator::error_code_13,
         "operator[]() return value must be convertible to "
         "iterator_traits::reference."},
        {legacy_random_access_iterator::error_code_14,
         "operator>() return value must be contextually convertible to "
         "bool."},
        {legacy_random_access_iterator::error_code_15,
         "operator<() return value must be contextually convertible to "
         "bool."},
        {legacy_random_access_iterator::error_code_16,
         "operator>=() return value must be contextually convertible to "
         "bool."},
        {legacy_random_access_iterator::error_code_17,
         "Iterator must have operator>=()."},
        {legacy_random_access_iterator::error_code_18,
         "operator>=() return value must be contextually convertible to "
         "bool."},
        {legacy_random_access_iterator::error_code_19,
         "Iterator must have operator>=()."},
        {legacy_random_access_iterator::error_code_20,
         "operator<=() return value must be contextually convertible to "
         "bool."},
        {legacy_random_access_iterator::error_code_21,
         "Iterator must have operator<=()."},
    });
  }
};

/**
 * @brief Class for storing error codes during requirements verification.
 * Safe to use inside kernel with SYCL 2020.
 *
 * @tparam Size parameter used for allocation of std::array with int
 */
template <size_t Size>
class error_codes_container {
 private:
  /* Array for storing error codes
   * We can't use std::string since it's not supported in kernel
   *
   * According to SYCL 2020 std::array and int are
   * supported on device side
   */
  std::array<int, Size> m_error_codes_container;
  // Index for storing next error message
  size_t m_index = 0;

 public:
  error_codes_container() { m_error_codes_container.fill(0); }
  /**
   * @brief Member function helps to add single error message
   */
  void add_error(int error_code) {
    if (m_index < Size) {
      m_error_codes_container[m_index] = error_code;
      ++m_index;
    } else {
      // If Size of container doesn't fit
      m_error_codes_container[Size - 1] = common::error_code_0;
    }
  }

  /**
   * @brief Member function helps to add array with errors messages
   */
  template <size_t N>
  void add_errors(const std::array<int, N> error_codes) {
    for (size_t i = 0; i < error_codes.size(); ++i) {
      // Avoid copying of empty messages
      if (0 != error_codes[i]) {
        add_error(error_codes[i]);
      }
    }
  }

  bool has_errors() const { return m_index != 0; }

  const std::array<int, Size>& get_array() const {
    return m_error_codes_container;
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_COMMON_H
