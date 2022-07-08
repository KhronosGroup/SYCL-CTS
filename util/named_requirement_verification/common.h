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
#include <string_view>
#include <utility>

namespace named_requirement_verification {

using string_view = std::basic_string_view<char>;

/**
 * @brief Class for storing error messages during requirements verification.
 * Safe to use inside kernel with SYCL 2020.
 *
 * @tparam Size parameter used for allocation of std::array with
 * std::basic_string
 */
template <size_t Size>
class error_messages_container {
 private:
  /* Array for storing error messages
   * We can't use std::string since it's not supported in kernel
   *
   * According to SYCL 2020 std::array and std::basic_string_view<char> are
   * supported on device side
   */
  std::array<string_view, Size> m_error_msgs_container;
  // Index for storing next error message
  size_t m_index = 0;
  // Variable will be set to true if add_error was called
  bool m_has_errors = false;

 public:
  /**
   * @brief Member function helps to add single error message
   */
  void add_error(const string_view msg) {
    if (m_index < Size) {
      m_error_msgs_container[m_index] = msg;
      ++m_index;
    } else {
      // If Size of container doesn't fit
      m_error_msgs_container[Size - 1] =
          "Size for error_messages_container setted wrong!";
    }
    m_has_errors = true;
  }

  /**
   * @brief Member function helps to add array with errors messages
   */
  template <size_t N>
  void add_errors(const std::array<string_view, N> msgs) {
    for (size_t i = 0; i < msgs.size(); ++i) {
      // Avoid copying of empty messages
      if (!msgs[i].empty()) {
        add_error(msgs[i]);
      }
    }
  }

  bool has_errors() const { return m_has_errors; }

  const std::array<string_view, Size>& get_array() const {
    return m_error_msgs_container;
  }
};
}  // namespace named_requirement_verification

#endif  // __SYCLCTS_TESTS_ITERATOR_REQUIREMENTS_COMMON_H
