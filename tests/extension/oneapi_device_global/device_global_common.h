/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common functions for device_global tests
//
*******************************************************************************/
#ifndef SYCL_CTS_TEST_DEVICE_GLOBAL_COMMON_H
#define SYCL_CTS_TEST_DEVICE_GLOBAL_COMMON_H

#include "../../common/get_cts_string.h"

namespace device_global_common_functions {

/** @brief Enum for kernel names in tests
 */
enum class test_names : size_t {
  get_multi_ptr_method,
  implicit_conversation,
  get_method,
  subscript_operator,
  arrow_operator,
};

/** @brief Generate test description string
 *  @tparam T variable type
 *  @tparam Decorated parameter
 *  @retval String with description
 */
template <typename T, sycl::access::decorated Decorated>
inline std::string get_case_description(const std::string& test_name,
                                        const std::string& info,
                                        const std::string& type_name) {
  std::string decorated{sycl_cts::get_cts_string::for_decorated<Decorated>()};
  std::string message;
  message += test_name + " error: " + info;
  message += " with tparams:";
  message += "<" + type_name + ">";
  message += "<" + decorated + ">";
  return message;
}

/** @brief Generate test description string
 *  @tparam T variable type
 *  @retval String with description
 */
template <typename T>
inline std::string get_case_description(const std::string& test_name,
                                        const std::string& info,
                                        const std::string& type_name) {
  std::string message;
  message += test_name + " error: " + info;
  message += " with tparams:";
  message += "<" + type_name + ">";
  return message;
}

/** @brief The function helps to change and compare non-array values
 *  @tparam T type of value
 */
template <typename T>
struct value_helper {
  static void change_val(T& value) { value = T{1}; }
  static bool compare_val(const T& left, const T& right) {
    return (left == right);
  }
};

/** @brief The function helps to change and compare arrays
 *  @tparam T type of array values
 *  @tparam N size of array
 */
template <typename T, size_t N>
struct value_helper<T[N]> {
  using arrayT = T[N];
  static void change_val(arrayT& value) {
    for (size_t i = 0; i < N; i++) {
      value[i] = T{1};
    }
  }
  static bool compare_val(const arrayT& left, const arrayT right) {
    for (size_t i = 0; i < N; i++) {
      if (left != right) return false;
    }
    return true;
  }
};
}  // namespace device_global_common_functions

#endif  // SYCL_CTS_TEST_DEVICE_GLOBAL_COMMON_H
