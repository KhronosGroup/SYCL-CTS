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
#include "../../common/value_operations.h"

namespace device_global_common_functions {

/** @brief Enum for kernel names
 */
enum class test_names : size_t {
  get_multi_ptr_method,
  implicit_conversation,
  get_method,
  subscript_operator,
  arrow_operator,
};

/**
 * @brief Enum class for kernel names in functional tests. Cannot use enums from
 * properties as they are nested in struct and therefore non forward declarable
 */
enum class property_tag : size_t {
  none,
  dev_image_scope,
  host_access_r,
  host_access_w,
  host_access_none,
  host_access_r_w,
  init_mode_trig_reset,
  init_mode_trig_reprogram,
  impl_in_csr_true,
  impl_in_csr_false
};

/** @brief Generate test description string
 *  @tparam Decorated flag that will be converted to string and added to the
 * result description
 *  @retval String with a description of test case
 *  @param text_name The string to show in the case description
 *  @param info The string with additional information about the test case
 *  @param type_name The string interpretation of data type in the test case
 */
template <sycl::access::decorated Decorated>
inline std::string get_case_description(const std::string& test_name,
                                        const std::string& info,
                                        const std::string& type_name) {
  const std::string decorated{
      sycl_cts::get_cts_string::for_decorated<Decorated>()};
  std::string message;
  message += test_name + " error: " + info;
  message += " with tparams:";
  message += "<" + type_name + ">";
  message += "<" + decorated + ">";
  return message;
}

/** @brief Generate test description string
 *  @retval The string with a description of test case
 *  @param text_name The string to show in the case description
 *  @param info The string with additional information about the test case
 *  @param type_name The string interpretation of data type in the test case
 */
inline std::string get_case_description(const std::string& test_name,
                                        const std::string& info,
                                        const std::string& type_name) {
  std::string message;
  message += test_name + " error: " + info;
  message += " with tparams:";
  message += "<" + type_name + ">";
  return message;
}

/** @brief The helper function to get variable address for pointer
 *  @tparam T Type of variable
 *  @param data variable to get pointer to
 */
template <typename T>
inline T* pointer_helper(T& data) {
  return &data;
}

/** @brief The helper function to get first element of array address for pointer
 *  @tparam T Type of array values
 *  @tparam N Size of array
 *  @param data array to get pointer to
 */
template <typename T, size_t N>
inline T* pointer_helper(T (&data)[N]) {
  return &data[0];
}
}  // namespace device_global_common_functions

#endif  // SYCL_CTS_TEST_DEVICE_GLOBAL_COMMON_H
