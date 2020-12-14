/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  This file is a common utility for the implementation of
//  accessor_constructors.cpp and accessor_api.cpp.
//
**************************************************************************/
#ifndef SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_COMMON_H
#define SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_COMMON_H

#include "../common/common.h"
#include "../common/type_coverage.h"

#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace accessor_utility {

/** user defined struct that is used in accessor tests
*/
struct user_struct {
  float a;
  int b;
  char c;

  using element_type = int;

  user_struct() : a(0), b(0), c(0) {};

  user_struct(int val) : a(0), b(val), c(0) {}

  element_type operator[](size_t index) const { return b; }

  friend bool operator==(const user_struct& lhs, const user_struct& rhs) {
    static constexpr auto eps = 1e-4f;
    return (((lhs.a + eps > rhs.a) && (lhs.a < rhs.a + eps)) &&
            (lhs.b == rhs.b) && (lhs.c == rhs.c));
  }
};

/** Convenient compile-time evaluation to determine if an accessor is an image
 *  accessor (of sorts)
 */
template <cl::sycl::access::target target>
struct is_image {
  static constexpr auto value =
      target == cl::sycl::access::target::image ||
      target == cl::sycl::access::target::host_image ||
      target == cl::sycl::access::target::image_array;
};

/** Convenient compile-time evaluation to determine if an accessor is an local
 *  accessor
 */
template <cl::sycl::access::target target>
struct is_local {
  static constexpr auto value = (target == cl::sycl::access::target::local);
};

/** Convenient compile-time evaluation to determine if an accessor is an buffer
 *  accessor (of sorts)
 */
template <cl::sycl::access::target target>
struct is_buffer {
  static constexpr auto value =
      !is_image<target>::value && !is_local<target>::value;
};

/**
 * @brief Retrive type of the accessor
 * @tparam T Underlying type of the accessor
 * @tparam dims Number of accessor dimensions
 * @tparam mode Access mode used
 * @tparam target Access target used
 * @tparam placeholder Whether the accessor is a placeholder
 * @param typeName The name of the underlying data type for scalar or vec types
 */
template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
std::string accessor_type_name(const std::string& dataType) {
  std::stringstream stream;
  stream << "accessor<" << type_name_string<T>::get(dataType) << ", " << dims
         << ", mode{" << static_cast<int>(mode) << "}, target{"
         << static_cast<int>(target) << "}";
  if (!is_image<target>::value) {
    stream << ", placeholder{"
           << (placeholder == cl::sycl::access::placeholder::true_t) << "}";
  }
  stream << ">";
  return stream.str();
}

/**
 * @brief Retrive type of the accessor
 * @tparam T Underlying type of the accessor
 * @tparam dims Number of accessor dimensions
 * @tparam mode Access mode used
 * @tparam target Access target used
 * @tparam placeholder Whether the accessor is a placeholder
 * @param typeName The name of the underlying data type for scalar or vec types
 */
template <typename T, int dims, cl::sycl::access::mode mode,
          cl::sycl::access::target target,
          cl::sycl::access::placeholder placeholder =
              cl::sycl::access::placeholder::false_t>
void fail_for_accessor(sycl_cts::util::logger& log,
                       const std::string& dataType,
                       const std::string& message) {
  const auto accTypeName =
      accessor_type_name<T, dims, mode, target, placeholder>(dataType);
  FAIL(log, (accTypeName + ": " + message));
}

}  // namespace accessor_utility

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_COMMON_H
