/*************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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

/** User defined struct that is used in accessor tests
 */
struct user_struct {
  float a;
  int b;
  char c;

  using element_type = int;

  user_struct() : a(1), b(2), c(3) {};

  user_struct(int val) : a(1), b(val), c(3) {}

  element_type operator[](size_t index) const { return b; }

  friend bool operator==(const user_struct& lhs, const user_struct& rhs) {
    return (lhs.a == rhs.a) && (lhs.b == rhs.b) && (lhs.c == rhs.c);
  }
};

namespace user_namespace {
  /** User defined type alias from different namespace that is used in accessor
   *  tests
   */
  using user_alias = int;

  /** User defined nested struct that is used in accessor tests
   */
  struct nested {
    struct user_struct {
      int a;
      accessor_utility::user_struct b;

      using element_type = int;

      user_struct() : a(1) {};

      user_struct(int val) : a(1), b(val) {}

      element_type operator[](size_t index) const { return b[index]; }

      friend bool operator==(const user_struct& lhs, const user_struct& rhs) {
        return (lhs.a == rhs.a) && (lhs.b == rhs.b);
      }
    };
  };
}

/** Convenient compile-time evaluation to determine if an accessor is an image
 *  accessor (of sorts)
 */
template <sycl::target target>
struct is_image {
  static constexpr auto value =
      target == sycl::target::image ||
      target == sycl::target::host_image ||
      target == sycl::target::image_array;
};

/** Convenient compile-time evaluation to determine if an accessor is an local
 *  accessor
 */
template <sycl::target target>
struct is_local {
  static constexpr auto value = (target == sycl::target::local);
};

/** Convenient compile-time evaluation to determine if an accessor is an buffer
 *  accessor (of sorts)
 */
template <sycl::target target>
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
template <typename T, int dims, sycl::access::mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder =
              sycl::access::placeholder::false_t>
std::string accessor_type_name(const std::string& dataType) {
  std::stringstream stream;
  stream << "accessor<" << type_name_string<T>::get(dataType) << ", " << dims
         << ", mode{" << static_cast<int>(mode) << "}, target{"
         << static_cast<int>(target) << "}";
  if (!is_image<target>::value) {
    stream << ", placeholder{"
           << (placeholder == sycl::access::placeholder::true_t) << "}";
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
template <typename T, int dims, sycl::access::mode mode,
          sycl::target target,
          sycl::access::placeholder placeholder =
              sycl::access::placeholder::false_t>
void fail_for_accessor(sycl_cts::util::logger& log,
                       const std::string& dataType,
                       const std::string& message) {
  const auto accTypeName =
      accessor_type_name<T, dims, mode, target, placeholder>(dataType);
  FAIL(log, (accTypeName + ": " + message));
}

}  // namespace accessor_utility

#endif  // SYCL_1_2_1_TESTS_ACCESSOR_ACCESSOR_UTILITY_COMMON_H
