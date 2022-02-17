/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_CONVERSION_H
#define __SYCLCTS_TESTS_COMMON_CONVERSION_H

#include <type_traits>

namespace {

/**
 * @brief Static cast scoped enum value to the underlying type
 */
template <typename enumT>
constexpr auto to_integral(enumT const& value) {
  if constexpr (std::is_enum_v<enumT>) {
    return static_cast<typename std::underlying_type<enumT>::type>(value);
  } else if constexpr (std::is_integral_v<enumT>) {
    return value;
  } else {
    constexpr bool always_false = !std::is_same_v<enumT, enumT>;
    static_assert(always_false, "Unsupported type");
  }
}

}  //  namespace

#endif  // __SYCLCTS_TESTS_COMMON_CONVERSION_H
