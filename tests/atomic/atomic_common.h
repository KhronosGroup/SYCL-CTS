/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common functions for atomic tests
//
*******************************************************************************/

#ifndef SYCL_CTS_TESTS_ATOMIC_COMMON_H
#define SYCL_CTS_TESTS_ATOMIC_COMMON_H

#include "../common/common.h"

#include <climits>
#include <type_traits>

/**
 * @brief Namespace that defines 64bit requirement tags for type given
 */
namespace atomic64_bits_tag {
struct generic {};
struct no : generic {};
struct yes : generic {};

template <typename T>
constexpr auto get() {
  using tag_t =
      typename std::conditional < sizeof(T) * CHAR_BIT<64, no, yes>::type;
  return tag_t{};
}
}  // namespace atomic64_bits_tag

#endif  // SYCL_CTS_TESTS_ATOMIC_COMMON_H
