/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common type traits support
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_TYPE_TRAITS_H
#define __SYCLCTS_UTIL_TYPE_TRAITS_H

#include <climits>
#include <type_traits>

namespace {

// Common type traits functions
template <typename ... T>
struct contains;

template <typename T>
struct contains<T> : std::false_type {};

/**
 * @brief Verify type is within the list of types
 */
template <typename T, typename headT, typename ... tailT>
struct contains<T, headT, tailT...> :
    std::conditional<std::is_same<T, headT>::value,
                     std::true_type,
                     contains<T, tailT...>>::type {};

/**
 * @brief Verify type has the given number of bits
 */
template <typename T, size_t bits>
using bits_eq = std::bool_constant<(sizeof(T) * CHAR_BIT) == bits>;

// Specific type traits functions
/**
 * @brief Verify type is within the list of types with atomic support
 *        Note that different implementation-defined aliases can actually fall
 *        into this set
 */
template <typename T>
using has_atomic_support =
    contains<T,
             int, unsigned int,
             long, unsigned long,
             long long, unsigned long long,
             float>;

/**
 * @brief Checks whether T is a floating-point sycl type
 */
template <typename T>
using is_cl_float_type =
    std::bool_constant<std::is_floating_point<T>::value ||
                       std::is_same<cl::sycl::half, T>::value ||
                       std::is_same<cl::sycl::cl_float, T>::value ||
                       std::is_same<cl::sycl::cl_double, T>::value ||
                       std::is_same<cl::sycl::cl_half, T>::value>;

} // namespace

#endif // __SYCLCTS_UTIL_TYPE_TRAITS_H
