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
#include <sycl/sycl.hpp>
#include <type_traits>

namespace {

// Common type traits functions
template <typename... T>
struct contains;

template <typename T>
struct contains<T> : std::false_type {};

/**
 * @brief Verify type is within the list of types
 */
template <typename T, typename headT, typename... tailT>
struct contains<T, headT, tailT...>
    : std::conditional<std::is_same<T, headT>::value, std::true_type,
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
using has_atomic_support = contains<T, int, unsigned int, long, unsigned long,
                                    long long, unsigned long long, float>;

/**
 * @brief Checks whether T is a floating-point sycl type
 */
template <typename T>
using is_sycl_floating_point =
    std::bool_constant<std::is_floating_point_v<T> ||
                       std::is_same_v<T, sycl::half>>;

template <typename T>
inline constexpr bool is_sycl_floating_point_v{
    is_sycl_floating_point<T>::value};

template <typename T>
using is_nonconst_rvalue_reference =
    std::bool_constant<std::is_rvalue_reference_v<T> &&
                       !std::is_const_v<typename std::remove_reference_t<T>>>;

template <typename T>
inline constexpr bool is_nonconst_rvalue_reference_v{
    is_nonconst_rvalue_reference<T>::value};

namespace has_static_member {

template <typename, typename = void>
struct to_string : std::false_type {};

template <typename T>
struct to_string<T, std::void_t<decltype(T::to_string())>> : std::true_type {};

}  // namespace has_static_member

/**
 * @brief Verify \c T has subscript subscript operator
 */
template <typename T, typename = void>
struct has_subscript_operator : std::false_type {};

template <typename T>
struct has_subscript_operator<
    T, std::void_t<decltype(std::declval<T>()[std::declval<size_t>()])>>
    : std::true_type {};

/**
 * @brief Shortcut for has_subscript_operator::type
 */
template <typename T>
using has_subscript_operator_t = typename has_subscript_operator<T>::type;

/**
 * @brief Shortcut for has_subscript_operator::value
 */
template <typename T>
inline constexpr bool has_subscript_operator_v =
    has_subscript_operator_t<T>::value;

}  // namespace

#endif  // __SYCLCTS_UTIL_TYPE_TRAITS_H
