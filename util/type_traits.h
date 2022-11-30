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

/**
 * @brief Verify \c T has implemented operator*()
 */
template <typename T, typename = void>
struct is_dereferenceable : std::false_type {};

template <typename T>
struct is_dereferenceable<T, std::void_t<decltype(*std::declval<T>())>>
    : std::true_type {};

/**
 * @brief Shortcut for is_dereferenceable::value
 */
template <typename T>
inline constexpr bool is_dereferenceable_v = is_dereferenceable<T>::value;

/**
 * @brief Verify \c T has size member function
 */
template <typename T, typename = void>
struct has_size : std::false_type {};

template <typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

/**
 * @brief Shortcut for has_size::type
 */
template <typename T>
using has_size_t = typename has_size<T>::type;

/**
 * @brief Shortcut for has_size::value
 */
template <typename T>
constexpr inline bool has_size_v = has_size_t<T>::value;

/**
 * @brief Verify \c T has both subscript operator and size member function
 */
template <typename T>
using has_subscript_and_size =
    std::conjunction<has_subscript_operator<T>, has_size<T>>;

/**
 * @brief Shortcut for has_subscript_and_size::value
 */
template <typename T>
constexpr inline bool has_subscript_and_size_v =
    has_subscript_and_size<T>::value;

}  // namespace

//  type_traits for specifying that provided type have one of the following
//  things:
//    - Some member types (e.g. value_type or difference_type)
//    - Implemented operators (e.g. operator++(), operator+=(), operator>())
namespace type_traits {
// Provide code to verify that provided datatype has different fields
namespace has_field {

template <typename T, typename = void>
struct value_type : std::false_type {};

template <typename T>
struct value_type<T, std::void_t<typename std::iterator_traits<T>::value_type>>
    : std::true_type {};

template <typename T>
inline constexpr bool value_type_v = value_type<T>::value;

template <typename T, typename = void>
struct difference_type : std::false_type {};

template <typename T>
struct difference_type<
    T, std::void_t<typename std::iterator_traits<T>::difference_type>>
    : std::true_type {};

template <typename T>
inline constexpr bool difference_type_v = difference_type<T>::value;

template <typename T, typename = void>
struct reference : std::false_type {};

template <typename T>
struct reference<T, std::void_t<typename std::iterator_traits<T>::reference>>
    : std::true_type {};

template <typename T>
inline constexpr bool reference_v = reference<T>::value;

template <typename T, typename = void>
struct pointer : std::false_type {};

template <typename T>
struct pointer<T, std::void_t<typename std::iterator_traits<T>::pointer>>
    : std::true_type {};

template <typename T>
inline constexpr bool pointer_v = pointer<T>::value;

template <typename T, typename = void>
struct iterator_category : std::false_type {};

template <typename T>
struct iterator_category<T,
                         std::void_t<typename std::iterator_traits<T>::pointer>>
    : std::true_type {};

template <typename T>
inline constexpr bool iterator_category_v = iterator_category<T>::value;

}  // namespace has_field

// Provide code to verify that provided datatype has compound assignment
namespace compound_assignment {
template <typename T, typename RightOperandT = int, typename = void>
struct addition : std::false_type {};

template <typename T, typename RightOperandT>
struct addition<
    T, RightOperandT,
    // Need reference here to handle rval of native C++ types like int
    std::void_t<decltype(std::declval<T&>() += std::declval<RightOperandT>())>>
    : std::true_type {};

template <typename T, typename RightOperandT = int>
inline constexpr bool addition_v = addition<T, RightOperandT>::value;

template <typename T, typename RightOperandT = int, typename = void>
struct subtraction : std::false_type {};

template <typename T, typename RightOperandT>
struct subtraction<
    T, RightOperandT,
    // Need reference here to handle rval of native C++ types like int
    std::void_t<decltype(std::declval<T&>() -= std::declval<RightOperandT>())>>
    : std::true_type {};

template <typename T, typename RightOperandT = int>
inline constexpr bool subtraction_v = subtraction<T>::value;
}  // namespace compound_assignment

// Provide code to verify that provided datatype has arithmetic operators
namespace has_arithmetic {

template <typename LeftOperand, typename RightOperand, typename = void>
struct addition : std::false_type {};

template <typename LeftOperand, typename RightOperand>
struct addition<LeftOperand, RightOperand,
                std::void_t<decltype(std::declval<LeftOperand>() +
                                     std::declval<RightOperand>())>>
    : std::true_type {};

template <typename LeftOperand, typename RightOperand>
inline constexpr bool addition_v = addition<LeftOperand, RightOperand>::value;

template <typename LeftOperand, typename RightOperand, typename = void>
struct subtraction : std::false_type {};

template <typename LeftOperand, typename RightOperand>
struct subtraction<LeftOperand, RightOperand,
                   std::void_t<decltype(std::declval<LeftOperand&>() -
                                        std::declval<RightOperand>())>>
    : std::true_type {};

template <typename LeftOperand, typename RightOperand>
inline constexpr bool subtraction_v =
    subtraction<LeftOperand, RightOperand>::value;

template <typename T, typename = void>
struct pre_increment : std::false_type {};

template <typename T>
// Need reference here to handle rval of native C++ types like int
struct pre_increment<T, std::void_t<decltype(++std::declval<T&>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool pre_increment_v = pre_increment<T>::value;

template <typename T, typename = void>
struct post_increment : std::false_type {};

template <typename T>
// Need reference here to handle rval of native C++ types like int
struct post_increment<T, std::void_t<decltype(std::declval<T&>()++)>>
    : std::true_type {};

template <typename T>
inline constexpr bool post_increment_v = post_increment<T>::value;

template <typename T, typename = void>
struct post_decrement : std::false_type {};

template <typename T>
// Need reference here to handle rval of native C++ types like int
struct post_decrement<T, std::void_t<decltype(std::declval<T&>()--)>>
    : std::true_type {};

template <typename T>
inline constexpr bool post_decrement_v = post_decrement<T>::value;

template <typename T, typename = void>
struct pre_decrement : std::false_type {};

template <typename T>
// Need reference here to handle rval of native C++ types like int
struct pre_decrement<T, std::void_t<decltype(--std::declval<T&>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool pre_decrement_v = pre_decrement<T>::value;
}  // namespace has_arithmetic

// Provide code to verify that provided datatype has comparison operators
namespace has_comparison {
template <typename T, typename = void>
struct is_equal : std::false_type {};

template <typename T>
struct is_equal<T,
                std::void_t<decltype(std::declval<T>() == std::declval<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_equal_v = is_equal<T>::value;

template <typename T, typename = void>
struct not_equal : std::false_type {};

template <typename T>
struct not_equal<T,
                 std::void_t<decltype(std::declval<T>() != std::declval<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool not_equal_v = not_equal<T>::value;

template <typename T, typename = void>
struct greater_than : std::false_type {};

template <typename T>
struct greater_than<
    T, std::void_t<decltype(std::declval<T>() > std::declval<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool greater_than_v = greater_than<T>::value;

template <typename T, typename = void>
struct less_than : std::false_type {};

template <typename T>
struct less_than<T,
                 std::void_t<decltype(std::declval<T>() < std::declval<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool less_than_v = less_than<T>::value;

template <typename T, typename = void>
struct greater_or_equal : std::false_type {};

template <typename T>
struct greater_or_equal<
    T, std::void_t<decltype(std::declval<T>() >= std::declval<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool greater_or_equal_v = greater_or_equal<T>::value;

template <typename T, typename = void>
struct less_or_equal : std::false_type {};

template <typename T>
struct less_or_equal<
    T, std::void_t<decltype(std::declval<T>() <= std::declval<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool less_or_equal_v = less_or_equal<T>::value;

}  // namespace has_comparison
}  // namespace type_traits

#endif  // __SYCLCTS_UTIL_TYPE_TRAITS_H
