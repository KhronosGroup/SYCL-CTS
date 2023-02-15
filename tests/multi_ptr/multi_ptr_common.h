/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common functions for multi_ptr tests
//
*******************************************************************************/

#ifndef SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_COMMON_H
#define SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_COMMON_H

#include "../common/common.h"
#include "../common/type_coverage.h"
#include "../common/type_list.h"

namespace Catch {
template <>
struct StringMaker<sycl::access::address_space> {
  using type = sycl::access::address_space;
  static std::string convert(type value) {
    switch (value) {
      case type::global_space:
        return "access::address_space::global_space";
      case type::local_space:
        return "access::address_space::local_space";
      case type::private_space:
        return "access::address_space::private_space";
      case type::generic_space:
        return "access::address_space::generic_space";
      default:
        // no stringification for deprecated ones
        return "unknown or deprecated address_space";
    }
  }
};

template <>
struct StringMaker<sycl::access::decorated> {
  using type = sycl::access::decorated;
  static std::string convert(type value) {
    switch (value) {
      case type::yes:
        return "access::decorated::yes";
      case type::no:
        return "access::decorated::no";
      case type::legacy:
        return "access::decorated::legacy";
      default:
        return "unknown";
    }
  }
};
}  // namespace Catch

namespace multi_ptr_common {

/**
 * @brief Factory function for getting type_pack with fields of
 *        sycl::access::address_space enums
 */
inline auto get_address_spaces() {
  return value_pack<
      sycl::access::address_space, sycl::access::address_space::global_space,
      sycl::access::address_space::local_space,
      sycl::access::address_space::private_space,
      sycl::access::address_space::generic_space>::generate_named();
}

/**
 * @brief Factory function for getting type_pack with fields of
 *        sycl::access::address_space enums
 */
inline auto get_decorated() {
  return value_pack<sycl::access::decorated, sycl::access::decorated::yes,
                    sycl::access::decorated::no>::generate_named();
}

/** @brief Legacy multi_ptr alias to enforce the access::decorated::legacy
 *         usage with no dependency on default multi_ptr template parameter
 *         values
 */
template <typename T, sycl::access::address_space Space>
using multi_ptr_legacy =
    sycl::multi_ptr<T, Space, sycl::access::decorated::legacy>;

template <typename T>
using global_ptr_legacy =
    multi_ptr_legacy<T, sycl::access::address_space::global_space>;

template <typename T>
using private_ptr_legacy =
    multi_ptr_legacy<T, sycl::access::address_space::private_space>;

template <typename T>
using local_ptr_legacy =
    multi_ptr_legacy<T, sycl::access::address_space::local_space>;

template <typename T>
using constant_ptr_legacy =
    multi_ptr_legacy<T, sycl::access::address_space::constant_space>;

/** @brief Factory method to enforce the same coverage for constructors and API
 */
inline auto get_types() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return named_type_pack<bool, float, char,           // types grouped
                         signed char, unsigned char,  // by sign
                         short, unsigned short,       //
                         int, unsigned int,           //
                         long, unsigned long,         //
                         long long,
                         unsigned long long>::generate("bool", "float", "char",
                                                       "signed char",
                                                       "unsigned char", "short",
                                                       "unsigned short", "int",
                                                       "unsigned int", "long",
                                                       "unsigned long",
                                                       "long long",
                                                       "unsigned long long");
#else
  return named_type_pack<int, float>::generate("int", "float");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
}

// custom data types that will be used in type coverage
inline auto get_composite_types() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return named_type_pack<user_def_types::no_cnstr, user_def_types::def_cnstr,
                         user_def_types::no_def_cnstr>::generate(
      "no_cnstr", "def_cnstr", "no_def_cnstr");
#else
  return named_type_pack<user_def_types::def_cnstr>::generate("def_cnstr");
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
}

template <typename... argsT>
void silence_warnings(argsT...) {
  // Just to avoid compiler warnings on unused variables
}

/** @brief Wrapper with type pairs for multi_ptr with 'void' type verification
 */
template <template <typename, typename> class action, typename T>
struct check_void_pointer {
  using data_t = typename std::remove_const<T>::type;

  template <typename... argsT>
  void operator()(argsT &&...args) {
    action<data_t, void>{}(std::forward<argsT>(args)..., "void");
    action<const data_t, const void>{}(std::forward<argsT>(args)..., "void");
  }
};

/** @brief Wrapper with type pairs for generic multi_ptr verification
 */
template <template <typename, typename> class action, typename T>
struct check_pointer {
  using data_t = typename std::remove_const<T>::type;

  template <typename... argsT>
  void operator()(argsT &&...args) {
    action<data_t, data_t>{}(std::forward<argsT>(args)...);
    action<const data_t, const data_t>{}(std::forward<argsT>(args)...);
  }
};

/** @brief Initializes all elements in a container with 0
 */
template <typename T, int N, int... Vals>
struct zero_init_container : zero_init_container<T, N - 1, 0, Vals...> {};
template <typename T, int... Vals>
struct zero_init_container<T, 0, Vals...> {
  using elem_type = typename T::value_type;
  static inline constexpr T value = {elem_type{Vals}...};
};

/** @brief Initializes an std::array with either 0's or using the default
 * constructor
 */
template <typename T, int N>
struct init_array : zero_init_container<std::array<T, N>, N> {};
template <int N>
struct init_array<user_def_types::def_cnstr, N> {
  static inline constexpr std::array<user_def_types::def_cnstr, N> value;
};

}  // namespace multi_ptr_common

#endif  // SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_COMMON_H
