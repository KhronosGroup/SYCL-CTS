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

namespace multi_ptr_common {

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
#ifdef SYCL_CTS_FULL_CONFORMANCE
  return named_type_pack<bool, float, double, char,   // types grouped
                         signed char, unsigned char,  // by sign
                         short, unsigned short,       //
                         int, unsigned int,           //
                         long, unsigned long,         //
                         long long, unsigned long long>::generate(
      "bool",        "float",
      "double",      "char",
      "signed char", "unsigned char",
      "short",       "unsigned short",
      "int",         "unsigned int",
      "long",        "unsigned long",
      "long long",   "unsigned long long");
#else
  return named_type_pack<int, float>::generate("int", "float");
#endif  // SYCL_CTS_FULL_CONFORMANCE
}

// custom data types that will be used in type coverage
inline auto get_composite_types() {
#ifdef SYCL_CTS_FULL_CONFORMANCE
  return named_type_pack<user_def_types::no_cnstr, user_def_types::def_cnstr,
                         user_def_types::no_def_cnstr>::generate(
      "no_cnstr", "def_cnstr", "no_def_cnstr");
#else
  return named_type_pack<user_def_types::def_cnstr>::generate("def_cnstr");
#endif  // SYCL_CTS_FULL_CONFORMANCE
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

}  // namespace multi_ptr_common

#endif  // SYCL_CTS_TEST_MULTI_PTR_MULTI_PTR_COMMON_H
