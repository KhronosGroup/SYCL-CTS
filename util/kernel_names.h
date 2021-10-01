/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide adapter for safely using different types with kernel names in
//  generic code.
//
//  For example, according to the SYCL 2020 for any type T that we cannot
//  forward-declare we are unable to have
//    single_task<kernel_name<T>>,
//  but we are able to have
//    using U = typename kernel_name::adapter<T>::type
//    single_task<kernel_name<U>>
//  in case we have a proper implementation of the kernel_name::adapter
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_KERNEL_NAMES_H
#define __SYCLCTS_UTIL_KERNEL_NAMES_H

#include <string>
#include <utility>

namespace sycl_cts {
namespace util {
namespace kernel_name {

namespace details {

template <typename T>
struct adapter_core {};

struct adapter_int8_t {};
struct adapter_int16_t {};
struct adapter_int32_t {};
struct adapter_int64_t {};
struct adapter_uint8_t {};
struct adapter_uint16_t {};
struct adapter_uint32_t {};
struct adapter_uint64_t {};

struct adapter_byte {};
struct adapter_size_t {};

/** @brief Provides type that is safe to use in kernel names based on type given
 *  @details There is a plenty of restrictions on types we may use for kernel
 *           naming. The most obvious is non-forward-declarable classes, but
 *           there are more of them.
 *           This function uses discarded statements to map input type into the
 *           return type we may use for the kernel name template parameters.
 *           Note: it does not provide different kernel names for the type
 *           aliases, so we may have the same kernel name
 *             - for `int` and `int32_t`, or
 *             - for `size_t` and `uint64_t`, or
 *             - for any other possible type alias
 *           on any specific SYCL implementation. We may say it provides adapter
 *           aliases for the type aliases.
 */
template <typename T>
constexpr auto adapter_helper() {
  // Fixed-width types are optional and implementation-defined
#ifdef INT8_MAX
  if constexpr (std::is_same_v<T, std::int8_t>) {
    return adapter_int8_t{};
  } else
#endif
#ifdef INT16_MAX
      if constexpr (std::is_same_v<T, std::int16_t>) {
    return adapter_int16_t{};
  } else
#endif
#ifdef INT32_MAX
      if constexpr (std::is_same_v<T, std::int32_t>) {
    return adapter_int32_t{};
  } else
#endif
#ifdef INT64_MAX
      if constexpr (std::is_same_v<T, std::int64_t>) {
    return adapter_int64_t{};
  } else
#endif
#ifdef UINT8_MAX
      if constexpr (std::is_same_v<T, std::uint8_t>) {
    return adapter_uint8_t{};
  } else
#endif
#ifdef UINT16_MAX
      if constexpr (std::is_same_v<T, std::uint16_t>) {
    return adapter_uint16_t{};
  } else
#endif
#ifdef UINT32_MAX
      if constexpr (std::is_same_v<T, std::uint32_t>) {
    return adapter_uint32_t{};
  } else
#endif
#ifdef UINT64_MAX
      if constexpr (std::is_same_v<T, std::uint64_t>) {
    return adapter_uint64_t{};
  } else
#endif
      if constexpr (std::is_same_v<T, std::byte>) {
    // std::byte is a scoped enum according to the C++17
    return adapter_byte{};
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    return adapter_size_t{};
  } else {
    return adapter_core<T>{};
  }
}

};  // namespace details

/** @brief Safe wrapper for types to use within kernel names
 *  @details According to C++17 [dcl.typedef] type alias does not introduce a
 *           new type. So we are free to provide adapter for types we cannot use
 *           for kernel names directly. For example, for any type we cannot
 *           forward-declare:
 *               struct outer {
 *                 struct inner {};
 *               };
 *           we are unable to have
 *               using T = outer::inner;
 *               single_task<kernel_name<T>>,
 *           but we are able to have
 *               using T = typename kernel_name::adapter<outer::inner>::type
 *               single_task<kernel_name<T>>
 *           in case we provide a proper template specialization for adapter.
 *
 *           Test developer is free to provide template specialization for any
 *           test-specific type.
 */
template <typename T>
struct adapter {
  using type = decltype(details::adapter_helper<T>());
};

/** @brief Syntax sugar for code brevity
 */
template <typename T>
using adapter_t = typename adapter<T>::type;

}  // namespace kernel_name
}  // namespace util
}  // namespace sycl_cts

#endif  // __SYCLCTS_UTIL_KERNEL_NAMES_H
