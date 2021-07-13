/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
// Provide common functions for type coverage
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_TYPE_COVERAGE_H
#define __SYCLCTS_TESTS_COMMON_TYPE_COVERAGE_H

#include "../common/sycl.h"
#include <string>
#include <utility>

/**
 * @brief Retrieve type name; by default just forward the given one
 */
template <typename T>
struct type_name_string {
    static std::string get(std::string dataType) {
        return dataType;
    }
};

/**
 * @brief Specialization of type name retrievement for sycl::vec class
 * @param T Type of the data stored in vector
 * @param nElements Number of elements stored in vector
 */
template <typename T, size_t nElements>
struct type_name_string<sycl::vec<T, nElements>> {
    static std::string get(const std::string& dataType) {
        return "sycl::vec<" + dataType + "," +
               std::to_string(nElements) + ">";
    }
};

/**
 * @brief Type pack to store types
 */
template <typename ... T>
struct type_pack {
};

/**
 * @brief Type pack to store types and underlying data type names to use with
 *        type_name_string
 */
template <typename ... T>
struct named_type_pack {
  const std::string names[sizeof...(T)];

  template <typename ... nameListT>
  named_type_pack(nameListT&&... nameList):
    names{std::forward<nameListT>(nameList)...}{
  }
};

/**
 * @brief Run action for each of types given by type_pack instance
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam types Deduced from type_pack parameter pack for list of types to use
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param args Arguments to forward into the call
 */
template <template<typename, typename...> class action,
          typename ... actionArgsT, typename ... types, typename ... argsT>
inline void for_all_types(const type_pack<types...>&, argsT&& ... args) {
  /** run action for each type from types... parameter pack
  */
  int packExpansion[] = {(
    action<types, actionArgsT...>{}(std::forward<argsT>(args)...),
    0 // Dummy initialization value
  )...};
  static_cast<void>(packExpansion);
}

/**
 * @brief Run action for each of types given by named_type_pack instance
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam types Deduced from type_pack parameter pack for list of types to use
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param typeList Named type pack instance with type names stored
 * @param args Arguments to forward into the call
 */
template <template<typename, typename...> class action,
          typename ... actionArgsT, typename ... types, typename ... argsT>
inline void for_all_types(const named_type_pack<types...>& typeList,
                          argsT&& ... args) {
  /** run action for each type from types... parameter pack
  */
  size_t typeNameIndex = 0;

  int packExpansion[] = {(
    action<types, actionArgsT...>{}(std::forward<argsT>(args)...,
                                    typeList.names[typeNameIndex]),
    ++typeNameIndex,
    0 // Dummy initialization value
  )...};
  /** Every initializer clause is sequenced before any initializer clause
   *  that follows it in the braced-init-list. Every expression in comma
   *  operator is also strictly sequnced. So we can use increment safely.
   *  We still should discard dummy results, but this initialization
   *  should not be optimized out due side-effects
   */
  static_cast<void>(packExpansion);
}

/**
 * @brief Run action for type and for all vectors of this type
 * @tparam action Functor template for action to run
 * @tparam T Type to instantiate functor template with
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param args Arguments to forward into the call
 */
template <template<typename, typename...> class action, typename T,
          typename ... actionArgsT, typename ... argsT>
void for_type_and_vectors(argsT&& ... args) {
  static const auto types = type_pack<T,
                                      typename sycl::template vec<T,1>,
                                      typename sycl::template vec<T,2>,
                                      typename sycl::template vec<T,3>,
                                      typename sycl::template vec<T,4>,
                                      typename sycl::template vec<T,8>,
                                      typename sycl::template vec<T,16>>{};
  // Use type_pack without names here for lazy log message construction
  for_all_types<action, actionArgsT...>(types, std::forward<argsT>(args)...);
}

/**
 * @brief Run action for each of types and vectors of types given by
 *        named_type_pack instance
 * @tparam action Functor template for action to run
 * @tparam actionArgsT Parameter pack to use for functor template instantiation
 * @tparam types Deduced from type_pack parameter pack for list of types to use
 * @tparam argsT Deduced parameter pack for arguments to forward into the call
 * @param typeList Named type pack instance with underlying type names stored
 * @param args Arguments to forward into the call
 */
template <template<typename, typename...> class action,
          typename ... actionArgsT, typename ... types, typename ... argsT>
void for_all_types_and_vectors(const named_type_pack<types...>& typeList,
                               argsT&& ... args) {
  /** run action for each type from types... parameter pack
  */
  size_t typeNameIndex = 0;

  int packExpansion[] = {(
    for_type_and_vectors<action, types, actionArgsT...>(std::forward<argsT>(args)...,
                                    typeList.names[typeNameIndex]),
    ++typeNameIndex,
    0 // Dummy initialization value
  )...};
  static_cast<void>(packExpansion);
}

#endif  // __SYCLCTS_TESTS_COMMON_TYPE_COVERAGE_H
