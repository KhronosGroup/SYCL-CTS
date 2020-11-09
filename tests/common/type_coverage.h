/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
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
 * @brief Specialization of type name retrievement for cl::sycl::vec class
 * @param T Type of the data stored in vector
 * @param nElements Number of elements stored in vector
 */
template <typename T, size_t nElements>
struct type_name_string<cl::sycl::vec<T, nElements>> {
    static std::string get(const std::string& dataType) {
        return "cl::sycl::vec<" + dataType + "," +
               std::to_string(nElements) + ">";
    }
};

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
  /** check scalar type
  */
  action<T, actionArgsT...>{}(std::forward<argsT>(args)...);
  /** check all vec types
  */
  action<typename cl::sycl::template vec<T,1>, actionArgsT...>{}(
      std::forward<argsT>(args)...);
  action<typename cl::sycl::template vec<T,2>, actionArgsT...>{}(
      std::forward<argsT>(args)...);
  action<typename cl::sycl::template vec<T,3>, actionArgsT...>{}(
      std::forward<argsT>(args)...);
  action<typename cl::sycl::template vec<T,4>, actionArgsT...>{}(
      std::forward<argsT>(args)...);
  action<typename cl::sycl::template vec<T,8>, actionArgsT...>{}(
      std::forward<argsT>(args)...);
  action<typename cl::sycl::template vec<T,16>, actionArgsT...>{}(
      std::forward<argsT>(args)...);
}

#endif  // __SYCLCTS_TESTS_COMMON_TYPE_COVERAGE_H
