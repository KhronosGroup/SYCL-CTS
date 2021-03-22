/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common template parameter packs support
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_VARIADIC_H
#define __SYCLCTS_UTIL_VARIADIC_H

#include <cstddef>
#include <type_traits>

/**
 * @brief Can be used to run functors with parameter pack and custom parameters
 * @tparam functorT Functor to use
 */
template <class functorT>
struct run_variadic {
  /**
   * @brief Call functor with all parameters accumulated.
   * @tparam nParams Number of params to use as parameter pack, should be 0
   * @tparam T Type of params storage
   * @tparam argsT Type of arguments to forward into the functor call
   * @param params Parameters storage used for recursion
   * @param args Arguments to forward into the functor call
   */
  template <size_t nParams, typename T, typename ... argsT>
  typename std::enable_if<(nParams == 0), typename functorT::returnT>::type
    static inline with(const T& params, argsT ... args) {
      static_cast<void>(params);
      return functorT{}(args...);
  }

  /**
   * @brief Extract strict number of parameters from storage into the parameter
   *        pack and call functor with custom parameters followed by parameter
   *        pack.
   * @tparam nParams Number of params to use as parameter pack
   * @tparam T Type of params storage
   * @tparam argsT Type of arguments to forward into the next call
   * @param params Storage to extract parameters from
   * @param args Arguments to forward into the next call
   */
  template <size_t nParams, typename T, typename ... argsT>
  typename std::enable_if<(nParams > 0), typename functorT::returnT>::type
    static inline with(T&& params, argsT ... args) {
      return with<nParams-1>(params, args..., params[nParams-1]);
  }

  /**
   * @brief Extract strict number of parameters from storage into the parameter
   *        pack and call functor with parameter pack.
   * @tparam nParams Number of params to use as parameter pack
   * @tparam T Type of params storage
   * @param params Storage to extract parameters from
   */
  template <size_t nParams, typename T>
  typename std::enable_if<(nParams > 0), typename functorT::returnT>::type
    static inline with(T&& params) {
      return with<nParams-1>(params, params[nParams-1]);
  }
};

#endif  // __SYCLCTS_TESTS_COMMON_VARIADIC_H
