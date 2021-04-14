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
 * @details The intent is to use a subset of collection to generate a given
 *          number of the functor parameters.
 *          Let's say we have:
 *          - events: an array of SYCL device_events
 *          - run_wait_for: some functor which can wait on any number of
 *            events provided as template parameter pack
 *
 *          If we wanted to call this functor for a single event, we could use
 *
 *              run_wait_for{}(events[0])
 *
 *          If we wanted to call this functor for 32 events, we could use
 *
 *              run_wait_for{}(events[0], events[1], ... events[31])
 *
 *          Using this class we can do it in a uniform way for any constexpr
 *          number of events:
 *
 *              run_variadic<run_wait_for>::with<1>(events)
 *              run_variadic<run_wait_for>::with<32>(events)
 *
 *          It also works with template instantiations and additional parameters
 *          for the functor call; so instead of calling
 *
 *              run_wait_for<args>{}(a, b, events[0], events[1], ... events[31])
 *
 *          we can use
 *
 *              run_variadic<run_wait_for<args>>::with<32>(events, a, b)
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
