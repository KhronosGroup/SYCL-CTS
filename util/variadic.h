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
#include <utility>

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
 *
 *          Also we are able to use a selected subset of events
 *
 *              run_wait_for{}(events[1], events[3], events[7])
 *
 *          by using std::integer_sequence like
 *
 *              std::integer_sequence<std::size_t, 1, 3, 7> selected;
 *              run_variadic<run_wait_for>::with(selected, events);
 */
template <class functorT>
struct run_variadic {
  /**
   * @brief Extract specific parameters from storage into the parameter pack and
   *        call functor with custom parameters followed by parameter pack.
   * @tparam T Type of params storage
   * @tparam argsT Type of arguments to forward into the functor call
   * @tparam Is Index sequence to use over params storage
   * @param params Storage to extract parameters from
   * @param args Custom arguments to forward into the functor call
   */
  template <typename T, typename ... argsT, size_t ... Is>
  static inline auto with(std::index_sequence<Is...>, const T& params,
                          argsT&&... args) {
      return functorT{}(std::forward<argsT>(args)..., params[Is]...);
  }

  /**
   * @brief Extract strict number of parameters from storage into the parameter
   *        pack and call functor with custom parameters followed by parameter
   *        pack.
   * @tparam nParams Number of params from the beginning of the storage to use
   *                 as parameter pack
   * @tparam T Type of params storage
   * @tparam argsT Type of arguments to forward into the next call
   * @param params Storage to extract parameters from
   * @param args Custom arguments to forward into the functor call
   */
  template <size_t nParams, typename T, typename ... argsT>
  static inline auto with(const T& params, argsT&& ... args) {
      return with(std::make_index_sequence<nParams>{}, params,
                  std::forward<argsT>(args)...);
  }
};

#endif  // __SYCLCTS_TESTS_COMMON_VARIADIC_H
