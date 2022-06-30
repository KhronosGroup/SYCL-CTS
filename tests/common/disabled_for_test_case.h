/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_DISABLED_FOR_TEST_CASE_H
#define __SYCLCTS_TESTS_COMMON_DISABLED_FOR_TEST_CASE_H

// This is required for detecting the active SYCL implementation
#include <sycl/sycl.hpp>

#include "macro_utils.h"

// TODO: Add other Catch2 test case variants, as needed

/**
 * Registers a test case that is compile-time disabled for one or more SYCL
 * implementations. This is useful for when a test case contains code that
 * currently does not compile for a given implementation, while other test cases
 * in the same translation unit would otherwise compile.
 *
 * The following implementations can be specified: ComputeCpp, DPCPP, hipSYCL.
 * A disabled test case will fail automatically at runtime.
 *
 * Usage example:
 * ```
 * DISABLED_FOR_TEST_CASE(hipSYCL, ComputeCpp)("my test case", "[my-tag]")({
 *  // ...
 * });
 * ```
 *
 * Apart from the initial list of implementations, the test case definition is
 * quite similar to a regular test case, with two additional caveats:
 *  1. The test case body needs to be wrapped in parentheses.
 *  2. Unlike for regular TEST_CASE, tags are non-optional and must be provided.
 */
#define DISABLED_FOR_TEST_CASE(...) \
  INTERNAL_CTS_DISABLED_FOR_TEST_CASE(__VA_ARGS__)

#define DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(...) \
  INTERNAL_CTS_DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(__VA_ARGS__)

// ------------------------------------------------------------------------------------

#if SYCL_CTS_COMPILING_WITH_COMPUTECPP
#define INTERNAL_CTS_SYCL_IMPL_ComputeCpp ()
#elif SYCL_CTS_COMPILING_WITH_DPCPP
#define INTERNAL_CTS_SYCL_IMPL_DPCPP ()
#elif SYCL_CTS_COMPILING_WITH_HIPSYCL
#define INTERNAL_CTS_SYCL_IMPL_hipSYCL ()
#else
#error Unknown SYCL implementation
#endif

// Clang 8 does not properly handle _Pragma inside macro expansions.
// See https://compiler-explorer.com/z/4WebY11TM.
// As a crude workaround we disable _Pragma altogether.
// This is required as of ComputeCpp 2.8.0, which is based on Clang 8.
#if defined(__clang__) && __clang_major__ <= 8
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#define _Pragma(...)
#pragma clang diagnostic pop
#endif

// ------------------------------------------------------------------------------------

#define INTERNAL_CTS_IMPL_PROBE(impl) \
  _INTERNAL_CTS_IMPL_PROBE_APPEND(    \
      INTERNAL_CTS_CAT(INTERNAL_CTS_SYCL_IMPL_, impl))
#define _INTERNAL_CTS_IMPL_PROBE_APPEND(x) _INTERNAL_CTS_IMPL_PROBE_EXPAND x
#define _INTERNAL_CTS_IMPL_PROBE_EXPAND(...) INTERNAL_CTS_PROBE()

// Returns 1 if INTERNAL_CTS_SYCL_IMPL_<impl> is set to `()`, 0 otherwise.
#define INTERNAL_CTS_IS_IMPL(impl) \
  INTERNAL_CTS_CHECK(INTERNAL_CTS_IMPL_PROBE(impl))

// ------------------------------------------------------------------------------------

// Helper macros for expanding recursive calls.
// This allows us to check for up to 16 + 1 SYCL implementations, which should
// be plenty. (Otherwise this could always be extended by another factor of
// two).
#define INTERNAL_CTS_EVAL(...) INTERNAL_CTS_EVAL16(__VA_ARGS__)
#define INTERNAL_CTS_EVAL16(...) \
  INTERNAL_CTS_EVAL8(INTERNAL_CTS_EVAL8(__VA_ARGS__))
#define INTERNAL_CTS_EVAL8(...) \
  INTERNAL_CTS_EVAL4(INTERNAL_CTS_EVAL4(__VA_ARGS__))
#define INTERNAL_CTS_EVAL4(...) \
  INTERNAL_CTS_EVAL2(INTERNAL_CTS_EVAL2(__VA_ARGS__))
#define INTERNAL_CTS_EVAL2(...) __VA_ARGS__

#define INTERNAL_CTS_EMPTY()
// Defers macro expansion to allow for recursion (prevent macros from being
// "painted blue"). Depending on the nesting level of the recursive call, we
// need multiple defers (3 in this case).
#define INTERNAL_CTS_DEFER3(x) \
  x INTERNAL_CTS_EMPTY INTERNAL_CTS_EMPTY INTERNAL_CTS_EMPTY()()()

// ------------------------------------------------------------------------------------

// In case we are compiling with a disabled implementation,
// replace test case with this auto-failing proxy.
// Note that we explicitly receive a description and tags here,
// so we can use the same macro for all types of test cases,
// including those that receive additional parameters.
// A downside of this is that we require test cases to provide
// tags, which normally would be optional.
#define INTERNAL_CTS_DISABLED_TEST_CASE(description, tags, ...) \
  TEST_CASE(description, tags) {                                \
    FAIL("This test case has been compile-time disabled.");     \
  }                                                             \
  _INTERNAL_CTS_DISCARD
#define _INTERNAL_CTS_DISCARD(...)

#define INTERNAL_CTS_ENABLED_TEST_CASE_BODY(...) \
  { __VA_ARGS__; }

#define INTERNAL_CTS_CHECK_ALL_IMPLS(catchMacroProxy, impl, ...)             \
  INTERNAL_CTS_IF(INTERNAL_CTS_IS_IMPL(impl))                                \
  (INTERNAL_CTS_DISABLED_TEST_CASE,                                          \
   INTERNAL_CTS_IF(INTERNAL_CTS_HAS_ARGS(__VA_ARGS__))(                      \
       INTERNAL_CTS_DEFER3(_INTERNAL_CTS_CHECK_ALL_IMPLS)()(catchMacroProxy, \
                                                            __VA_ARGS__),    \
       catchMacroProxy))
#define _INTERNAL_CTS_CHECK_ALL_IMPLS() INTERNAL_CTS_CHECK_ALL_IMPLS

#define INTERNAL_CTS_MAYBE_DISABLE_TEST_CASE(catchMacroProxy, ...) \
  INTERNAL_CTS_EVAL(INTERNAL_CTS_CHECK_ALL_IMPLS(catchMacroProxy, __VA_ARGS__))

// Define proxies for all supported test macro types

#define INTERNAL_CTS_ENABLED_TEST_CASE(...) \
  TEST_CASE(__VA_ARGS__) INTERNAL_CTS_ENABLED_TEST_CASE_BODY
#define INTERNAL_CTS_DISABLED_FOR_TEST_CASE(...)                       \
  INTERNAL_CTS_MAYBE_DISABLE_TEST_CASE(INTERNAL_CTS_ENABLED_TEST_CASE, \
                                       __VA_ARGS__)

#define INTERNAL_CTS_ENABLED_TEMPLATE_TEST_CASE_SIG(...) \
  TEMPLATE_TEST_CASE_SIG(__VA_ARGS__) INTERNAL_CTS_ENABLED_TEST_CASE_BODY
#define INTERNAL_CTS_DISABLED_FOR_TEMPLATE_TEST_CASE_SIG(...) \
  INTERNAL_CTS_MAYBE_DISABLE_TEST_CASE(                       \
      INTERNAL_CTS_ENABLED_TEMPLATE_TEST_CASE_SIG, __VA_ARGS__)

#endif
