/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2020-2021 The Khronos Group Inc.
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_MACROS_H
#define __SYCLCTS_TESTS_COMMON_MACROS_H

#include <sycl/sycl.hpp>

#include <catch2/catch_test_macros.hpp>
#undef FAIL  // We define our own FAIL macro

#include "../../util/type_names.h"
#include "macro_utils.h"

#define TEST_FILE __FILE__
#define TEST_BUILD_DATE __DATE__
#define TEST_BUILD_TIME __TIME__

/** convert a parameter to a string
 */
#define TOSTRING(X) STRINGIFY(X)
#define STRINGIFY(X) #X

/*
 * Macros for the creation of the TEST_NAMESPACE macro, which will
 * always be TEST_NAME__. For example, given the test for context
 * creation, with
 * #define TEST_NAME context_constructors
 * these macros will do the equivalent of
 * #define TEST_NAMESPACE context_constructors__
 * This is nice because it's automatic and stops copy-paste errors.
 *
 * It works because NS_CAT hides the concatenation operation behind
 * a macro which allows expansion to take place. The NS_NAMESPACE()
 * macro is function-like and therefore expands its parameters before
 * substituting them into the macro body. The result is that the
 * expanded TEST_NAME is concatenated with two underscores.
 */
#define NS_CAT(a, b) a##b
#define NS_NAMESPACE(X) NS_CAT(X, __)
#define TEST_NAMESPACE NS_NAMESPACE(TEST_NAME)

/** helper function used to construct a typical test info
 *  structure
 */
inline void set_test_info(sycl_cts::util::test_base::info& out,
                          const std::string& name, const char* file) {
  out.m_name = name;
  out.m_file = file;
  out.m_buildDate = TEST_BUILD_DATE;
  out.m_buildTime = TEST_BUILD_TIME;
}

/**
 *
 */
inline void log_exception(sycl_cts::util::logger& log,
                          const sycl::exception& e) {
  // notify that an exception was thrown
  log.note("sycl exception caught");

  // log exception error string
  std::string what = e.what();
  if (!what.empty()) {
    log.note("what - " + what);
  }
}

/**
 * Explicitly marks a test case as failed.
 *
 * In most situations it is preferable to use an assertion macro (CHECK,
 * REQUIRE, ...) instead.
 *
 * Failure messages can utilize iostream-style streaming expressions:
 * FAIL("foo " << 123 << " bar").
 *
 * Note that this macro used to receive a util::logger instance as its first
 * parameter. This is no longer required and can be omitted.
 *
 * TODO: Replace with Catch2's macro once we've fully removed util::logger
 */
#define FAIL(msg_or_logger, ...)                                          \
  INTERNAL_CATCH_MSG("FAIL", Catch::ResultWas::ExplicitFailure,           \
                     Catch::ResultDisposition::Normal,                    \
                     msg_or_logger INTERNAL_CTS_IF(INTERNAL_CTS_HAS_ARGS( \
                         __VA_ARGS__))(<< __VA_ARGS__, << ""));

#if defined(SYCL_CTS_TEST_OPENCL_INTEROP)
#define CHECK_CL_SUCCESS(log, error) \
  ([&] {                             \
    CHECK(error == CL_SUCCESS);      \
    return error == CL_SUCCESS;      \
  })()
#else
#define CHECK_CL_SUCCESS(...) \
  CATCH_FAIL("OpenCL interop tests are not enabled ")
#endif

#define CHECK_VALUE(log, received, expected, index) \
  ([&] {                                            \
    INFO("For element " + std::to_string(index));   \
    CHECK(received == expected);                    \
    return received == expected;                    \
  })()

#define CHECK_VALUE_SCALAR(log, received, expected) \
  ([&] {                                            \
    CHECK(received == expected);                    \
    return received == expected;                    \
  })()

#define CHECK_TYPE(log, T, U)                                \
  ([&] {                                                     \
    INFO("For types " + type_name<decltype(T)>() + " and " + \
         type_name<decltype(U)>());                          \
    CHECK(typeid(T) == typeid(U));                           \
    return (typeid(T) == typeid(U));                         \
  })()

#define ASSERT_RETURN_TYPE(expectedT, returnVal, MSG) \
  { static_assert(std::is_same<decltype(returnVal), expectedT>::value, MSG); }

#endif  // __SYCLCTS_TESTS_COMMON_MACROS_H
