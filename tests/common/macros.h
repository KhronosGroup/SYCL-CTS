/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2020-2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_MACROS_H
#define __SYCLCTS_TESTS_COMMON_MACROS_H

#include <catch2/catch_test_macros.hpp>
#undef FAIL  // We define our own FAIL macro

#include "../../util/logger.h"
#include "../../util/test_base.h"
#include "../../util/type_names.h"
#include "macro_utils.h"

#include <string>

#define TEST_FILE __FILE__

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

/**
 * Adds a placeholder test case that should be implemented in the future.
 *
 * A description must be provided, optionally tags can be provided as well.
 * Example usage:
 *
 * TODO_TEST_CASE("my feature works as expected", "[my-feature]");
 *
 * The test case is marked with the [todo] tag, so it can be listed on the
 * command line.
 */
#define TODO_TEST_CASE(description, ...)                      \
  TEST_CASE(description, __VA_ARGS__ "[todo][!shouldfail]") { \
    FAIL("This test case is not yet implemented.");           \
  }

#if SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS
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
