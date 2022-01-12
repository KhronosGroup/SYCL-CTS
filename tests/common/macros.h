/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_MACROS_H
#define __SYCLCTS_TESTS_COMMON_MACROS_H

#include <sycl/sycl.hpp>

#include "../../util/opencl_helper.h"
#include "../../util/type_names.h"

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

namespace {

/** helper function used to construct a typical test info
 *  structure
 */
inline void set_test_info(sycl_cts::util::test_base::info &out,
                          const std::string &name,
                          const char *file) {
  out.m_name = name;
  out.m_file = file;
}

/**
 *
 */
inline void log_exception(sycl_cts::util::logger &log,
                          const sycl::exception &e) {
  // notify that an exception was thrown
  log.note("sycl exception caught");

  // log exception error string
  std::string what = e.what();
  if (!what.empty()) {
    log.note("what - " + what);
  }
}

/* helper function for test failure cases */
template <typename string_t>
inline bool fail_proxy(sycl_cts::util::logger &log, string_t&& msg, int line) {
  log.fail(msg, line);
  return false;
}

// Note: fail_proxy has a return type and can be used as expression in code
// base.
//             conversion to braces turns into statement and breaks these areas
//             of code.
#define FAIL(LOG, MSG) (fail_proxy(LOG, MSG, __LINE__))

/* proxy to the check_cl_success function */
inline bool check_cl_success_proxy(sycl_cts::util::logger &log, int error,
                                   int line) {
  using sycl_cts::util::get;
  using sycl_cts::util::opencl_helper;
  return get<opencl_helper>().check_cl_success(log, error, line);
}
#define CHECK_CL_SUCCESS(LOG, ERROR) \
  check_cl_success_proxy(LOG, ERROR, __LINE__)

/* macro to check if provided value is equal to expected value */
template <typename T1, typename T2>
bool check_value_proxy(sycl_cts::util::logger &log, const T1 &got,
                       const T2 &expected, std::string gotStr,
                       std::string expectedStr, int element,
                       int line, const bool useElement = true) {

  if (got != expected) {
    std::string msg = "Expected " + std::to_string(expected) + " {" +
                       expectedStr + "} but got " + std::to_string(got) + " {" +
                       gotStr + "}";
    if (useElement) {
      msg += " at element " + std::to_string(element);
    }
    fail_proxy(log, msg.c_str(), line);
    /* values are different */
    return false;
  }
  /* values are equal */
  return true;
}
#define CHECK_VALUE(LOG, GOT, EXPECTED, INDEX) \
  check_value_proxy(LOG, GOT, EXPECTED, #GOT, #EXPECTED, INDEX, __LINE__)

#define CHECK_VALUE_SCALAR(LOG, GOT, EXPECTED) \
  check_value_proxy(LOG, GOT, EXPECTED, #GOT, #EXPECTED, 0, __LINE__, false)

/* verify that val_a and val_b are of the same type */
template <typename T1, typename T2>
bool check_type_proxy(sycl_cts::util::logger &log, const T1 &val_a,
                      const T2 &val_b, int line) {

  if (typeid(val_a) != typeid(val_b)) {
    std::string msg = "Type mismatch between " +
                       type_name<T1>() + " and " +
                       type_name<T2>();
    fail_proxy(log, msg.c_str(), line);
    return false;
  }
  return true;
}
#define CHECK_TYPE(LOG, TYPE_A, TYPE_B) \
  check_type_proxy(LOG, TYPE_A, TYPE_B, __LINE__)

#define ASSERT_RETURN_TYPE(expectedT, returnVal, MSG) \
  { static_assert(std::is_same<decltype(returnVal), expectedT>::value, MSG); }

} /* namespace {} */

#endif  // __SYCLCTS_TESTS_COMMON_MACROS_H
