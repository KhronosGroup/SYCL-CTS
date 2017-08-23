/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#pragma once

// include our proxy to the real sycl header
#include "sycl.h"

// test framework specific device selector
#include "../common/cts_selector.h"
#include "../common/cts_async_handler.h"
#include "../common/get_cts_object.h"
#include "../../util/proxy.h"
#include "macros.h"

#include "../../util/opencl_helper.h"
#include "../../util/test_base.h"
#include "../../util/test_base_opencl.h"
#include "../../util/math_vector.h"

#include <string>

/**
 * @brief Helper function to check the return type of a function.
 */
template <typename expectedT, typename returnT>
void check_return_type(sycl_cts::util::logger &log, returnT returnVal,
                       cl::sycl::string_class functionName) {
  if (typeid(returnT) != typeid(expectedT)) {
    FAIL(log, functionName + "has incorrect return type -> " +
                  cl::sycl::string_class(typeid(returnT).name()));
  }
}

/**
 * @brief Helper function to check two types are equal.
 */
template <typename expectedT, typename actualT>
void check_equal_type(sycl_cts::util::logger &log, actualT actualVal,
                      cl::sycl::string_class logMsg) {
  if (typeid(expectedT) != typeid(actualT)) {
    FAIL(log, logMsg + "\nGot type -> " +
                  cl::sycl::string_class(typeid(actualT).name()) +
                  "\nExpected type -> " +
                  cl::sycl::string_class(typeid(expectedT).name()));
  }
}

/**
 * @brief Helper function to check for the existence of an enum class value.
 */
template <typename enumT>
void check_enum_class_value(enumT value) {
  enumT tmp = value;
}

/**
 * @brief Helper function to check an enum is of the correct underlying type.
 */
template <typename enumT, typename underlyingT>
void check_enum_underlying_type(sycl_cts::util::logger &log) {
  if (typeid(typename std::underlying_type<enumT>::type) !=
      typeid(underlyingT)) {
    FAIL(log, cl::sycl::string_class(
                  typeid(typename std::underlying_type<enumT>::type).name()) +
                  " enum underlying type is not " +
                  cl::sycl::string_class(typeid(underlyingT).name()));
  }
}

/**
 * @brief Helper function to check an info parameter.
 */
template <typename enumT, typename returnT, enumT kValue, typename objectT>
void check_get_info_param(sycl_cts::util::logger &log, const objectT &object) {
  /** check param_traits return type
  */
  using paramTraitsType =
      typename cl::sycl::info::param_traits<enumT, kValue>::return_type;
  if (typeid(paramTraitsType) != typeid(returnT)) {
    FAIL(log, "param_trait specialization has incorrect return type");
  }

  /** check get_info return type
  */
  auto returnValue = object.template get_info<kValue>();
  check_return_type<returnT>(log, returnValue, "object::get_info()");
}

/**
 * @brief Helper function to check the equality of two SYCL objects.
 */
template <typename T>
void check_equality(sycl_cts::util::logger &log, T &a, T &b, bool checkGet) {
  /** check is_host
  */
  if (a.is_host() != b.is_host()) {
    FAIL(log, "two objects are not equal (is_host)");
  }

  /** check get
  */
  if (checkGet) {
    if (a.get() != b.get()) {
      FAIL(log, "two objects are not equal (get)");
    }
  }
};

/**
 * @brief Helper function to test two arrays have equal elements
 */
template <typename arrT, int size>
void check_array_equality(sycl_cts::util::logger &log, arrT *arr1, arrT *arr2) {
  for (int i = 0; i < size; i++) {
    if (arr1[i] != arr2[i]) {
      FAIL(log, "arrays are not equal");
    }
  }
}

#define TEST_TYPE_TRAIT(syclType, param, syclObject)                    \
  if (typeid(cl::sycl::info::param_traits<                              \
             cl::sycl::info::syclType,                                  \
             cl::sycl::info::syclType::param>::return_type) !=          \
      typeid(syclObject.get_info<cl::sycl::info::syclType::param>())) { \
    FAIL(log, #syclType                                                 \
         ".get_info<cl::sycl::info::" #syclType "::" #param             \
         ">() does not return "                                         \
         "cl::sycl::info::param_traits<cl::sycl::info::" #syclType      \
         ", cl::sycl::info::" #syclType "::" #param ">::return_type");  \
  }
