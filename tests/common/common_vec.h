/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_COMMON_VEC_H
#define __SYCLCTS_TESTS_COMMON_COMMON_VEC_H

// include our proxy to the real sycl header
#include "sycl.h"

#include "../../util/math_vector.h"
#include "../../util/proxy.h"
#include "../../util/test_base.h"
#include "../common/common.h"
#include "../common/cts_async_handler.h"
#include "../common/cts_selector.h"
#include "../common/get_cts_object.h"
#include "macros.h"

#include "../../util/accuracy.h"
#include "../../util/math_vector.h"
#include "../../util/test_base.h"
#include "../../util/type_traits.h"

#include <string>
#include <type_traits>

namespace {

/**
 * @brief Helper function to check the size of a vector is correct.
 */
template <typename vecType, int numOfElems>
bool check_vector_size(cl::sycl::vec<vecType, numOfElems> vector) {
  int count = (vector.get_count() == 3) ? 4 : vector.get_count();
  return ((sizeof(vecType) * count) == vector.get_size());
}

/**
 * @brief Helper function to check vector values are correct.
 */
template <typename vecType, int numOfElems>
bool check_vector_values(cl::sycl::vec<vecType, numOfElems> vector,
                         vecType* vals) {
  for (int i = 0; i < numOfElems; i++) {
    if ((vals[i] != getElement(vector, i))) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Helper function to check that vector floating-point values
 *        for division result are accurate enough
 */
template <typename vecType, int numOfElems>
typename std::enable_if<is_cl_float_type<vecType>::value, bool>::type
check_vector_values_div(cl::sycl::vec<vecType, numOfElems> vector,
                        vecType *vals) {
  for (int i = 0; i < numOfElems; i++) {
    vecType vectorValue = getElement(vector, i);
    if (vals[i] == vectorValue)
      continue;
    const vecType ulpsExpected = 2.5; // Min Accuracy for x / y
    const vecType difference = cl::sycl::fabs(vectorValue - vals[i]);
    // using sycl functions to get ulp because it used in kernel
    const vecType differenceExpected = ulpsExpected * get_ulp_sycl(vals[i]);

    if (difference > differenceExpected) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Helper function to check that vector values for division are correct
 */
template <typename vecType, int numOfElems>
typename std::enable_if<!is_cl_float_type<vecType>::value, bool>::type
check_vector_values_div(cl::sycl::vec<vecType, numOfElems> vector,
                        vecType *vals) {
  return check_vector_values(vector, vals);
}

/**
 *  @brief Helper function to test a single vector operator.
 */
template <int vecSize, typename vectorType, typename lambdaFunc>
bool check_single_vector_op(vectorType vector1, lambdaFunc lambda) {
  auto vector2 = lambda();
  if (!check_return_type_bool<vectorType>(vector2)) {
    return false;
  }
  if (!std::is_same<vectorType, decltype(vector2)>::value) {
    return false;
  }
  for (int i = 0; i < vecSize; i++) {
    if (getElement(vector1, i) != getElement(vector2, i)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 * convert()
 * as()
 * vec<dataT, numElement>(genvector clVector)
 */
template <typename vecType, typename convertType, typename asType>
bool check_vector_member_functions(cl::sycl::vec<vecType, 1> inputVec,
                                   vecType* vals) {
  // get_count()
  int count = inputVec.get_count();
  if (count != 1) {
    return false;
  }
  count = inputVec.template swizzle<cl::sycl::elem::s0>().get_count();
  if (count != 1) {
    return false;
  }

  // get_size()
  int size = inputVec.get_size();
  if (size != sizeof(vecType) * 1) {
    return false;
  }
  size = inputVec.template swizzle<cl::sycl::elem::s0>().get_size();
  if (size != sizeof(vecType) * 1) {
    return false;
  }

  // convert()
  cl::sycl::vec<convertType, 1> convertedVec =
      inputVec
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();
  convertedVec =
      inputVec.template swizzle<cl::sycl::elem::s0>()
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();

  // as()
  cl::sycl::vec<asType, 1> asVec =
      inputVec.template as<cl::sycl::vec<asType, 1>>();
  asVec = inputVec.template swizzle<cl::sycl::elem::s0>()
              .template as<cl::sycl::vec<asType, 1>>();
  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 * convert()
 * as()
 * vec<dataT, numElement>(genvector clVector)
 */
template <typename vecType, typename convertType, typename asType>
bool check_vector_member_functions(cl::sycl::vec<vecType, 2> inputVec,
                                   vecType* vals) {
  // get_count()
  int count = inputVec.get_count();
  if (count != 2) {
    return false;
  }
  count = inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
              .get_count();
  if (count != 2) {
    return false;
  }

  // get_size()
  int size = inputVec.get_size();
  if (size != sizeof(vecType) * 2) {
    return false;
  }
  size = inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
             .get_size();
  if (size != sizeof(vecType) * 2) {
    return false;
  }

  // convert()
  cl::sycl::vec<convertType, 2> convertedVec =
      inputVec
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();
  convertedVec =
      inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();

  // as()
  cl::sycl::vec<asType, 2> asVec =
      inputVec.template as<cl::sycl::vec<asType, 2>>();
  asVec = inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
              .template as<cl::sycl::vec<asType, 2>>();
  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 * convert()
 * as()
 * vec<dataT, numElement>(genvector clVector)
 */
template <typename vecType, typename convertType, typename asType>
bool check_vector_member_functions(cl::sycl::vec<vecType, 3> inputVec,
                                   vecType* vals) {
  // get_count()
  int count = inputVec.get_count();
  if (count != 3) {
    return false;
  }
  count = inputVec
              .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                cl::sycl::elem::s2>()
              .get_count();
  if (count != 3) {
    return false;
  }

  // get_size()
  int size = inputVec.get_size();
  if (size != sizeof(vecType) * 4) {
    return false;
  }
  size = inputVec
             .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                               cl::sycl::elem::s2>()
             .get_size();
  if (size != sizeof(vecType) * 4) {
    return false;
  }

  // convert()
  cl::sycl::vec<convertType, 3> convertedVec =
      inputVec
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();
  convertedVec =
      inputVec
          .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                            cl::sycl::elem::s2>()
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();

  // as()
  cl::sycl::vec<asType, 3> asVec =
      inputVec.template as<cl::sycl::vec<asType, 3>>();
  asVec = inputVec
              .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                cl::sycl::elem::s2>()
              .template as<cl::sycl::vec<asType, 3>>();
  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 * convert()
 * as()
 * vec<dataT, numElement>(genvector clVector)
 */
template <typename vecType, typename convertType, typename asType>
bool check_vector_member_functions(cl::sycl::vec<vecType, 4> inputVec,
                                   vecType* vals) {
  // get_count()
  int count = inputVec.get_count();
  if (count != 4) {
    return false;
  }
  count = inputVec
              .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                cl::sycl::elem::s2, cl::sycl::elem::s3>()
              .get_count();
  if (count != 4) {
    return false;
  }

  // get_size()
  int size = inputVec.get_size();
  if (size != sizeof(vecType) * 4) {
    return false;
  }
  size = inputVec
             .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                               cl::sycl::elem::s2, cl::sycl::elem::s3>()
             .get_size();
  if (size != sizeof(vecType) * 4) {
    return false;
  }

  // convert()
  cl::sycl::vec<convertType, 4> convertedVec =
      inputVec
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();
  convertedVec =
      inputVec
          .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                            cl::sycl::elem::s2, cl::sycl::elem::s3>()
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();

  // as()
  cl::sycl::vec<asType, 4> asVec =
      inputVec.template as<cl::sycl::vec<asType, 4>>();
  asVec = inputVec
              .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                cl::sycl::elem::s2, cl::sycl::elem::s3>()
              .template as<cl::sycl::vec<asType, 4>>();
  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 * convert()
 * as()
 * vec<dataT, numElement>(genvector clVector)
 */
template <typename vecType, typename convertType, typename asType>
bool check_vector_member_functions(cl::sycl::vec<vecType, 8> inputVec,
                                   vecType* vals) {
  // get_count()
  int count = inputVec.get_count();
  if (count != 8) {
    return false;
  }
  count = inputVec
              .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                cl::sycl::elem::s2, cl::sycl::elem::s3,
                                cl::sycl::elem::s4, cl::sycl::elem::s5,
                                cl::sycl::elem::s6, cl::sycl::elem::s7>()
              .get_count();
  if (count != 8) {
    return false;
  }

  // get_size()
  int size = inputVec.get_size();
  if (size != sizeof(vecType) * 8) {
    return false;
  }
  size = inputVec
             .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                               cl::sycl::elem::s2, cl::sycl::elem::s3,
                               cl::sycl::elem::s4, cl::sycl::elem::s5,
                               cl::sycl::elem::s6, cl::sycl::elem::s7>()
             .get_size();
  if (size != sizeof(vecType) * 8) {
    return false;
  }

  // convert()
  cl::sycl::vec<convertType, 8> convertedVec =
      inputVec
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();
  convertedVec =
      inputVec
          .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                            cl::sycl::elem::s2, cl::sycl::elem::s3,
                            cl::sycl::elem::s4, cl::sycl::elem::s5,
                            cl::sycl::elem::s6, cl::sycl::elem::s7>()
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();

  // as()
  cl::sycl::vec<asType, 8> asVec =
      inputVec.template as<cl::sycl::vec<asType, 8>>();
  asVec = inputVec
              .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                cl::sycl::elem::s2, cl::sycl::elem::s3,
                                cl::sycl::elem::s4, cl::sycl::elem::s5,
                                cl::sycl::elem::s6, cl::sycl::elem::s7>()
              .template as<cl::sycl::vec<asType, 8>>();
  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 * convert()
 * as()
 * vec<dataT, numElement>(genvector clVector)
 */
template <typename vecType, typename convertType, typename asType>
bool check_vector_member_functions(cl::sycl::vec<vecType, 16> inputVec,
                                   vecType* vals) {
  // get_count()
  int count = inputVec.get_count();
  if (count != 16) {
    return false;
  }
  count = inputVec
              .template swizzle<
                  cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                  cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                  cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
                  cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
                  cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
                  cl::sycl::elem::sF>()
              .get_count();
  if (count != 16) {
    return false;
  }

  // get_size()
  int size = inputVec.get_size();
  if (size != sizeof(vecType) * 16) {
    return false;
  }
  size = inputVec
             .template swizzle<
                 cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                 cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                 cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
                 cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
                 cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
                 cl::sycl::elem::sF>()
             .get_size();
  if (size != sizeof(vecType) * 16) {
    return false;
  }

  // convert()
  cl::sycl::vec<convertType, 16> convertedVec =
      inputVec
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();
  convertedVec =
      inputVec
          .template swizzle<
              cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
              cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
              cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
              cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
              cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
              cl::sycl::elem::sF>()
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();

  // as()
  cl::sycl::vec<asType, 16> asVec =
      inputVec.template as<cl::sycl::vec<asType, 16>>();
  asVec = inputVec
              .template swizzle<
                  cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                  cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                  cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
                  cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
                  cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
                  cl::sycl::elem::sF>()
              .template as<cl::sycl::vec<asType, 16>>();
  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * lo()
 * hi()
 * odd()
 * even()
 */
template <typename vecType>
bool check_lo_hi_odd_even(cl::sycl::vec<vecType, 2> inputVec, vecType* vals) {
  constexpr size_t mid = 1;
  // lo()
  {
    cl::sycl::vec<vecType, mid> loVec{inputVec.lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> loVec{
        inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
            .lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  // As the second element from hi() on 3 element vectors is undefined, don't
  // test it
  {
    // hi()
    cl::sycl::vec<vecType, mid> hiVec{inputVec.hi()};
    vecType hiVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      hiVals[i] = vals[i + mid];
    }
    if (!check_vector_values<vecType, mid>(hiVec, hiVals)) {
      return false;
    }
  }
  {
    // hi()
    cl::sycl::vec<vecType, mid> hiVec{
        inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
            .hi()};
    vecType hiVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      hiVals[i] = vals[i + mid];
    }
    if (!check_vector_values<vecType, mid>(hiVec, hiVals)) {
      return false;
    }
  }
  // As the second element from odd() on 3 element vectors is undefined, don't
  // test it
  {
    // odd()
    cl::sycl::vec<vecType, mid> oddVec{inputVec.odd()};
    vecType oddVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      oddVals[i] = vals[i * 2 + 1];
    }
    if (!check_vector_values<vecType, mid>(oddVec, oddVals)) {
      return false;
    }
  }
  {
    // odd()
    cl::sycl::vec<vecType, mid> oddVec{
        inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
            .odd()};
    vecType oddVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      oddVals[i] = vals[i * 2 + 1];
    }
    if (!check_vector_values<vecType, mid>(oddVec, oddVals)) {
      return false;
    }
  }
  // even()
  {
    cl::sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> evenVec{
        inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
            .even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }

  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * lo()
 * hi()
 * odd()
 * even()
 */
template <typename vecType>
bool check_lo_hi_odd_even(cl::sycl::vec<vecType, 3> inputVec, vecType* vals) {
  constexpr size_t mid = 2;
  // lo()
  {
    cl::sycl::vec<vecType, mid> loVec{inputVec.lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> loVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2>()
            .lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  // As the second element from hi() on 3 element vectors is undefined, don't
  // test it
  // As the second element from odd() on 3 element vectors is undefined, don't
  // test it
  // even()
  {
    cl::sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> evenVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2>()
            .even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }

  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * lo()
 * hi()
 * odd()
 * even()
 */
template <typename vecType>
bool check_lo_hi_odd_even(cl::sycl::vec<vecType, 4> inputVec, vecType* vals) {
  constexpr size_t mid = 2;
  // lo()
  {
    cl::sycl::vec<vecType, mid> loVec{inputVec.lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> loVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2, cl::sycl::elem::s3>()
            .lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  // As the second element from hi() on 3 element vectors is undefined, don't
  // test it
  {
    // hi()
    cl::sycl::vec<vecType, mid> hiVec{inputVec.hi()};
    vecType hiVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      hiVals[i] = vals[i + mid];
    }
    if (!check_vector_values<vecType, mid>(hiVec, hiVals)) {
      return false;
    }
  }
  {
    // hi()
    cl::sycl::vec<vecType, mid> hiVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2, cl::sycl::elem::s3>()
            .hi()};
    vecType hiVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      hiVals[i] = vals[i + mid];
    }
    if (!check_vector_values<vecType, mid>(hiVec, hiVals)) {
      return false;
    }
  }
  // As the second element from odd() on 3 element vectors is undefined, don't
  // test it
  {
    // odd()
    cl::sycl::vec<vecType, mid> oddVec{inputVec.odd()};
    vecType oddVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      oddVals[i] = vals[i * 2 + 1];
    }
    if (!check_vector_values<vecType, mid>(oddVec, oddVals)) {
      return false;
    }
  }
  {
    // odd()
    cl::sycl::vec<vecType, mid> oddVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2, cl::sycl::elem::s3>()
            .odd()};
    vecType oddVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      oddVals[i] = vals[i * 2 + 1];
    }
    if (!check_vector_values<vecType, mid>(oddVec, oddVals)) {
      return false;
    }
  }
  // even()
  {
    cl::sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> evenVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2, cl::sycl::elem::s3>()
            .even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }

  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * lo()
 * hi()
 * odd()
 * even()
 */
template <typename vecType>
bool check_lo_hi_odd_even(cl::sycl::vec<vecType, 8> inputVec, vecType* vals) {
  constexpr size_t mid = 4;
  // lo()
  {
    cl::sycl::vec<vecType, mid> loVec{inputVec.lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> loVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2, cl::sycl::elem::s3,
                              cl::sycl::elem::s4, cl::sycl::elem::s5,
                              cl::sycl::elem::s6, cl::sycl::elem::s7>()
            .lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  // As the second element from hi() on 3 element vectors is undefined, don't
  // test it
  {
    // hi()
    cl::sycl::vec<vecType, mid> hiVec{inputVec.hi()};
    vecType hiVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      hiVals[i] = vals[i + mid];
    }
    if (!check_vector_values<vecType, mid>(hiVec, hiVals)) {
      return false;
    }
  }
  {
    // hi()
    cl::sycl::vec<vecType, mid> hiVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2, cl::sycl::elem::s3,
                              cl::sycl::elem::s4, cl::sycl::elem::s5,
                              cl::sycl::elem::s6, cl::sycl::elem::s7>()
            .hi()};
    vecType hiVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      hiVals[i] = vals[i + mid];
    }
    if (!check_vector_values<vecType, mid>(hiVec, hiVals)) {
      return false;
    }
  }
  // As the second element from odd() on 3 element vectors is undefined, don't
  // test it
  {
    // odd()
    cl::sycl::vec<vecType, mid> oddVec{inputVec.odd()};
    vecType oddVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      oddVals[i] = vals[i * 2 + 1];
    }
    if (!check_vector_values<vecType, mid>(oddVec, oddVals)) {
      return false;
    }
  }
  {
    // odd()
    cl::sycl::vec<vecType, mid> oddVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2, cl::sycl::elem::s3,
                              cl::sycl::elem::s4, cl::sycl::elem::s5,
                              cl::sycl::elem::s6, cl::sycl::elem::s7>()
            .odd()};
    vecType oddVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      oddVals[i] = vals[i * 2 + 1];
    }
    if (!check_vector_values<vecType, mid>(oddVec, oddVals)) {
      return false;
    }
  }
  // even()
  {
    cl::sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> evenVec{
        inputVec
            .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                              cl::sycl::elem::s2, cl::sycl::elem::s3,
                              cl::sycl::elem::s4, cl::sycl::elem::s5,
                              cl::sycl::elem::s6, cl::sycl::elem::s7>()
            .even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }

  return true;
}

/**
 * @brief Helper function to test the following functions of a vec
 * lo()
 * hi()
 * odd()
 * even()
 */
template <typename vecType>
bool check_lo_hi_odd_even(cl::sycl::vec<vecType, 16> inputVec, vecType* vals) {
  constexpr size_t mid = 8;
  // lo()
  {
    cl::sycl::vec<vecType, mid> loVec{inputVec

                                          .lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> loVec{
        inputVec
            .template swizzle<
                cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
                cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
                cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
                cl::sycl::elem::sF>()
            .lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  // As the second element from hi() on 3 element vectors is undefined, don't
  // test it
  {
    // hi()
    cl::sycl::vec<vecType, mid> hiVec{inputVec.hi()};
    vecType hiVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      hiVals[i] = vals[i + mid];
    }
    if (!check_vector_values<vecType, mid>(hiVec, hiVals)) {
      return false;
    }
  }
  {
    // hi()
    cl::sycl::vec<vecType, mid> hiVec{
        inputVec
            .template swizzle<
                cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
                cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
                cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
                cl::sycl::elem::sF>()
            .hi()};
    vecType hiVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      hiVals[i] = vals[i + mid];
    }
    if (!check_vector_values<vecType, mid>(hiVec, hiVals)) {
      return false;
    }
  }
  // As the second element from odd() on 3 element vectors is undefined, don't
  // test it
  {
    // odd()
    cl::sycl::vec<vecType, mid> oddVec{inputVec.odd()};
    vecType oddVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      oddVals[i] = vals[i * 2 + 1];
    }
    if (!check_vector_values<vecType, mid>(oddVec, oddVals)) {
      return false;
    }
  }
  {
    // odd()
    cl::sycl::vec<vecType, mid> oddVec{
        inputVec
            .template swizzle<
                cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
                cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
                cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
                cl::sycl::elem::sF>()
            .odd()};
    vecType oddVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      oddVals[i] = vals[i * 2 + 1];
    }
    if (!check_vector_values<vecType, mid>(oddVec, oddVals)) {
      return false;
    }
  }
  // even()
  {
    cl::sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    cl::sycl::vec<vecType, mid> evenVec{
        inputVec
            .template swizzle<
                cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
                cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
                cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
                cl::sycl::elem::sF>()
            .even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }

  return true;
}
}

#endif  // __SYCLCTS_TESTS_COMMON_COMMON_VEC_H
