/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_COMMON_VEC_H
#define __SYCLCTS_TESTS_COMMON_COMMON_VEC_H

// include our proxy to the real sycl header
#include "sycl.h"

// test framework specific device selector
#include "../common/common.h"
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
#include <type_traits>

namespace {

/**
 * @brief Helper function to check the size of a vector is correct.
 */
template <typename vecType, int numOfElems>
bool check_vector_size(cl::sycl::vec<vecType, numOfElems> vector) {
  return ((sizeof(vecType) * vector.get_count()) == vector.get_size());
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
 *  @brief Helper function to test a single vector operator.
 */
template <int vecSize, typename vectorType, typename lambdaFunc>
bool check_single_vector_op(vectorType vector1, lambdaFunc lambda) {
  auto vector2 = lambda();
  if (check_return_type_bool<vectorType>(vector2)) {
    return false;
  }
  if (std::is_same<vectorType, decltype(vector2)>::value) {
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
template <typename vecType, int vecCount, typename convertType, typename asType>
bool check_vector_member_functions(cl::sycl::vec<vecType, vecCount> inputVec,
                                   vecType* vals) {
  // get_count()
  int count = inputVec.get_count();
  if (count != vecCount) {
    return false;
  }

  // get_size()
  int size = inputVec.get_size();
  if (size != sizeof(vecType) * vecCount) {
    return false;
  }

  // convert()
  cl::sycl::vec<convertType, vecCount> convertedVec =
      inputVec
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();

  // as()
  cl::sycl::vec<asType, vecCount> asVec =
      inputVec.template as<asType, vecCount>();

  // swizzle()
  if (vecCount == 1) {
    auto swizzleVec = inputVec.template swizzle<1>();
    auto swizzleVecDetail = inputVec.template swizzle<cl::sycl::elem::s0>();
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
template <typename vecType, int vecCount>
bool check_lo_hi_odd_even(cl::sycl::vec<vecType, vecCount> inputVec,
                          vecType* vals) {
  constexpr size_t stdIndex = ((vecCount / 2) + (vecCount % 2));
  constexpr size_t oddIndex = vecCount / 2;
  constexpr size_t length = ((vecCount + vecCount % 2) / 2);
  // lo()
  cl::sycl::vec<vecType, stdIndex> loVec = inputVec.lo();
  vecType loVals[stdIndex] = {0};
  for (int i = 0; i < length; i++) {
    loVals[i] = vals[i];
  }
  if (!check_vector_values<vecType, stdIndex>(loVec, loVals)) {
    return false;
  }
  // hi()
  cl::sycl::vec<vecType, stdIndex> hiVec = inputVec.hi();
  vecType hiVals[stdIndex] = {0};
  for (int i = length; i < vecCount; i++) {
    hiVals[i] = vals[i + vecCount + vecCount % 2];
  }
  if (!check_vector_values<vecType, stdIndex>(hiVec, hiVals)) {
    return false;
  }
  // odd()
  cl::sycl::vec<vecType, oddIndex> oddVec = inputVec.odd();
  vecType oddVals[oddIndex] = {0};
  for (int i = 0; i < oddIndex; ++i) {
    oddVals[i] = vals[i * 2 + 1];
  }
  if (!check_vector_values<vecType, oddIndex>(oddVec, oddVals)) {
    return false;
  }
  // even()
  cl::sycl::vec<vecType, stdIndex> evenVec = inputVec.even();
  vecType evenVals[stdIndex] = {0};
  for (int i = 0; i < stdIndex; ++i) {
    evenVals[i] = vals[i * 2];
  }
  if (!check_vector_values<vecType, stdIndex>(evenVec, evenVals)) {
    return false;
  }
  return true;
}
}

#endif  // __SYCLCTS_TESTS_COMMON_COMMON_VEC_H