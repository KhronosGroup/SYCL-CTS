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
      inputVec.template convert<cl::sycl::vec<convertType, vecCount>,
                                cl::sycl::rounding_mode::automatic>();

  // as()
  cl::sycl::vec<asType, vecCount> asVec =
      inputVec.template as<cl::sycl::vec<asType, vecCount>>();
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
  constexpr size_t mid = ((vecCount / 2) + (vecCount % 2));
  // lo()
  cl::sycl::vec<vecType, mid> loVec{inputVec.lo()};
  vecType loVals[mid] = {0};
  for (size_t i = 0; i < mid; i++) {
    loVals[i] = vals[i];
  }
  if (!check_vector_values<vecType, mid>(loVec, loVals)) {
    return false;
  }
  // As the second element from hi() on 3 element vectors is undefined, don't
  // test it
  if (vecCount != 3) {
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
  // As the second element from odd() on 3 element vectors is undefined, don't
  // test it
  if (vecCount != 3) {
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
  // even()
  cl::sycl::vec<vecType, mid> evenVec{inputVec.even()};
  vecType evenVals[mid] = {0};
  for (size_t i = 0; i < mid; ++i) {
    evenVals[i] = vals[i * 2];
  }
  if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
    return false;
  }

  return true;
}
}

#endif  // __SYCLCTS_TESTS_COMMON_COMMON_VEC_H