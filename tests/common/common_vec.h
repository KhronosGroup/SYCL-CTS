/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#pragma once

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

/**
 * @brief Helper function to check the size of a vector is correct.
 */
template <typename vecType, int numOfElems>
void check_vector_size(sycl_cts::util::logger &log,
                       cl::sycl::vec<vecType, numOfElems> vector) {
  if ((sizeof(vecType) * vector.get_count()) != vector.get_size()) {
    FAIL(log, cl::sycl::string_class("Vector was wrong size") +
                  "\nExpected size -> " +
                  cl::sycl::string_class(
                      std::to_string(sizeof(vecType) * vector.get_count())) +
                  "\nActual size -> " + std::to_string(vector.get_size()));
  }
}

/**
 * @brief Helper function to check vector values are correct.
 */
template <typename vecType, int numOfElems>
void check_vector_values(sycl_cts::util::logger &log,
                         cl::sycl::vec<vecType, numOfElems> vector,
                         vecType *vals) {
  for (int i = 0; i < numOfElems; i++) {
    if ((vals[i] != getElement(vector, i))) {
      FAIL(log, cl::sycl::string_class("Vector had incorrect value") +
                    "\nExpected value -> " + std::to_string(vals[i]) +
                    "\nActual value ->   " +
                    std::to_string(getElement(vector, i)));
    }
  }
}

/**
 *  @brief Helper function to test a single vector operator
 */
template <int vecSize, typename vectorType, typename lambdaFunc>
void check_single_vector_op(sycl_cts::util::logger &log, vectorType vector1,
                            lambdaFunc lambda, cl::sycl::string_class opName) {
  bool failed = false;
  auto vector2 = lambda();
  check_return_type<vectorType>(log, vector2, opName);
  if (std::is_same<vectorType, declType(vector2)>::value) {
    failed = true;
    FAIL(log, cl::sycl::string_class("Vectors are not of equal type\n"));
  }
  for (int i = 0; i < vecSize; i++) {
    if (getElement(vector1, i) != getElement(vector2, i)) {
      FAIL(log, cl::sycl::string_class("Vector has incorrect value\n") +
                    "Got " + std::to_string(getElement(vector2, i)) +
                    " at element " + std::to_string(i) + "\nExpected " +
                    std::to_string(getElement(vector1, i)));
      failed = true;
      break;
    }
  }
  if (failed) {
    FAIL(log, "Vector test for operator " + opName + " failed");
  }
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 * convert()
 * as()
 * genvector()
 * vec<dataT, numElement>(genvector clVector)
 */
template <typename vecType, int vecCount, typename convertType, typename asType>
void check_vector_member_functions(sycl_cts::util::logger &log,
                                   cl::sycl::vec<vecType, vecCount> inputVec,
                                   vecType *vals) {
  // get_count()
  int count = inputVec.get_count();
  if (count != vecCount) {
    FAIL(log, cl::sycl::string_class("failed get_count()"));
  }

  // get_size()
  int size = inputVec.get_size();
  if (size != sizeof(vecType) * vecCount) {
    FAIL(log, cl::sycl::string_class("failed get_size()"));
  }

  // convert()
  cl::sycl::vec<convertType, vecCount> convertedVec =
      inputVec
          .template convert<convertType, cl::sycl::rounding_mode::automatic>();

  // as()
  cl::sycl::vec<asType, vecCount> asVec =
      inputVec.template as<cl::sycl::vec<asType, vecCount>>();

  // genvector()
  auto genVector = inputVec.genvector();

  // vec<dataT, numElement>(genvector clVector)
  cl::sycl::vec<vecType, vecCount> syclVector =
      cl::sycl::vec<vecType, vecCount>(genVector);

  // swizzle()
  if (vecCount == 1) {
    auto swizzleVec = inputVec.template swizzle<1>();
    auto swizzleVecDetail = inputVec.template swizzle<cl::sycl::elem::s0>();
  }
}

/**
 * @brief Helper function to test the following functions of a vec
 * lo()
 * hi()
 * odd()
 * even()
 */
template <typename vecType,
          int vecCount check_lo_hi_odd_even(
              sycl_cts::util::logger &log,
              cl::sycl::vec<vecType, vecCount> inputVec, vecType *vals) {
  // lo()
  cl::sycl::vec<vecType, vecCount / 2 + vecCount % 2> loVec = inputVec.lo();

  vecType loVals[vecCount / 2 + vecCount % 2] = {0};
  for (int i = 0; i < (vecCount + vecCount % 2) / 2; i++) {
    loVals[i] = vals[i];
  }
  check_vector_values<vecType, vecCount / 2 + vecCount % 2>(log, loVec, loVals);
  // hi()
  cl::sycl::vec<vecType, vecCount / 2 + vecCount % 2> hiVec = inputVec.hi();

  vecType hiVals[vecCount / 2 + vecCount % 2] = {0};
  for (int i = (vecCount + vecCount % 2) / 2; i < vecCount; i++) {
    hiVals[i] = vals[i + vecCount + vecCount % 2];
  }
  check_vector_values<vecType, vecCount / 2 + vecCount % 2>(log, hiVec, hiVals);

  // odd()
  cl::sycl::vec<vecType, vecCount / 2 + vecCount % 2> oddVec = inputVec.odd();

  vecType oddVals[vecCount / 2 + vecCount % 2] = {0};
  for (int i = 0; i < vecCount / 2 + vecCount % 2; ++i) {
    oddVals[i] = vals[i * 2 + 1];
  }
  check_vector_values<vecType, vecCount / 2 + vecCount % 2>(log, oddVec,
                                                            oddVals);
  // even()
  cl::sycl::vec<vecType, vecCount / 2 + vecCount % 2> evenVec = inputVec.even();

  vecType evenVals[vecCount / 2 + vecCount % 2] = {0};
  for (int i = 0; i < vecCount / 2 + vecCount % 2; ++i) {
    evenVals[i] = vals[i * 2];
  }
  check_vector_values<vecType, vecCount / 2 + vecCount % 2>(log, evenVec,
                                                            evenVals);
}
