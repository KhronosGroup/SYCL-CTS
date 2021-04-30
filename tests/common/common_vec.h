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

#include "../../util/accuracy.h"
#include "../../util/math_vector.h"
#include "../../util/math_vector.h"
#include "../../util/proxy.h"
#include "../../util/test_base.h"
#include "../../util/type_traits.h"
#include "../common/common.h"
#include "../common/cts_async_handler.h"
#include "../common/cts_selector.h"
#include "../common/get_cts_object.h"
#include "macros.h"

#include <cstdint>
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
 */
template <typename vecType>
bool check_vector_get_count_get_size(cl::sycl::vec<vecType, 1> inputVec) {
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
  return true;
}
template <typename T1, typename T2>
T2 float_map_match(T1 *floats, T2 *vals, int size, T1 src) {
  for (int i = 0; i < size; ++i) {
    if (floats[i] == src) {
      return vals[i];
    }
  }
  return T2{};
}

template <typename T> struct is_unsigned_type {
  static constexpr bool value = std::is_unsigned_v<T> ||
                                std::is_same_v<T, cl::sycl::cl_uchar> ||
                                std::is_same_v<T, cl::sycl::cl_uint> ||
                                std::is_same_v<T, cl::sycl::cl_ulong> ||
                                std::is_same_v<T, cl::sycl::cl_ushort>;
};

template <typename sourceType, typename targetType, typename retType>
using enableIfFPtoNonFPConv = typename std::enable_if<
    is_cl_float_type<sourceType>::value &&
        !is_cl_float_type<targetType>::value,
    retType>::type;

template <typename sourceType, typename targetType, typename retType>
using enableIfNotFPtoNonFPConv = typename std::enable_if<
    !(is_cl_float_type<sourceType>::value &&
      !is_cl_float_type<targetType>::value),
    retType>::type;

template <typename sourceType, typename targetType, typename retType>
using enableIfFPtoUnsignedConv = typename std::enable_if<
    is_cl_float_type<sourceType>::value &&
        is_unsigned_type<targetType>::value,
    retType>::type;

template <typename sourceType, typename targetType, typename retType>
using enableIfNotFPtoUnsignedConv = typename std::enable_if<
    !(is_cl_float_type<sourceType>::value &&
      is_unsigned_type<targetType>::value),
    retType>::type;

template <typename vecType, int N, typename convertType>
cl::sycl::vec<convertType, N> convert_vec(cl::sycl::vec<vecType, N> inputVec) {
  cl::sycl::vec<convertType, N> resVec;
  for (size_t i = 0; i < N; ++i) {
    vecType elem = getElement(inputVec, i);
    setElement<convertType, N>(resVec, i, convertType(elem));
  }
  return resVec;
}

// rte
template <typename vecType, int N, typename convertType>
enableIfFPtoNonFPConv<vecType, convertType, cl::sycl::vec<convertType, N>>
rte(cl::sycl::vec<vecType, N> inputVec) {
  const int size = 8;
  vecType floats[size] = {2.3f, 3.8f, 1.5f, 2.5f, -2.3f, -3.8f, -1.5f, -2.5f};
  convertType vals[size] = {2, 4, 2, 2,
                            static_cast<convertType>(-2),
                            static_cast<convertType>(-4),
                            static_cast<convertType>(-2),
                            static_cast<convertType>(-2)};
  cl::sycl::vec<convertType, N> resVec;
  for (size_t i = 0; i < N; ++i) {
    vecType elem = getElement(inputVec, i);
    auto elemConvert = float_map_match(floats, vals, size, elem);
    setElement<convertType, N>(resVec, i, elemConvert);
  }
  return resVec;
}

template <typename vecType, int N, typename convertType>
enableIfNotFPtoNonFPConv<vecType, convertType, cl::sycl::vec<convertType, N>>
rte(cl::sycl::vec<vecType, N> inputVec) {
  return convert_vec<vecType, N, convertType>(inputVec);
}

// rtz
template <typename vecType, int N, typename convertType>
enableIfFPtoNonFPConv<vecType, convertType, cl::sycl::vec<convertType, N>>
rtz(cl::sycl::vec<vecType, N> inputVec) {
  const int size = 8;
  vecType floats[size] = {2.3f, 3.8f, 1.5f, 2.5f, -2.3f, -3.8f, -1.5f, -2.5f};
  convertType vals[size] = {2, 3, 1, 2,
                            static_cast<convertType>(-2),
                            static_cast<convertType>(-3),
                            static_cast<convertType>(-1),
                            static_cast<convertType>(-2)};
  cl::sycl::vec<convertType, N> resVec;
  for (size_t i = 0; i < N; ++i) {
    vecType elem = getElement(inputVec, i);
    auto elemConvert = float_map_match(floats, vals, size, elem);
    setElement<convertType, N>(resVec, i, elemConvert);
  }
  return resVec;
}

template <typename vecType, int N, typename convertType>
enableIfNotFPtoNonFPConv<vecType, convertType, cl::sycl::vec<convertType, N>>
rtz(cl::sycl::vec<vecType, N> inputVec) {
  return convert_vec<vecType, N, convertType>(inputVec);
}

// rtp
template <typename vecType, int N, typename convertType>
enableIfFPtoNonFPConv<vecType, convertType, cl::sycl::vec<convertType, N>>
rtp(cl::sycl::vec<vecType, N> inputVec) {
  const int size = 8;
  vecType floats[size] = {2.3f, 3.8f, 1.5f, 2.5f, -2.3f, -3.8f, -1.5f, -2.5f};
  convertType vals[size] = {3, 4, 2, 3,
                            static_cast<convertType>(-2),
                            static_cast<convertType>(-3),
                            static_cast<convertType>(-1),
                            static_cast<convertType>(-2)};
  cl::sycl::vec<convertType, N> resVec;
  for (size_t i = 0; i < N; ++i) {
    vecType elem = getElement(inputVec, i);
    auto elemConvert = float_map_match(floats, vals, size, elem);
    setElement<convertType, N>(resVec, i, elemConvert);
  }
  return resVec;
}
template <typename vecType, int N, typename convertType>
enableIfNotFPtoNonFPConv<vecType, convertType, cl::sycl::vec<convertType, N>>
rtp(cl::sycl::vec<vecType, N> inputVec) {
  return convert_vec<vecType, N, convertType>(inputVec);
}

// rtn
template <typename vecType, int N, typename convertType>
enableIfFPtoNonFPConv<vecType, convertType, cl::sycl::vec<convertType, N>>
rtn(cl::sycl::vec<vecType, N> inputVec) {
  const int size = 8;
  vecType floats[size] = {2.3f, 3.8f, 1.5f, 2.5f, -2.3f, -3.8f, -1.5f, -2.5f};
  convertType vals[size] = {2, 3, 1, 2,
                            static_cast<convertType>(-3),
                            static_cast<convertType>(-4),
                            static_cast<convertType>(-2),
                            static_cast<convertType>(-3)};
  cl::sycl::vec<convertType, N> resVec;
  for (size_t i = 0; i < N; ++i) {
    vecType elem = getElement(inputVec, i);
    auto elemConvert = float_map_match(floats, vals, size, elem);
    setElement<convertType, N>(resVec, i, elemConvert);
  }
  return resVec;
}
template <typename vecType, int N, typename convertType>
enableIfNotFPtoNonFPConv<vecType, convertType, cl::sycl::vec<convertType, N>>
rtn(cl::sycl::vec<vecType, N> inputVec) {
  return convert_vec<vecType, N, convertType>(inputVec);
}

template <typename vecType, int N, typename convertType>
enableIfNotFPtoUnsignedConv<vecType, convertType, void>
handleFPToUnsignedConv(cl::sycl::vec<vecType, N> &inputVec) {}

// Converting floating point values outside of (-1, max unsigned integer type
// value + 1) to unsigned integer types is undefined behaviour. Since the
// initial vectors contain negative values, check conversion of their absolute
// values instead.
template <typename vecType, int N, typename convertType>
enableIfFPtoUnsignedConv<vecType, convertType, void>
handleFPToUnsignedConv(cl::sycl::vec<vecType, N> &inputVec) {
  for (size_t i = 0; i < N; ++i) {
    vecType elem = getElement(inputVec, i);
    if (elem < 0)
      setElement<vecType, N>(inputVec, i, -elem);
  }
}

template <typename vecType, int N, typename convertType,
          cl::sycl::rounding_mode mode>
bool check_vector_convert(cl::sycl::vec<vecType, N> inputVec) {
  handleFPToUnsignedConv<vecType, N, convertType>(inputVec);
  cl::sycl::vec<convertType, N> convertedVec =
      inputVec.template convert<convertType, mode>();

  cl::sycl::vec<convertType, N> expectedVec;
  switch (mode) {
  case cl::sycl::rounding_mode::automatic:
    expectedVec = convert_vec<vecType, N, convertType>(inputVec);
    break;
  case cl::sycl::rounding_mode::rte:
    expectedVec = rte<vecType, N, convertType>(inputVec);
    break;
  case cl::sycl::rounding_mode::rtz:
    expectedVec = rtz<vecType, N, convertType>(inputVec);
    break;
  case cl::sycl::rounding_mode::rtp:
    expectedVec = rtp<vecType, N, convertType>(inputVec);
    break;
  case cl::sycl::rounding_mode::rtn:
    expectedVec = rtn<vecType, N, convertType>(inputVec);
    break;
  }
  if (!check_equal_values(convertedVec, expectedVec)) {
    return false;
  }
  return true;
}

template <typename vecType, int N, typename convertType>
bool check_vector_convert_modes(cl::sycl::vec<vecType, N> inputVec) {
  bool flag = true;
  flag &= check_vector_convert<vecType, N, convertType,
                               cl::sycl::rounding_mode::automatic>(inputVec);
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  flag &= check_vector_convert<vecType, N, convertType,
                               cl::sycl::rounding_mode::rte>(inputVec);
  flag &= check_vector_convert<vecType, N, convertType,
                               cl::sycl::rounding_mode::rtz>(inputVec);
  flag &= check_vector_convert<vecType, N, convertType,
                               cl::sycl::rounding_mode::rtp>(inputVec);
  flag &= check_vector_convert<vecType, N, convertType,
                               cl::sycl::rounding_mode::rtn>(inputVec);
#endif // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return flag;
}

/**
 * @brief Helper function to test the convert() function of a vec
 */
template <typename vecType, typename convertType>
bool check_vector_convert(cl::sycl::vec<vecType, 1> inputVec) {
  // convert()
  return check_vector_convert_modes<vecType, 1, convertType>(inputVec) &&
         check_vector_convert_modes<vecType, 1, convertType>(
             inputVec.template swizzle<cl::sycl::elem::s0>());
}

template <typename vecType, int N, typename asType, int asN>
asType check_as_result(cl::sycl::vec<vecType, N> inputVec,
                       cl::sycl::vec<asType, asN> asVec) {
  vecType tmp_ptr[N];
  for (size_t i = 0; i < N; ++i) {
    tmp_ptr[i] = getElement(inputVec, i);
  }
  asType *exp_ptr = reinterpret_cast<asType *>(tmp_ptr);
  for (size_t i = 0; i < asN; ++i) {
    if (exp_ptr[i] != getElement(asVec, i)) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Helper function to test as() function of a vec for asType
 * as()
 */
template <typename vecType, typename asType, int asN>
bool check_vector_as(cl::sycl::vec<vecType, 1> inputVec) {
  using asVecType = cl::sycl::vec<asType, asN>;
  asVecType asVec = inputVec.template as<asVecType>();
  asVecType asVecSwizzle =
      inputVec.template swizzle<cl::sycl::elem::s0>().template as<asVecType>();
  return check_as_result(inputVec, asVec) &&
         check_as_result(inputVec, asVecSwizzle);
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 */
template <typename vecType>
bool check_vector_get_count_get_size(cl::sycl::vec<vecType, 2> inputVec) {
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
  return true;
}

/**
 * @brief Helper function to test the convert() function of a vec
 */
template <typename vecType, typename convertType>
bool check_vector_convert(cl::sycl::vec<vecType, 2> inputVec) {
  // convert()
  return check_vector_convert_modes<vecType, 2, convertType>(inputVec) &&
         check_vector_convert_modes<vecType, 2, convertType>(
             inputVec
                 .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>());
}

/**
 * @brief Helper function to test as() function of a vec for asType
 * as()
 */
template <typename vecType, typename asType, int asN>
bool check_vector_as(cl::sycl::vec<vecType, 2> inputVec) {
  // as()
  using asVecType = cl::sycl::vec<asType, asN>;
  asVecType asVec = inputVec.template as<asVecType>();
  asVecType asVecSwizzle =
      inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1>()
          .template as<asVecType>();
  return check_as_result(inputVec, asVec) &&
         check_as_result(inputVec, asVecSwizzle);
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 */
template <typename vecType>
bool check_vector_get_count_get_size(cl::sycl::vec<vecType, 3> inputVec) {
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
  return true;
}

/**
 * @brief Helper function to test the convert() function of a vec
 */
template <typename vecType, typename convertType>
bool check_vector_convert(cl::sycl::vec<vecType, 3> inputVec) {
  // convert()
  return check_vector_convert_modes<vecType, 3, convertType>(inputVec) &&
         check_vector_convert_modes<vecType, 3, convertType>(
             inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                       cl::sycl::elem::s2>());
}

/**
 * @brief Helper function to test as() function of a vec for asType
 * as()
 */
template <typename vecType, typename asType, int asN>
bool check_vector_as(cl::sycl::vec<vecType, 3> inputVec) {
  // as()
  using asVecType = cl::sycl::vec<asType, asN>;
  asVecType asVec = inputVec.template as<asVecType>();
  asVecType asVecSwizzle =
      inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                cl::sycl::elem::s2>()
          .template as<asVecType>();
  return check_as_result(inputVec, asVec) &&
         check_as_result(inputVec, asVecSwizzle);
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 */
template <typename vecType>
bool check_vector_get_count_get_size(cl::sycl::vec<vecType, 4> inputVec) {
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
  return true;
}

/**
 * @brief Helper function to test the convert() function of a vec
 */
template <typename vecType, typename convertType>
bool check_vector_convert(cl::sycl::vec<vecType, 4> inputVec) {
  // convert()
  return check_vector_convert_modes<vecType, 4, convertType>(inputVec) &&
         check_vector_convert_modes<vecType, 4, convertType>(
             inputVec
                 .template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                   cl::sycl::elem::s2, cl::sycl::elem::s3>());
}

/**
 * @brief Helper function to test as() function of a vec for asType
 * as()
 */
template <typename vecType, typename asType, int asN>
bool check_vector_as(cl::sycl::vec<vecType, 4> inputVec) {
  // as()
  using asVecType = cl::sycl::vec<asType, asN>;
  asVecType asVec = inputVec.template as<asVecType>();
  asVecType asVecSwizzle =
      inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                cl::sycl::elem::s2, cl::sycl::elem::s3>()
          .template as<asVecType>();
  return check_as_result(inputVec, asVec) &&
         check_as_result(inputVec, asVecSwizzle);
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 */
template <typename vecType>
bool check_vector_get_count_get_size(cl::sycl::vec<vecType, 8> inputVec) {
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
  return true;
}

/**
 * @brief Helper function to test the convert() function of a vec
 */
template <typename vecType, typename convertType>
bool check_vector_convert(cl::sycl::vec<vecType, 8> inputVec) {
  // convert()
  return check_vector_convert_modes<vecType, 8, convertType>(inputVec) &&
         check_vector_convert_modes<vecType, 8, convertType>(
             inputVec.template swizzle<
                 cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                 cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                 cl::sycl::elem::s6, cl::sycl::elem::s7>());
}

/**
 * @brief Helper function to test as() function of a vec for asType
 * as()
 */
template <typename vecType, typename asType, int asN>
bool check_vector_as(cl::sycl::vec<vecType, 8> inputVec) {
  // as()
  using asVecType = cl::sycl::vec<asType, asN>;
  asVecType asVec = inputVec.template as<asVecType>();
  asVecType asVecSwizzle =
      inputVec.template swizzle<cl::sycl::elem::s0, cl::sycl::elem::s1,
                                cl::sycl::elem::s2, cl::sycl::elem::s3,
                                cl::sycl::elem::s4, cl::sycl::elem::s5,
                                cl::sycl::elem::s6, cl::sycl::elem::s7>()
          .template as<asVecType>();
  return check_as_result(inputVec, asVec) &&
         check_as_result(inputVec, asVecSwizzle);
}

/**
 * @brief Helper function to test the following functions of a vec
 * get_count()
 * get_size()
 */
template <typename vecType>
bool check_vector_get_count_get_size(cl::sycl::vec<vecType, 16> inputVec) {
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
  return true;
}

/**
 * @brief Helper function to test the convert() function of a vec
 */
template <typename vecType, typename convertType>
bool check_vector_convert(cl::sycl::vec<vecType, 16> inputVec) {
  // convert()
  return check_vector_convert_modes<vecType, 16, convertType>(inputVec) &&
         check_vector_convert_modes<vecType, 16, convertType>(
             inputVec.template swizzle<
                 cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                 cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                 cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
                 cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
                 cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
                 cl::sycl::elem::sF>());
}

/**
 * @brief Helper function to test as() function of a vec for asType
 * as()
 */
template <typename vecType, typename asType, int asN>
bool check_vector_as(cl::sycl::vec<vecType, 16> inputVec) {
  // as()
  using asVecType = cl::sycl::vec<asType, asN>;
  asVecType asVec = inputVec.template as<asVecType>();
  asVecType asVecSwizzle =
      inputVec.template swizzle<
                  cl::sycl::elem::s0, cl::sycl::elem::s1, cl::sycl::elem::s2,
                  cl::sycl::elem::s3, cl::sycl::elem::s4, cl::sycl::elem::s5,
                  cl::sycl::elem::s6, cl::sycl::elem::s7, cl::sycl::elem::s8,
                  cl::sycl::elem::s9, cl::sycl::elem::sA, cl::sycl::elem::sB,
                  cl::sycl::elem::sC, cl::sycl::elem::sD, cl::sycl::elem::sE,
                  cl::sycl::elem::sF>()
          .template as<asVecType>();
  return check_as_result(inputVec, asVec) &&
         check_as_result(inputVec, asVecSwizzle);
}

/**
 * @brief Helper function to test as() function of a vec for asType
 * as()
 */
template <typename vecType, int N, typename asType, int asN>
typename std::enable_if<sizeof(cl::sycl::vec<vecType, N>) ==
                            sizeof(cl::sycl::vec<asType, asN>),
                        bool>::type
check_vectorN_as(cl::sycl::vec<vecType, N> inputVec) {
  return check_vector_as<vecType, asType, asN>(inputVec);
}

/**
 * @brief Helper function to exclude types that with different storage size for
 * as() tests
 * as()
 */
template <typename vecType, int N, typename asType, int asN>
typename std::enable_if<sizeof(cl::sycl::vec<vecType, N>) !=
                            sizeof(cl::sycl::vec<asType, asN>),
                        bool>::type
    check_vectorN_as(cl::sycl::vec<vecType, N>) {
  return true;
}

/**
 * @brief Helper function to test as() and convert() functions for all vector
 * sizes
 */
template <typename vecType, int N, typename newVecType>
bool check_convert_as_all_dims(cl::sycl::vec<vecType, N> inputVec) {
  bool result = true;
  result += check_vector_convert<vecType, newVecType>(inputVec);

  result += check_vectorN_as<vecType, N, newVecType, 1>(inputVec);
  result += check_vectorN_as<vecType, N, newVecType, 2>(inputVec);
  result += check_vectorN_as<vecType, N, newVecType, 3>(inputVec);
  result += check_vectorN_as<vecType, N, newVecType, 4>(inputVec);
  result += check_vectorN_as<vecType, N, newVecType, 8>(inputVec);
  result += check_vectorN_as<vecType, N, newVecType, 16>(inputVec);

  return result;
}

/**
 * @brief Helper function to test as() and convert() functions for all types
 */
template <typename vecType, int N>
bool check_convert_as_all_types(cl::sycl::vec<vecType, N> inputVec) {
  bool result = true;
  result += check_convert_as_all_dims<vecType, N, char>(inputVec);
  result += check_convert_as_all_dims<vecType, N, signed char>(inputVec);
  result += check_convert_as_all_dims<vecType, N, unsigned char>(inputVec);
  result += check_convert_as_all_dims<vecType, N, short int>(inputVec);
  result += check_convert_as_all_dims<vecType, N, unsigned short int>(inputVec);
  result += check_convert_as_all_dims<vecType, N, int>(inputVec);
  result += check_convert_as_all_dims<vecType, N, unsigned int>(inputVec);
  result += check_convert_as_all_dims<vecType, N, long int>(inputVec);
  result += check_convert_as_all_dims<vecType, N, unsigned long int>(inputVec);
  result += check_convert_as_all_dims<vecType, N, long long int>(inputVec);
  result +=
      check_convert_as_all_dims<vecType, N, unsigned long long int>(inputVec);
  result += check_convert_as_all_dims<vecType, N, float>(inputVec);
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  result += check_convert_as_all_dims<vecType, N, cl::sycl::byte>(inputVec);

  result += check_convert_as_all_dims<vecType, N, cl::sycl::cl_char>(inputVec);
  result += check_convert_as_all_dims<vecType, N, cl::sycl::cl_uchar>(inputVec);
  result += check_convert_as_all_dims<vecType, N, cl::sycl::cl_short>(inputVec);
  result +=
      check_convert_as_all_dims<vecType, N, cl::sycl::cl_ushort>(inputVec);
  result += check_convert_as_all_dims<vecType, N, cl::sycl::cl_int>(inputVec);
  result += check_convert_as_all_dims<vecType, N, cl::sycl::cl_uint>(inputVec);
  result += check_convert_as_all_dims<vecType, N, cl::sycl::cl_long>(inputVec);
  result += check_convert_as_all_dims<vecType, N, cl::sycl::cl_ulong>(inputVec);
  result += check_convert_as_all_dims<vecType, N, cl::sycl::cl_float>(inputVec);

#ifdef INT8_MAX
  result += check_convert_as_all_dims<vecType, N, std::int8_t>(inputVec);
#endif
#ifdef INT16_MAX
  result += check_convert_as_all_dims<vecType, N, std::int16_t>(inputVec);
#endif
#ifdef INT32_MAX
  result += check_convert_as_all_dims<vecType, N, std::int32_t>(inputVec);
#endif
#ifdef INT64_MAX
  result += check_convert_as_all_dims<vecType, N, std::int64_t>(inputVec);
#endif
#ifdef UINT8_MAX
  result += check_convert_as_all_dims<vecType, N, std::uint8_t>(inputVec);
#endif
#ifdef UINT16_MAX
  result += check_convert_as_all_dims<vecType, N, std::uint16_t>(inputVec);
#endif
#ifdef UINT32_MAX
  result += check_convert_as_all_dims<vecType, N, std::uint32_t>(inputVec);
#endif
#ifdef UINT64_MAX
  result += check_convert_as_all_dims<vecType, N, std::uint64_t>(inputVec);
#endif
#endif // ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return result;
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
