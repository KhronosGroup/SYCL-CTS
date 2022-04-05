/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_COMMON_VEC_H
#define __SYCLCTS_TESTS_COMMON_COMMON_VEC_H

#include <sycl/sycl.hpp>

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
bool check_vector_size(sycl::vec<vecType, numOfElems> vector) {
  int count = (vector.size() == 3) ? 4 : vector.size();
  return ((sizeof(vecType) * count) == vector.byte_size());
}

/**
 * @brief Helper function to check vector values are correct.
 */
template <typename vecType, int numOfElems>
bool check_vector_values(sycl::vec<vecType, numOfElems> vector,
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
typename std::enable_if<is_sycl_floating_point<vecType>::value, bool>::type
check_vector_values_div(sycl::vec<vecType, numOfElems> vector,
                        vecType *vals) {
  for (int i = 0; i < numOfElems; i++) {
    vecType vectorValue = getElement(vector, i);
    if (vals[i] == vectorValue)
      continue;
    const vecType ulpsExpected = 2.5; // Min Accuracy for x / y
    const vecType difference = sycl::fabs(vectorValue - vals[i]);
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
typename std::enable_if<!is_sycl_floating_point<vecType>::value, bool>::type
check_vector_values_div(sycl::vec<vecType, numOfElems> vector,
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

// match float values to expected integer value
template <typename T1, typename T2>
T2 float_map_match(T1 floats[], T2 vals[], int size, T1 src) {
  for (int i = 0; i < size; ++i) {
    if (floats[i] == src) {
      return vals[i];
    }
  }
  return T2{};
}

template <typename sourceType, typename targetType>
static constexpr bool if_FP_to_non_FP_conv_v =
    is_sycl_floating_point<sourceType>::value && !is_sycl_floating_point<targetType>::value;

template <typename vecType, int N, typename convertType>
sycl::vec<convertType, N> convert_vec(sycl::vec<vecType, N> inputVec) {
  sycl::vec<convertType, N> resVec;
  for (size_t i = 0; i < N; ++i) {
    vecType elem = getElement(inputVec, i);
    setElement<convertType, N>(resVec, i, convertType(elem));
  }
  return resVec;
}

template <typename vecType, int N, typename convertType>
sycl::vec<convertType, N> rte(sycl::vec<vecType, N> inputVec) {
  if constexpr (if_FP_to_non_FP_conv_v<vecType, convertType>) {
    const int size = 8;
    vecType floats[size] = {2.3f, 3.8f, 1.5f, 2.5f, -2.3f, -3.8f, -1.5f, -2.5f};
    convertType vals[size] = {2,
                              4,
                              2,
                              2,
                              static_cast<convertType>(-2),
                              static_cast<convertType>(-4),
                              static_cast<convertType>(-2),
                              static_cast<convertType>(-2)};
    sycl::vec<convertType, N> resVec;
    for (size_t i = 0; i < N; ++i) {
      vecType elem = getElement(inputVec, i);
      auto elemConvert = float_map_match(floats, vals, size, elem);
      setElement<convertType, N>(resVec, i, elemConvert);
    }
    return resVec;
  }
  return convert_vec<vecType, N, convertType>(inputVec);
}

// rtz
template <typename vecType, int N, typename convertType>
sycl::vec<convertType, N> rtz(sycl::vec<vecType, N> inputVec) {
  if constexpr (if_FP_to_non_FP_conv_v<vecType, convertType>) {
    const int size = 8;
    vecType floats[size] = {2.3f, 3.8f, 1.5f, 2.5f, -2.3f, -3.8f, -1.5f, -2.5f};
    convertType vals[size] = {2,
                              3,
                              1,
                              2,
                              static_cast<convertType>(-2),
                              static_cast<convertType>(-3),
                              static_cast<convertType>(-1),
                              static_cast<convertType>(-2)};
    sycl::vec<convertType, N> resVec;
    for (size_t i = 0; i < N; ++i) {
      vecType elem = getElement(inputVec, i);
      auto elemConvert = float_map_match(floats, vals, size, elem);
      setElement<convertType, N>(resVec, i, elemConvert);
    }
    return resVec;
  }
  return convert_vec<vecType, N, convertType>(inputVec);
}

// rtp
template <typename vecType, int N, typename convertType>
sycl::vec<convertType, N> rtp(sycl::vec<vecType, N> inputVec) {
  if constexpr (if_FP_to_non_FP_conv_v<vecType, convertType>) {
    const int size = 8;
    vecType floats[size] = {2.3f, 3.8f, 1.5f, 2.5f, -2.3f, -3.8f, -1.5f, -2.5f};
    convertType vals[size] = {3,
                              4,
                              2,
                              3,
                              static_cast<convertType>(-2),
                              static_cast<convertType>(-3),
                              static_cast<convertType>(-1),
                              static_cast<convertType>(-2)};
    sycl::vec<convertType, N> resVec;
    for (size_t i = 0; i < N; ++i) {
      vecType elem = getElement(inputVec, i);
      auto elemConvert = float_map_match(floats, vals, size, elem);
      setElement<convertType, N>(resVec, i, elemConvert);
    }
    return resVec;
  }
  return convert_vec<vecType, N, convertType>(inputVec);
}

// rtn
template <typename vecType, int N, typename convertType>
sycl::vec<convertType, N> rtn(sycl::vec<vecType, N> inputVec) {
  if constexpr (if_FP_to_non_FP_conv_v<vecType, convertType>) {
    const int size = 8;
    vecType floats[size] = {2.3f, 3.8f, 1.5f, 2.5f, -2.3f, -3.8f, -1.5f, -2.5f};
    convertType vals[size] = {2,
                              3,
                              1,
                              2,
                              static_cast<convertType>(-3),
                              static_cast<convertType>(-4),
                              static_cast<convertType>(-2),
                              static_cast<convertType>(-3)};
    sycl::vec<convertType, N> resVec;
    for (size_t i = 0; i < N; ++i) {
      vecType elem = getElement(inputVec, i);
      auto elemConvert = float_map_match(floats, vals, size, elem);
      setElement<convertType, N>(resVec, i, elemConvert);
    }
    return resVec;
  }
  return convert_vec<vecType, N, convertType>(inputVec);
}

// Converting floating point values outside of (-1, max unsigned integer type
// value + 1) to unsigned integer types is undefined behaviour. Since the
// initial vectors contain negative values, check conversion of their absolute
// values instead.
template <typename vecType, int N, typename convertType>
void handleFPToUnsignedConv(sycl::vec<vecType, N>& inputVec) {
  if constexpr (is_sycl_floating_point<vecType>::value &&
                std::is_unsigned_v<convertType>) {
    for (size_t i = 0; i < N; ++i) {
      vecType elem = getElement(inputVec, i);
      if (elem < 0) setElement<vecType, N>(inputVec, i, -elem);
    }
  }
}

template <typename vecType, int N, typename convertType,
          sycl::rounding_mode mode>
bool check_vector_convert_result(sycl::vec<vecType, N> inputVec) {
  handleFPToUnsignedConv<vecType, N, convertType>(inputVec);
  sycl::vec<convertType, N> convertedVec =
      inputVec.template convert<convertType, mode>();

  sycl::vec<convertType, N> expectedVec;
  switch (mode) {
    case sycl::rounding_mode::automatic:
      expectedVec = convert_vec<vecType, N, convertType>(inputVec);
      break;
    case sycl::rounding_mode::rte:
      expectedVec = rte<vecType, N, convertType>(inputVec);
      break;
    case sycl::rounding_mode::rtz:
      expectedVec = rtz<vecType, N, convertType>(inputVec);
      break;
    case sycl::rounding_mode::rtp:
      expectedVec = rtp<vecType, N, convertType>(inputVec);
      break;
    case sycl::rounding_mode::rtn:
      expectedVec = rtn<vecType, N, convertType>(inputVec);
      break;
  }
  if (!check_equal_values(convertedVec, expectedVec)) {
    return false;
  }
  return true;
}

template <typename vecType, int N, typename convertType>
bool check_vector_convert_modes(sycl::vec<vecType, N> inputVec) {
  bool flag = true;
  flag &=
      check_vector_convert_result<vecType, N, convertType,
                                  sycl::rounding_mode::automatic>(inputVec);
#ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
  flag &= check_vector_convert_result<vecType, N, convertType,
                                      sycl::rounding_mode::rte>(inputVec);
  flag &= check_vector_convert_result<vecType, N, convertType,
                                      sycl::rounding_mode::rtz>(inputVec);
  flag &= check_vector_convert_result<vecType, N, convertType,
                                      sycl::rounding_mode::rtp>(inputVec);
  flag &= check_vector_convert_result<vecType, N, convertType,
                                      sycl::rounding_mode::rtn>(inputVec);
#endif  // SYCL_CTS_ENABLE_FULL_CONFORMANCE
  return flag;
}

template <typename T, int N>
struct vector_swizzle_check {
  static auto get_swizzle(sycl::vec<T, N>) {}
};

template <typename T>
struct vector_swizzle_check<T, 1> {
  static auto get_swizzle(sycl::vec<T, 1> v) {
    return v.template swizzle<sycl::elem::s0>();
  }
};

template <typename T>
struct vector_swizzle_check<T, 2> {
  static auto get_swizzle(sycl::vec<T, 2> v) {
    return v.template swizzle<sycl::elem::s0, sycl::elem::s1>();
  }
};

template <typename T>
struct vector_swizzle_check<T, 3> {
  static auto get_swizzle(sycl::vec<T, 3> v) {
    return v.template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2>();
  }
};

template <typename T>
struct vector_swizzle_check<T, 4> {
  static auto get_swizzle(sycl::vec<T, 4> v) {
    return v.template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3>();
  }
};

template <typename T>
struct vector_swizzle_check<T, 8> {
  static auto get_swizzle(sycl::vec<T, 8> v) {
    return v.template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3,
                              sycl::elem::s4, sycl::elem::s5,
                              sycl::elem::s6, sycl::elem::s7>();
  }
};

template <typename T>
struct vector_swizzle_check<T, 16> {
  static auto get_swizzle(sycl::vec<T, 16> v) {
    return v.template swizzle<
        sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,
        sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,
        sycl::elem::s6, sycl::elem::s7, sycl::elem::s8,
        sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,
        sycl::elem::sC, sycl::elem::sD, sycl::elem::sE,
        sycl::elem::sF>();
  }
};

/**
 * @brief Helper function to test the following functions of a vec
 * size()
 * byte_size()
 * get_count()
 * get_size()
 */
template <typename vecType, int N>
bool check_vector_size_byte_size(sycl::vec<vecType, N> inputVec) {
  // size()
  size_t count = inputVec.size();
  if (count != N) {
    return false;
  }
  count = vector_swizzle_check<vecType, N>::get_swizzle(inputVec).size();
  if (count != N) {
    return false;
  }

  // get_count()
  // TODO: mark this check as testing deprecated functionality
  size_t count_depr = inputVec.get_count();
  if (count_depr != N) {
    return false;
  }
  count_depr =
      vector_swizzle_check<vecType, N>::get_swizzle(inputVec).get_count();
  if (count_depr != N) {
    return false;
  }

  // byte_size()
  size_t size = inputVec.byte_size();
  size_t M = (N == 3) ? 4 : N;
  if (size != sizeof(vecType) * M) {
    return false;
  }
  size = vector_swizzle_check<vecType, N>::get_swizzle(inputVec).byte_size();
  if (size != sizeof(vecType) * M) {
    return false;
  }

  // get_size()
  // TODO: mark this check as testing deprecated functionality
  size_t size_depr = inputVec.get_size();
  if (size_depr != sizeof(vecType) * M) {
    return false;
  }
  size_depr =
      vector_swizzle_check<vecType, N>::get_swizzle(inputVec).get_size();
  if (size_depr != sizeof(vecType) * M) {
    return false;
  }
  return true;
}

/**
 * @brief Helper function to test the convert() function of a vec
 */
template <typename vecType, int N, typename convertType>
bool check_vector_convert(sycl::vec<vecType, N> inputVec) {
  // convert()
  return check_vector_convert_modes<vecType, N, convertType>(inputVec) &&
         check_vector_convert_modes<vecType, N, convertType>(
             vector_swizzle_check<vecType, N>::get_swizzle(inputVec));
}

template <typename vecType, int N, typename asType, int asN>
asType check_as_result(sycl::vec<vecType, N> inputVec,
                       sycl::vec<asType, asN> asVec) {
  vecType tmp_ptr[N];
  for (size_t i = 0; i < N; ++i) {
    tmp_ptr[i] = getElement(inputVec, i);
  }
  asType* exp_ptr = reinterpret_cast<asType*>(tmp_ptr);
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
template <typename vecType, int N, typename asType, int asN>
bool check_vector_as(sycl::vec<vecType, N> inputVec) {
  using asVecType = sycl::vec<asType, asN>;
  asVecType asVec = inputVec.template as<asVecType>();
  asVecType asVecSwizzle =
      vector_swizzle_check<vecType, N>::get_swizzle(inputVec)
          .template as<asVecType>();
  return check_as_result(inputVec, asVec) &&
         check_as_result(inputVec, asVecSwizzle);
}

/**
 * @brief Helper function to test as() function of a vec for asType
 * as()
 */
template <typename vecType, int N, typename asType, int asN>
bool check_vectorN_as(sycl::vec<vecType, N> inputVec) {
  if constexpr (sizeof(sycl::vec<vecType, N>) ==
                sizeof(sycl::vec<asType, asN>))
    return check_vector_as<vecType, N, asType, asN>(inputVec);
  else
    return true;
}

/**
 * @brief Helper function to test as() and convert() functions for all vector
 * sizes
 */
template <typename vecType, int N, typename newVecType>
bool check_convert_as_all_dims(sycl::vec<vecType, N> inputVec) {
  bool result = true;
  result += check_vector_convert<vecType, N, newVecType>(inputVec);

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
bool check_convert_as_all_types(sycl::vec<vecType, N> inputVec) {
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
  result += check_convert_as_all_dims<vecType, N, sycl::byte>(inputVec);

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
#endif  // ifdef SYCL_CTS_ENABLE_FULL_CONFORMANCE
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
bool check_lo_hi_odd_even(sycl::vec<vecType, 2> inputVec, vecType* vals) {
  constexpr size_t mid = 1;
  // lo()
  {
    sycl::vec<vecType, mid> loVec{inputVec.lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  {
    sycl::vec<vecType, mid> loVec{
        inputVec.template swizzle<sycl::elem::s0, sycl::elem::s1>()
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
    sycl::vec<vecType, mid> hiVec{inputVec.hi()};
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
    sycl::vec<vecType, mid> hiVec{
        inputVec.template swizzle<sycl::elem::s0, sycl::elem::s1>()
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
    sycl::vec<vecType, mid> oddVec{inputVec.odd()};
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
    sycl::vec<vecType, mid> oddVec{
        inputVec.template swizzle<sycl::elem::s0, sycl::elem::s1>()
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
    sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    sycl::vec<vecType, mid> evenVec{
        inputVec.template swizzle<sycl::elem::s0, sycl::elem::s1>()
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
bool check_lo_hi_odd_even(sycl::vec<vecType, 3> inputVec, vecType* vals) {
  constexpr size_t mid = 2;
  // lo()
  {
    sycl::vec<vecType, mid> loVec{inputVec.lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  {
    sycl::vec<vecType, mid> loVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2>()
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
    sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    sycl::vec<vecType, mid> evenVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2>()
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
bool check_lo_hi_odd_even(sycl::vec<vecType, 4> inputVec, vecType* vals) {
  constexpr size_t mid = 2;
  // lo()
  {
    sycl::vec<vecType, mid> loVec{inputVec.lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  {
    sycl::vec<vecType, mid> loVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3>()
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
    sycl::vec<vecType, mid> hiVec{inputVec.hi()};
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
    sycl::vec<vecType, mid> hiVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3>()
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
    sycl::vec<vecType, mid> oddVec{inputVec.odd()};
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
    sycl::vec<vecType, mid> oddVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3>()
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
    sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    sycl::vec<vecType, mid> evenVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3>()
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
bool check_lo_hi_odd_even(sycl::vec<vecType, 8> inputVec, vecType* vals) {
  constexpr size_t mid = 4;
  // lo()
  {
    sycl::vec<vecType, mid> loVec{inputVec.lo()};
    vecType loVals[mid] = {0};
    for (size_t i = 0; i < mid; i++) {
      loVals[i] = vals[i];
    }
    if (!check_vector_values<vecType, mid>(loVec, loVals)) {
      return false;
    }
  }
  {
    sycl::vec<vecType, mid> loVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3,
                              sycl::elem::s4, sycl::elem::s5,
                              sycl::elem::s6, sycl::elem::s7>()
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
    sycl::vec<vecType, mid> hiVec{inputVec.hi()};
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
    sycl::vec<vecType, mid> hiVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3,
                              sycl::elem::s4, sycl::elem::s5,
                              sycl::elem::s6, sycl::elem::s7>()
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
    sycl::vec<vecType, mid> oddVec{inputVec.odd()};
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
    sycl::vec<vecType, mid> oddVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3,
                              sycl::elem::s4, sycl::elem::s5,
                              sycl::elem::s6, sycl::elem::s7>()
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
    sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    sycl::vec<vecType, mid> evenVec{
        inputVec
            .template swizzle<sycl::elem::s0, sycl::elem::s1,
                              sycl::elem::s2, sycl::elem::s3,
                              sycl::elem::s4, sycl::elem::s5,
                              sycl::elem::s6, sycl::elem::s7>()
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
bool check_lo_hi_odd_even(sycl::vec<vecType, 16> inputVec, vecType* vals) {
  constexpr size_t mid = 8;
  // lo()
  {
    sycl::vec<vecType, mid> loVec{inputVec

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
    sycl::vec<vecType, mid> loVec{
        inputVec
            .template swizzle<
                sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,
                sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,
                sycl::elem::s6, sycl::elem::s7, sycl::elem::s8,
                sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,
                sycl::elem::sC, sycl::elem::sD, sycl::elem::sE,
                sycl::elem::sF>()
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
    sycl::vec<vecType, mid> hiVec{inputVec.hi()};
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
    sycl::vec<vecType, mid> hiVec{
        inputVec
            .template swizzle<
                sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,
                sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,
                sycl::elem::s6, sycl::elem::s7, sycl::elem::s8,
                sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,
                sycl::elem::sC, sycl::elem::sD, sycl::elem::sE,
                sycl::elem::sF>()
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
    sycl::vec<vecType, mid> oddVec{inputVec.odd()};
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
    sycl::vec<vecType, mid> oddVec{
        inputVec
            .template swizzle<
                sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,
                sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,
                sycl::elem::s6, sycl::elem::s7, sycl::elem::s8,
                sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,
                sycl::elem::sC, sycl::elem::sD, sycl::elem::sE,
                sycl::elem::sF>()
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
    sycl::vec<vecType, mid> evenVec{inputVec.even()};
    vecType evenVals[mid] = {0};
    for (size_t i = 0; i < mid; ++i) {
      evenVals[i] = vals[i * 2];
    }
    if (!check_vector_values<vecType, mid>(evenVec, evenVals)) {
      return false;
    }
  }
  {
    sycl::vec<vecType, mid> evenVec{
        inputVec
            .template swizzle<
                sycl::elem::s0, sycl::elem::s1, sycl::elem::s2,
                sycl::elem::s3, sycl::elem::s4, sycl::elem::s5,
                sycl::elem::s6, sycl::elem::s7, sycl::elem::s8,
                sycl::elem::s9, sycl::elem::sA, sycl::elem::sB,
                sycl::elem::sC, sycl::elem::sD, sycl::elem::sE,
                sycl::elem::sF>()
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
