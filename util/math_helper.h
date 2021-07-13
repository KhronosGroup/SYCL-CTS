/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_MATH_HELPER_H
#define __SYCLCTS_UTIL_MATH_HELPER_H

#include "../tests/common/sycl.h"
#include "../util/stl.h"
#include "./../oclmath/mt19937.h"
#include "./math_vector.h"
#include <map>

#include <climits>

namespace sycl_cts {
/** math utility functions
 */

template <typename returnT> struct resultRef {
  returnT res;
  std::map<int, bool> undefined;

  template <typename U>
  resultRef(U res_t, std::map<int, bool> und_t)
      : res(res_t), undefined(und_t) {}

  template <typename U>
  resultRef(U res_t, bool und_t)
      : res(res_t), undefined({{0, und_t}}) {}

  template <typename U> resultRef(U res_t) : res(res_t) {}

  template <class U>
  resultRef(const resultRef<U> &other)
      : res(other.res), undefined(other.undefined) {}
};

namespace math {

template <typename R, typename T, int N, typename funT, typename... Args>
sycl::vec<R, N> run_func_on_vector(funT fun, Args... args) {
  sycl::vec<R, N> res;
  for (int i = 0; i < N; i++) {
    setElement<R, N>(res, i, fun(getElement(args, i)...));
  }
  return res;
}

/* helper for relational functions where true result gives 1 for scalar
    and -1 for vector argument types */
template <typename R, typename T, int N, typename funT, typename... Args>
sycl::vec<R, N> run_rel_func_on_vector(funT fun, Args... args) {
  sycl::vec<R, N> res;
  for (int i = 0; i < N; i++) {
    if (fun(getElement<T, N>(args, i)...))
      setElement<R, N>(res, i, -1);
    else
      setElement<R, N>(res, i, 0);
  }
  return res;
}

template <typename T, int N, typename funT, typename... Args>
sycl_cts::resultRef<sycl::vec<T, N>>
run_func_on_vector_result_ref(funT fun, Args... args) {
  sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    resultRef<T> element = fun(getElement(args, i)...);
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::vec<T, N>>(res, undefined);
}

template <typename T>
struct rel_funcs_return;
template <>
struct rel_funcs_return<float> {
  using type = int32_t;
};
template <>
struct rel_funcs_return<double> {
  using type = int64_t;
};


template<template<class> class funT, typename T, typename... Args>
typename rel_funcs_return<T>::type rel_func_dispatcher(T a, Args... args) {
  return funT<T>()(a, args...);
}

template <template <class> class funT, typename T, int N, typename... Args>
typename sycl::vec<typename rel_funcs_return<T>::type, N>
rel_func_dispatcher(sycl::vec<T, N> a, Args... args) {
  return run_rel_func_on_vector<typename rel_funcs_return<T>::type, T, N>(
      funT<T>{}, a, args...);
}

template <typename funT, typename T, typename... Args>
typename rel_funcs_return<T>::type rel_func_dispatcher(funT fun, T a,
                                                       Args... args) {
  return fun(a, args...);
}

template <typename funT, typename T, int N, typename... Args>
typename sycl::vec<typename rel_funcs_return<T>::type, N>
rel_func_dispatcher(funT fun, sycl::vec<T, N> a, Args... args) {
  return run_rel_func_on_vector<typename rel_funcs_return<T>::type, T, N>(
      fun, a, args...);
}

template <typename T>
int32_t num_bits(T) {
  return int32_t(sizeof(T) * CHAR_BIT);
}

template <typename T> bool if_bit_set(T num, int bit) {
  return (num >> bit) & 1;
}

template <typename T> bool if_msb_set(T x) {
  return if_bit_set(x, num_bits(x) - 1);
}

/* cast an integer to a float */
float int_to_float(uint32_t x);

void fill(float &e, float v);
void fill(sycl::float2 &e, float v);
void fill(sycl::float3 &e, float v);
void fill(sycl::float4 &e, float v);
void fill(sycl::float8 &e, float v);
void fill(sycl::float16 &e, float v);

/* return number of elements in a type */
int numElements(const float &);

/* return number of elements in a type */
int numElements(const int &);

template <typename T, int numElems>
int numElements(const sycl::vec<T, numElems> &) {
  return numElems;
}

/* extract an individual elements */
float getElement(const float &f, int ix);

/* extract individual elements of an integer type */
int getElement(const int &f, int ix);

template <typename T, int dim>
T getElement(sycl::vec<T, dim> &f, int ix) {
  return getComponent<T, dim>()(f, ix);
}

template <typename T, int dim>
void setElement(sycl::vec<T, dim> &f, int ix, T value) {
  setComponent<T, dim>()(f, ix, value);
}

/* create random floats with an integer range [-0x7fffffff to 0x7fffffff]
 */
void rand(MTdata &rng, float *buf, int num);
void rand(MTdata &rng, sycl::float2 *buf, int num);
void rand(MTdata &rng, sycl::float3 *buf, int num);
void rand(MTdata &rng, sycl::float4 *buf, int num);
void rand(MTdata &rng, sycl::float8 *buf, int num);
void rand(MTdata &rng, sycl::float16 *buf, int num);

/* generate a stream of random integer data
 */
void rand(MTdata &rng, uint8_t *buf, int size);

} /* namespace math     */
} /* namespace sycl_cts */

#endif // __SYCLCTS_UTIL_MATH_HELPER_H
