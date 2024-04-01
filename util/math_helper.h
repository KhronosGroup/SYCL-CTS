/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
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

#ifndef __SYCLCTS_UTIL_MATH_HELPER_H
#define __SYCLCTS_UTIL_MATH_HELPER_H

#include <map>
#include <climits>

#include <sycl/sycl.hpp>

#include "../util/stl.h"
#include "./../oclmath/mt19937.h"

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

/* Generic function for both scalar and vector types to
 * return the number of elements in a type. */
int numElements(const float &);

/* Generic function for both scalar and vector types to
 * return the number of elements in a type. */
int numElements(const int &);

/* Generic function for both scalar and vector types to
 * return the number of elements in a type. */
template <typename T, int numElems>
int numElements(const sycl::vec<T, numElems> &) {
  return numElems;
}

/* Generic function for both scalar and marray types to
 * return the number of elements in a type. */
// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t numElems>
int numElements(const sycl::marray<T, numElems> &) {
  return numElems;
}
#endif

/* Generic function for both scalar and vector types to
 * extract an individual element. */
template <typename T>
T getElement(const T &f, int) {
  return f;
}

/* Generic function for both scalar and vector types to
 * extract an individual element. */
template <typename T, int dim>
T getElement(sycl::vec<T, dim> &f, int ix) {
  return f[ix];
}

// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
/* Generic function for both scalar and vector types to
 * extract an individual element. */
template <typename T, size_t dim>
T getElement(sycl::marray<T, dim> &f, size_t ix) {
  return f[ix];
}
#endif

template <typename T, int dim>
void setElement(sycl::vec<T, dim> &f, int ix, T value) {
  f[ix] = value;
}

template <typename R, typename T, int N, typename funT, typename... Args>
sycl::vec<R, N> run_func_on_vector(funT fun, Args... args) {
  sycl::vec<R, N> res;
  for (int i = 0; i < N; i++) {
    setElement<R, N>(res, i, fun(getElement(args, i)...));
  }
  return res;
}

// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename R, typename T, size_t N, typename funT, typename... Args>
sycl::marray<R, N> run_func_on_marray(funT fun, Args... args) {
  sycl::marray<R, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = fun(getElement(args, i)...);
  }
  return res;
}
#endif

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

// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N, typename funT, typename... Args>
sycl::marray<bool, N> run_rel_func_on_marray(funT fun, Args... args) {
  sycl::marray<bool, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = (fun(getElement<T, N>(args, i)...));
  }
  return res;
}
#endif

template <typename T, int N, typename funT, typename... Args>
sycl_cts::resultRef<sycl::vec<T, N>>
run_func_on_vector_result_ref(funT fun, Args... args) {
  sycl::vec<T, N> res;
  std::map<int, bool> undefined;
  for (int i = 0; i < N; i++) {
    // bool does not have a default value so we initialize it ourselves to false
    undefined[i] = false;
    resultRef<T> element = fun(getElement(args, i)...);
    if (element.undefined.empty())
      setElement<T, N>(res, i, element.res);
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::vec<T, N>>(res, undefined);
}

// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename T, size_t N, typename funT, typename... Args>
sycl_cts::resultRef<sycl::marray<T, N>> run_func_on_marray_result_ref(
    funT fun, Args... args) {
  sycl::marray<T, N> res;
  std::map<int, bool> undefined;
  for (size_t i = 0; i < N; i++) {
    resultRef<T> element = fun(getElement(args, i)...);
    if (element.undefined.empty())
      res[i] = element.res;
    else
      undefined[i] = true;
  }
  return sycl_cts::resultRef<sycl::marray<T, N>>(res, undefined);
}
#endif

template <typename T>
struct rel_funcs_return;
template <>
struct rel_funcs_return<sycl::half> {
  using type = int16_t;
};
template <>
struct rel_funcs_return<float> {
  using type = int32_t;
};
template <>
struct rel_funcs_return<double> {
  using type = int64_t;
};

template <template <class> class funT, typename T, typename... Args>
bool rel_func_dispatcher(T a, Args... args) {
  return funT<T>()(a, args...);
}

template <template <class> class funT, typename T, int N, typename... Args>
typename sycl::vec<typename rel_funcs_return<T>::type, N>
rel_func_dispatcher(sycl::vec<T, N> a, Args... args) {
  return run_rel_func_on_vector<typename rel_funcs_return<T>::type, T, N>(
      funT<T>{}, a, args...);
}

// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <template <class> class funT, typename T, size_t N, typename... Args>
sycl::marray<bool, N> rel_func_dispatcher(sycl::marray<T, N> a, Args... args) {
  return run_rel_func_on_marray<T, N>(funT<T>{}, a, args...);
}
#endif

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

// FIXME: hipSYCL does not support marray
#ifndef SYCL_CTS_COMPILING_WITH_HIPSYCL
template <typename funT, typename T, size_t N, typename... Args>
sycl::marray<bool, N> rel_func_dispatcher(funT fun, sycl::marray<T, N> a,
                                          Args... args) {
  return run_rel_func_on_marray<T, N>(fun, a, args...);
}
#endif

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
