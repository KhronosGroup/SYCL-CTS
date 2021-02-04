/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_MATH_HELPER_H
#define __SYCLCTS_UTIL_MATH_HELPER_H

#include "../util/stl.h"
#include "../tests/common/sycl.h"
#include "./../oclmath/mt19937.h"
#include "./math_vector.h"

namespace sycl_cts {
/** math utility functions
 */
namespace math {

template <typename R, typename T, int N, typename funT, typename... Args>
cl::sycl::vec<R, N> run_func_on_vector(funT fun, Args... args) {
  cl::sycl::vec<R, N> res;
  for (int i = 0; i < N; i++) {
    setElement<R, N>(res, i, fun(getElement<T, N>(args, i)...));
  }
  return res;
}

/* helper for relational functions where true result gives 1 for scalar
    and -1 for vector argument types */
template <typename R, typename T, int N, typename funT, typename... Args>
cl::sycl::vec<R, N> run_rel_func_on_vector(funT fun, Args... args) {
  cl::sycl::vec<R, N> res;
  for (int i = 0; i < N; i++) {
    if (fun(getElement<T, N>(args, i)...) == 1)
      setElement<R, N>(res, i, -1);
    else
      setElement<R, N>(res, i, 0);
  }
  return res;
}

/* cast an integer to a float */
float int_to_float(uint32_t x);

void fill(float &e, float v);
void fill(cl::sycl::float2 &e, float v);
void fill(cl::sycl::float3 &e, float v);
void fill(cl::sycl::float4 &e, float v);
void fill(cl::sycl::float8 &e, float v);
void fill(cl::sycl::float16 &e, float v);

/* return number of elements in a type */
int numElements(const float &);

/* return number of elements in a type */
int numElements(const int &);

template <typename T, int numElems>
int numElements(const cl::sycl::vec<T, numElems> &) {
  return numElems;
}

/* extract an individual elements */
float getElement(const float &f, int ix);

/* extract individual elements of an integer type */
int getElement(const int &f, int ix);

template <typename T, int dim>
T getElement(cl::sycl::vec<T, dim> &f, int ix) {
  return getComponent<T, dim>()(f, ix);
}

template <typename T, int dim>
void setElement(cl::sycl::vec<T, dim> &f, int ix, T value) {
  setComponent<T, dim>()(f, ix, value);
}

/* create random floats with an integer range [-0x7fffffff to 0x7fffffff]
 */
void rand(MTdata &rng, float *buf, int num);
void rand(MTdata &rng, cl::sycl::float2 *buf, int num);
void rand(MTdata &rng, cl::sycl::float3 *buf, int num);
void rand(MTdata &rng, cl::sycl::float4 *buf, int num);
void rand(MTdata &rng, cl::sycl::float8 *buf, int num);
void rand(MTdata &rng, cl::sycl::float16 *buf, int num);

/* generate a stream of random integer data
 */
void rand(MTdata &rng, uint8_t *buf, int size);

} /* namespace math     */
} /* namespace sycl_cts */

#endif // __SYCLCTS_UTIL_MATH_HELPER_H
