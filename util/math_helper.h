/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#pragma once

#include "../util/stl.h"
#include "../tests/common/sycl.h"
#include "./../oclmath/mt19937.h"
#include "./math_vector.h"

namespace sycl_cts {
/** math utility functions
 */
namespace math {
using namespace cl::sycl;

/* cast an integer to a float */
float int_to_float(uint32_t x);

void fill(float &e, float v);
void fill(float2 &e, float v);
void fill(float3 &e, float v);
void fill(float4 &e, float v);
void fill(float8 &e, float v);
void fill(float16 &e, float v);

/* return number of elements in a type */
int numElements(const float &);
int numElements(const float2 &);
int numElements(const float3 &);
int numElements(const float4 &);
int numElements(const float8 &);
int numElements(const float16 &);

/* return number of elements in a type */
int numElements(const int &);
int numElements(const int2 &);
int numElements(const int3 &);
int numElements(const int4 &);
int numElements(const int8 &);
int numElements(const int16 &);

/* extract an individual elements */
float getElement(const float &f, int ix);

/* extract individual elements of an integer type */
int getElement(const int &f, int ix);

template <typename T, int dim> T getElement(cl::sycl::vec<T, dim> &f, int ix) {
  return getComponent<T, dim>()(f, ix);
}

template <typename T, int dim>
void setElement(cl::sycl::vec<T, dim> &f, int ix, T value) {
  setComponent<T, dim>()(f, ix, value);
}

/* create random floats with an integer range [-0x7fffffff to 0x7fffffff]
 */
void rand(MTdata &rng, float *buf, int num);
void rand(MTdata &rng, float2 *buf, int num);
void rand(MTdata &rng, float3 *buf, int num);
void rand(MTdata &rng, float4 *buf, int num);
void rand(MTdata &rng, float8 *buf, int num);
void rand(MTdata &rng, float16 *buf, int num);

/* generate a stream of random integer data
 */
void rand(MTdata &rng, uint8_t *buf, int size);

} /* namespace math     */
} /* namespace sycl_cts */
