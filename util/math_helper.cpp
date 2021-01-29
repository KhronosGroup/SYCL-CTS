/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "math_helper.h"

namespace sycl_cts {
/** math utility functions
 */
namespace math {

/* cast an integer to a float */
float int_to_float(uint32_t x) {
  static_assert(sizeof(x) == sizeof(float), "incompatible type sizes");
  return *reinterpret_cast<float *>(&x);
}

void fill(float &e, float v) { e = v; }

void fill(cl::sycl::float2 &e, float v) {
  e.x() = v;
  e.y() = v;
}

void fill(cl::sycl::float3 &e, float v) { e.x() = e.y() = e.z() = v; }

void fill(cl::sycl::float4 &e, float v) { e.x() = e.y() = e.z() = e.w() = v; }

void fill(cl::sycl::float8 &e, float v) {
  e.s0() = e.s1() = e.s2() = e.s3() = v;
  e.s4() = e.s5() = e.s6() = e.s7() = v;
}

void fill(cl::sycl::float16 &e, float v) {
  e.s0() = e.s1() = e.s2() = e.s3() = v;
  e.s4() = e.s5() = e.s6() = e.s7() = v;
  e.s8() = e.s9() = e.sA() = e.sB() = v;
  e.sC() = e.sD() = e.sE() = e.sF() = v;
}

/* return number of elements in a type */
int numElements(const float &) { return 1; }

/* return number of elements in a type */
int numElements(const int &) { return 1; }

/* extract an individual elements of a float type */
float getElement(const float &f, int ix) { return f; }

/* extract an individual elements of an int type */
int getElement(const int &f, int ix) { return f; }

/* create random floats with full integer range */
void rand(MTdata &rng, float *buf, int num) {
  for (int i = 0; i < num; i++) buf[i] = (float)int32_t(genrand_int32(rng));
}

void rand(MTdata &rng, cl::sycl::float2 *buf, int num) {
  const int nDim = int(sizeof(cl::sycl::float2) / sizeof(float));
  rand(rng, (float *)buf, num * nDim);
}

void rand(MTdata &rng, cl::sycl::float3 *buf, int num) {
  const int nDim = int(sizeof(cl::sycl::float3) / sizeof(float));
  rand(rng, (float *)buf, num * nDim);
}

void rand(MTdata &rng, cl::sycl::float4 *buf, int num) {
  const int nDim = int(sizeof(cl::sycl::float4) / sizeof(float));
  rand(rng, (float *)buf, num * nDim);
}

void rand(MTdata &rng, cl::sycl::float8 *buf, int num) {
  const int nDim = int(sizeof(cl::sycl::float8) / sizeof(float));
  rand(rng, (float *)buf, num * nDim);
}

void rand(MTdata &rng, cl::sycl::float16 *buf, int num) {
  const int nDim = int(sizeof(cl::sycl::float16) / sizeof(float));
  rand(rng, (float *)buf, num * nDim);
}

/* generate a stream of random integer data */
void rand(MTdata &rng, uint8_t *buf, int size) {
  uint32_t r = 0;
  for (int i = 0; i < size; i++) {
    if ((i % 4) == 0) r = genrand_int32(rng);
    buf[i] = r & 0xff;
    r >>= 8;
  }
}

} /* namespace math     */
} /* namespace sycl_cts */
