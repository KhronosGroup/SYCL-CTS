/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_UTIL_TYPE_NAMES_H
#define __SYCLCTS_UTIL_TYPE_NAMES_H

#include "../tests/common/sycl.h"
#include "stl.h"

template <typename T>
std::string type_name() {
  using std::string;

#define MAKENAME(X)                                                \
  {                                                                \
    if (typeid(T) == typeid(X)) return std::string(#X); \
  }

#define MAKESYCLNAME(X)                                                      \
  {                                                                          \
    if (typeid(T) == typeid(sycl::X)) return std::string(#X); \
  }

#define MAKESTDNAME(X)                                               \
  {                                                                  \
    if (typeid(T) == typeid(::X)) return std::string(#X); \
  }

  /* float types */
  MAKENAME(float);
  MAKENAME(double);

  /* scalar types */
  MAKESTDNAME(int8_t);
  MAKESTDNAME(uint8_t);
  MAKESTDNAME(int16_t);
  MAKESTDNAME(uint16_t);
  MAKESTDNAME(int32_t);
  MAKESTDNAME(uint32_t);
  MAKESTDNAME(int64_t);
  MAKESTDNAME(uint64_t);

  /* std types */
  MAKESTDNAME(size_t);

  /* vector types */
  MAKESYCLNAME(char2);
  MAKESYCLNAME(uchar2);
  MAKESYCLNAME(char3);
  MAKESYCLNAME(uchar3);
  MAKESYCLNAME(char4);
  MAKESYCLNAME(uchar4);
  MAKESYCLNAME(char8);
  MAKESYCLNAME(uchar8);
  MAKESYCLNAME(char16);
  MAKESYCLNAME(uchar16);
  MAKESYCLNAME(short2);
  MAKESYCLNAME(ushort2);
  MAKESYCLNAME(short3);
  MAKESYCLNAME(ushort3);
  MAKESYCLNAME(short4);
  MAKESYCLNAME(ushort4);
  MAKESYCLNAME(short8);
  MAKESYCLNAME(ushort8);
  MAKESYCLNAME(short16);
  MAKESYCLNAME(ushort16);
  MAKESYCLNAME(int2);
  MAKESYCLNAME(uint2);
  MAKESYCLNAME(int3);
  MAKESYCLNAME(uint3);
  MAKESYCLNAME(int4);
  MAKESYCLNAME(uint4);
  MAKESYCLNAME(int8);
  MAKESYCLNAME(uint8);
  MAKESYCLNAME(int16);
  MAKESYCLNAME(uint16);
  MAKESYCLNAME(long2);
  MAKESYCLNAME(ulong2);
  MAKESYCLNAME(long3);
  MAKESYCLNAME(ulong3);
  MAKESYCLNAME(long4);
  MAKESYCLNAME(ulong4);
  MAKESYCLNAME(long8);
  MAKESYCLNAME(ulong8);
  MAKESYCLNAME(long16);
  MAKESYCLNAME(ulong16);

  /* float vector types */
  MAKESYCLNAME(float2);
  MAKESYCLNAME(float3);
  MAKESYCLNAME(float4);
  MAKESYCLNAME(float8);
  MAKESYCLNAME(float16);

  /* double vector types */
  MAKESYCLNAME(double2);
  MAKESYCLNAME(double3);
  MAKESYCLNAME(double4);
  MAKESYCLNAME(double8);
  MAKESYCLNAME(double16);

  /* fall back to the implementation defined name */
  const char *fallback_name = typeid(T).name();
  return std::string(fallback_name);

#undef MAKENAME
#undef MAKESTDNAME
#undef MAKESYCLNAME
}

#endif  // __SYCLCTS_UTIL_TYPE_NAMES_H
