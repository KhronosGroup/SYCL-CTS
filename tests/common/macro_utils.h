/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2021 The Khronos Group Inc.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_COMMON_MACRO_UTILS_H
#define __SYCLCTS_TESTS_COMMON_MACRO_UTILS_H

#define INTERNAL_CTS_CAT(a, b) _INTERNAL_CTS_CAT(a, b)
#define _INTERNAL_CTS_CAT(a, b) a##b

#define INTERNAL_CTS_IF(c) INTERNAL_CTS_CAT(_INTERNAL_CTS_IF_, c)
#define _INTERNAL_CTS_IF_0(t, f) f
#define _INTERNAL_CTS_IF_1(t, f) t

#define INTERNAL_CTS_NOT(x) \
  INTERNAL_CTS_CHECK(INTERNAL_CTS_CAT(_INTERNAL_CTS_NOT_, x))
#define _INTERNAL_CTS_NOT_0 INTERNAL_CTS_PROBE()
#define INTERNAL_CTS_BOOL(x) INTERNAL_CTS_NOT(INTERNAL_CTS_NOT(x))

#define INTERNAL_CTS_CHECK(...) _INTERNAL_CTS_CHECK(__VA_ARGS__, 0)
#define _INTERNAL_CTS_CHECK(_, v, ...) v
#define INTERNAL_CTS_PROBE() ~, 1

#define INTERNAL_CTS_FIRST(a, ...) a
#define INTERNAL_CTS_HAS_ARGS(...) \
  INTERNAL_CTS_BOOL(               \
      INTERNAL_CTS_FIRST(_INTERNAL_CTS_END_OF_ARGS_ __VA_ARGS__)())
#define _INTERNAL_CTS_END_OF_ARGS_() 0

#endif  // __SYCLCTS_TESTS_COMMON_MACRO_UTILS_H