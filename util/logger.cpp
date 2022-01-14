/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2020-2021 The Khronos Group Inc.
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "logger.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>

#include <catch2/catch_test_macros.hpp>
namespace sycl_cts {
namespace util {

void logger::fail(const std::string &str, const int line) {
  FAIL("line " << line << ": " << str);
  (void)line;
}

void logger::note(const std::string &str) { WARN(str); }

void logger::note(const char *fmt, ...) {
  assert(fmt != nullptr);

  char buffer[1024];

  va_list args;
  va_start(args, fmt);
  if (vsnprintf(buffer, sizeof(buffer), fmt, args) <= 0)
    assert(!"vsnprintf failed");
  va_end(args);

  // enforce terminal character
  buffer[sizeof(buffer) - 1] = '\0';

  note(std::string(buffer));
}

void logger::log_debug(const std::string &str) { INFO(str); }

}  // namespace util
}  // namespace sycl_cts
