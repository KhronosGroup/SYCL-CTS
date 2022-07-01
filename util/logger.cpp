/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2020-2022 The Khronos Group Inc.
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
