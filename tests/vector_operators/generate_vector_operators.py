#!/usr/bin/env python3
# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#
#   Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
#   Copyright (c) 2022 The Khronos Group Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ************************************************************************

import sys
import argparse
from string import Template
sys.path.append('../common/')
from common_python_vec import (Data, ReverseData, wrap_with_kernel,
                               wrap_with_test_func, make_func_call,
                               write_source_file, get_types, cast_to_bool,
                               make_fp_or_byte_explicit)

TEST_NAME = 'OPERATORS'

all_type_test_template = Template("""
  /** Performs a test of each vector operator available to all types,
   *  on a given type, size, and two given values of that type
   */
  auto testVec1 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec1Copy = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  sycl::vec<${type}, ${size}> resVec;
  ${type} resArr[${size}];
  ${type} resArr2[${size}];

  // Arithmetic operators
#if !${type_is_std_byte}
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) + static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 + testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 + testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 + static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} + testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} + testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} + static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) + testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) + testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) - static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 - testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 - testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 - static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} - testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} - testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} - static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) - testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) - testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) * static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 * testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 * testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 * static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} * testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} * testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} * static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) * testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) * testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) / static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 / testVec2;
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 / testVec2.${swizzle};
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 / static_cast<${type}>(${test_value_2});
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} / testVec2;
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} / testVec2.${swizzle};
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} / static_cast<${type}>(${test_value_2});
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) / testVec2;
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) / testVec2.${swizzle};
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1});
  }
  resVec = +testVec1;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = +testVec1.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = -(static_cast<${type}>(${test_value_1}));
  }
  resVec = -testVec1;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = -testVec1.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }

  // Post and pre increment and decrement
  // The standard does not require ++ and -- (either prefix or postfix) for the
  // bool type, so we skip tests for these operators if DataT is bool. This cannot
  // be done via `if constexpr` because the type is not a dependent type.
#if !${type_is_bool}
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1});
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr2[i] = static_cast<${type}>(${test_value_1}) + static_cast<${type}>(1);
  }
  testVec1Copy = testVec1;
  resVec = testVec1Copy++;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  if (!check_vector_values(testVec1Copy, resArr2)) {
    resAcc[0] = false;
  }
  testVec1Copy = testVec1;
  resVec = testVec1Copy.${swizzle}++;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  if (!check_vector_values(testVec1Copy, resArr2)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) + static_cast<${type}>(1);
  }
  testVec1Copy = testVec1;
  resVec = ++testVec1Copy;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  if (!check_vector_values(testVec1Copy, resArr)) {
    resAcc[0] = false;
  }
  testVec1Copy = testVec1;
  resVec = ++testVec1Copy.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  if (!check_vector_values(testVec1Copy, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1});
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr2[i] = static_cast<${type}>(${test_value_1}) - static_cast<${type}>(1);
  }
  testVec1Copy = testVec1;
  resVec = testVec1Copy--;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  if (!check_vector_values(testVec1Copy, resArr2)) {
    resAcc[0] = false;
  }
  testVec1Copy = testVec1;
  resVec = testVec1Copy.${swizzle}--;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  if (!check_vector_values(testVec1Copy, resArr2)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) - static_cast<${type}>(1);
  }
  testVec1Copy = testVec1;
  resVec = --testVec1Copy;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  if (!check_vector_values(testVec1Copy, resArr)) {
    resAcc[0] = false;
  }
  testVec1Copy = testVec1;
  resVec = --testVec1Copy.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  if (!check_vector_values(testVec1Copy, resArr)) {
    resAcc[0] = false;
  }
#endif
#endif

  // Assignment operators
#if !${type_is_std_byte}
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) + static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec += testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec += testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec += static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} += testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} += testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} += static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) - static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec -= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec -= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec -= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} -= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} -= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} -= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) * static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec *= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec *= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec *= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} *= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} *= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} *= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) / static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec /= testVec2;
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec /= testVec2.${swizzle};
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec /= static_cast<${type}>(${test_value_2});
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} /= testVec2;
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} /= testVec2.${swizzle};
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} /= static_cast<${type}>(${test_value_2});
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
#endif
""")

specific_return_type_test_template = Template("""
  /** Tests each logical and relational operator available to vector types
   */
  auto testVec1 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  sycl::vec<${ret_type}, ${size}> resVec;
  ${ret_type} resArr[${size}];

  // Logical operators
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${ret_type}>(-(static_cast<${type}>(${test_value_1}) && static_cast<${type}>(${test_value_2})));
  }
  resVec = testVec1 && testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 && testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 && static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} && testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} && testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} && static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) && testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) && testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${ret_type}>(-(static_cast<${type}>(${test_value_1}) || static_cast<${type}>(${test_value_2})));
  }
  resVec = testVec1 || testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 || testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 || static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} || testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} || testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} || static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) || testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) || testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${ret_type}>(-(!static_cast<${type}>(${test_value_1})));
  }
  resVec = !testVec1;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = !testVec1.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }

  // Relational Operators
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${ret_type}>(-(static_cast<${type}>(${test_value_1}) == static_cast<${type}>(${test_value_2})));
  }
  resVec = testVec1 == testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 == testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 == static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} == testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} == testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} == static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) == testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) == testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${ret_type}>(-(static_cast<${type}>(${test_value_1}) != static_cast<${type}>(${test_value_2})));
  }
  resVec = testVec1 != testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 != testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 != static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} != testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} != testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} != static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) != testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) != testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${ret_type}>(-(static_cast<${type}>(${test_value_1}) <= static_cast<${type}>(${test_value_2})));
  }
  resVec = testVec1 <= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 <= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 <= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} <= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} <= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} <= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) <= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) <= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${ret_type}>(-(static_cast<${type}>(${test_value_1}) >= static_cast<${type}>(${test_value_2})));
  }
  resVec = testVec1 >= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 >= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 >= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} >= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} >= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} >= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) >= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) >= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${ret_type}>(-(static_cast<${type}>(${test_value_1}) < static_cast<${type}>(${test_value_2})));
  }
  resVec = testVec1 < testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 < testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 < static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} < testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} < testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} < static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) < testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) < testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${ret_type}>(-(static_cast<${type}>(${test_value_1}) > static_cast<${type}>(${test_value_2})));
  }
  resVec = testVec1 > testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 > testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 > static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} > testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} > testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} > static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) > testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) > testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
""")

non_fp_bitwise_test_template = Template("""
  /** Performs a test of each vector bitwise operator not available to floating
   *  point types, on a given type, size, and two given values of that type
   */
  auto testVec1 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  sycl::vec<${type}, ${size}> resVec;
  ${type} resArr[${size}];

  // Bitwise operations
#if !${type_is_std_byte}
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) >> static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 >> testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 >> testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 >> static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} >> testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} >> testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} >> static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) >> testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) >> testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) << static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 << testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 << testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 << static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} << testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} << testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} << static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) << testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) << testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
#endif
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) | static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 | testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 | testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 | static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} | testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} | testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} | static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) | testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) | testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) ^ static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 ^ testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 ^ testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 ^ static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} ^ testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} ^ testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} ^ static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) ^ testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) ^ testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) & static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 & testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 & testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 & static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} & testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} & testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} & static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) & testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) & testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = ~static_cast<${type}>(${test_value_1});
  }
  resVec = ~testVec1;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = ~testVec1.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
""")

non_fp_assignment_test_template = Template("""
  /** Performs a test of each vector assignment operator not available to floating
   *  point types, on a given type, size, and two given values of that type
   */
  auto testVec1 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  sycl::vec<${type}, ${size}> resVec;
  ${type} resArr[${size}];

  // Assignment operations
#if !${type_is_std_byte}
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) % static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec %= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec %= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec %= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} %= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} %= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} %= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
#endif
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) | static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec |= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec |= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec |= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} |= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} |= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} |= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) ^ static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec ^= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec ^= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec ^= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} ^= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} ^= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} ^= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) & static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec &= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec &= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec &= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} &= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} &= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} &= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
#if !${type_is_std_byte}
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) >> static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec >>= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec >>= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec >>= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} >>= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} >>= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} >>= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) << static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1;
  resVec <<= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec <<= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec <<= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} <<= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} <<= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} <<= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
#endif
""")

non_fp_arithmetic_test_template = Template("""
  /** Performs a test of each vector operator not available to floating point
   *  types, on a given type, size, and two given values of that type
   */
  auto testVec1 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  sycl::vec<${type}, ${size}> resVec;
  ${type} resArr[${size}];

  // Arithmetic operations
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) & static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 & testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 & testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 & static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} & testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} & testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} & static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) & testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) & testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
#if !${type_is_std_byte}
  for (int i = 0; i < ${size}; ++i) {
    resArr[i] = static_cast<${type}>(${test_value_1}) % static_cast<${type}>(${test_value_2});
  }
  resVec = testVec1 % testVec2;
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 % testVec2.${swizzle};
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 % static_cast<${type}>(${test_value_2});
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} % testVec2;
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} % testVec2.${swizzle};
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} % static_cast<${type}>(${test_value_2});
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) % testVec2;
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) % testVec2.${swizzle};
  if (!check_vector_values_div(resVec, resArr)) {
    resAcc[0] = false;
  }
#endif

""")

subscript_operator_test_template = Template("""
  {
    ${type} data[] = { ${data} };
    const sycl::vec<${type}, ${size}> subscriptVec1(${data});

    // subscript operator assignment
    sycl::vec<${type}, ${size}> subscriptVec2;
    for (int i = 0; i < ${size}; ++i) {
      subscriptVec2[i] = data[i];
    }

    // operator[] type check
    if constexpr (!std::is_same_v<decltype(subscriptVec2[0]), ${type}&>) {
      resAcc[0] = false;
    }

    // const operator[] type check
    if constexpr (!std::is_same_v<decltype(subscriptVec1[0]), const ${type}&>) {
      resAcc[0] = false;
    }

    // check subscript operator value
    for (int i = 0; i < ${size}; ++i) {
      if (subscriptVec1[i] != data[i] || subscriptVec2[i] != data[i]
        || subscriptVec1.${swizzle}[i] != data[i]
        || subscriptVec2.${swizzle}[i] != data[i]
        )
      {
        resAcc[0] = false;
      }
    }
  }
""")

vector_t_operator_test_template = Template("""
#ifdef __SYCL_DEVICE_ONLY__
  // check operator vector_t() const
  {
    ${type} val = ${val};
    const sycl::vec<${type}, 1> testVec(val);
    sycl::vec<${type}, ${size}>::vector_t data = testVec;

    const sycl::vec<${type}, 1> testVec2(data);

    if (!(testVec == testVec2)) {
      resAcc[0] = false;
    }
  }
#endif  // __SYCL_DEVICE_ONLY__
""")

dataT_operator_test_template = Template("""
  // check operator DataT() const
  {
    ${type} val = ${val};
    const sycl::vec<${type}, 1> testVec(${val});
    ${type} data = testVec;
    if (data != val) {
      resAcc[0] = false;
    }

    data = testVec.${swizzle};
    if (data != val) {
      resAcc[0] = false;
    }
  }
""")

assign_dataT_operator_test_template = Template("""
  // check operator=(const DataT&)
  {
    const ${type} val = ${val};
    sycl::vec<${type}, ${size}> testVec;

    testVec = val;
    ${type} resArr[${size}];
    for (int i = 0; i < ${size}; ++i) {
      resArr[i] = val;
    }
    if (!check_vector_values(testVec, resArr)) {
      resAcc[0] = false;
    }
    testVec.${swizzle} = val;
  }
""")

def get_swizzle(size):
    return 'template swizzle<' + ', '.join(
        Data.swizzle_elem_list_dict[size]) + '>()'


def generate_all_type_test(type_str, size):
    test_string = subscript_operator_test_template.substitute(
        type=type_str,
        size=str(size),
        swizzle=get_swizzle(size),
        data=', '.join(make_fp_or_byte_explicit(type_str, Data.vals_list_dict[size])))
    if size == 1:
        test_string += dataT_operator_test_template.substitute(
          type=type_str,
          swizzle=get_swizzle(size),
          val=Data.value_default_dict[type_str])
        test_string += vector_t_operator_test_template.substitute(
          type=type_str,
          size=str(size),
          swizzle=get_swizzle(size),
          val=Data.value_default_dict[type_str])
    test_string += assign_dataT_operator_test_template.substitute(
        type=type_str,
        size=str(size),
        swizzle=get_swizzle(size),
        val=Data.value_default_dict[type_str])
    test_string += all_type_test_template.substitute(
        type=type_str,
        type_is_bool=int(type_str == 'bool'),
        type_is_std_byte=int(type_str == 'std::byte'),
        size=str(size),
        swizzle=get_swizzle(size),
        test_value_1=1,
        test_value_2=2)
    return wrap_with_kernel(type_str,
                            'VEC_ALL_TYPE_OPERATOR_KERNEL_' + type_str + str(size),
                            'All types operator test, sycl::vec<' +
                            type_str + ', ' + str(size) + '>', test_string)


def generate_all_types_specific_return_type_test(type_str, size):
    test_string = specific_return_type_test_template.substitute(
        type=type_str,
        size=str(size),
        swizzle=get_swizzle(size),
        ret_type=Data.opencl_sized_return_type_dict[type_str],
        test_value_1=1,
        test_value_2=2)
    return wrap_with_kernel(
        type_str, 'VEC_SPECIFIC_RETURN_TYPE_OPERATOR_KERNEL_' +
        type_str + str(size),
        'Specific return type operator test, sycl::vec<' + type_str +
        ', ' + str(size) + '>', test_string)


def generate_non_fp_bitwise_test(type_str, size):
    test_string = non_fp_bitwise_test_template.substitute(
        type=type_str,
        type_is_std_byte=int(type_str == 'std::byte'),
        size=str(size),
        swizzle=get_swizzle(size),
        test_value_1=1,
        test_value_2=2)
    return wrap_with_kernel(
        type_str, 'VEC_NON_FP_BITWISE_OPERATOR_KERNEL_' + type_str.replace(
            'sycl::', '').replace(' ', '').replace('std::', '') + str(size),
        'Non FP bitwise operator test, sycl::vec<' + type_str + ', ' +
        str(size) + '>', test_string)


def generate_non_fp_assignment_test(type_str, size):
    test_string = non_fp_assignment_test_template.substitute(
        type=type_str,
        type_is_std_byte=int(type_str == 'std::byte'),
        size=str(size),
        swizzle=get_swizzle(size),
        test_value_1=1,
        test_value_2=2)
    return wrap_with_kernel(
        type_str, 'VEC_NON_FP_ASSIGNMENT_OPERATOR_KERNEL_' + type_str.replace(
            'sycl::', '').replace(' ', '').replace('std::', '') + str(size),
        'Non FP assignment operator test, sycl::vec<' + type_str + ', ' +
        str(size) + '>', test_string)


def generate_non_fp_arithmetic_test(type_str, size):
    test_string = non_fp_arithmetic_test_template.substitute(
        type=type_str,
        type_is_std_byte=int(type_str == 'std::byte'),
        size=str(size),
        swizzle=get_swizzle(size),
        test_value_1=1,
        test_value_2=2)
    return wrap_with_kernel(
        type_str, 'VEC_NON_FP_ARITHMETIC_OPERATOR_KERNEL_' + type_str.replace(
            'sycl::', '').replace(' ', '').replace('std::', '') + str(size),
        'Non FP arithmetic operator test, sycl::vec<' + type_str + ', ' +
        str(size) + '>', test_string)


def generate_operator_tests(type_str, input_file, output_file):
    """"""
    if type_str == 'bool':
        Data.vals_list_dict = cast_to_bool(Data.vals_list_dict)
    test_func_str = ''
    func_calls = ''
    for size in Data.standard_sizes:
        test_str = generate_all_type_test(type_str, size)
        test_func_str += wrap_with_test_func(TEST_NAME + '_ALL_TYPES',
                                             type_str, test_str, str(size))
        func_calls += make_func_call(TEST_NAME + '_ALL_TYPES', type_str,
                                     str(size))
        if not type_str in [
                'float', 'double', 'sycl::half'
        ]:
            test_str = generate_non_fp_assignment_test(type_str, size)
            test_func_str += wrap_with_test_func(
                TEST_NAME + '_NON_FP_ASSIGNMENT', type_str, test_str,
                str(size))
            func_calls += make_func_call(TEST_NAME + '_NON_FP_ASSIGNMENT',
                                         type_str, str(size))
            test_str = generate_non_fp_bitwise_test(type_str, size)
            test_func_str += wrap_with_test_func(TEST_NAME + '_NON_FP_BITWISE',
                                                 type_str, test_str, str(size))
            func_calls += make_func_call(TEST_NAME + '_NON_FP_BITWISE',
                                         type_str, str(size))
            test_str = generate_non_fp_arithmetic_test(type_str, size)
            test_func_str += wrap_with_test_func(
                TEST_NAME + '_NON_FP_ARITHMETIC', type_str, test_str,
                str(size))
            func_calls += make_func_call(TEST_NAME + '_NON_FP_ARITHMETIC',
                                         type_str, str(size))
    write_source_file(test_func_str, func_calls, TEST_NAME, input_file,
                      output_file, type_str)

def main():
    argparser = argparse.ArgumentParser(
        description='Generates vector swizzles opencl test')
    argparser.add_argument(
        'template',
        metavar='<code template path>',
        help='Path to code template')
    argparser.add_argument(
        '-type',
        dest='ty',
        required=True,
        choices=get_types(),
        help='Type to generate the test for')
    argparser.add_argument(
        '-o',
        required=True,
        dest="output",
        metavar='<out file>',
        help='CTS test output')
    args = argparser.parse_args()

    generate_operator_tests(args.ty, args.template, args.output)


if __name__ == '__main__':
    main()
