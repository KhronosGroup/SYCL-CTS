# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
from string import Template
sys.path.append('../common/')
from common_python_vec import (Data, wrap_with_kernel, wrap_with_test_func,
                               make_func_call, write_source_file)

TEST_NAME = 'OPERATORS'

all_type_test_template = Template("""
  /** Performs a test of each vector operator available to all types,
   *  on a given type, size, and two given values of that type
   */
  auto testVec1 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec1Copy = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  cl::sycl::vec<${type}, ${size}> resVec;
  ${type} resArr[${size}];
  ${type} resArr2[${size}];

  // Arithmetic operators
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
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 / testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1 / static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} / testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} / testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1.${swizzle} / static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) / testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = static_cast<${type}>(${test_value_1}) / testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }

  // Post and pre increment and decrement
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

  // Assignment operators
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
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec /= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec /= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} /= testVec2;
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} /= testVec2.${swizzle};
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
  resVec = testVec1;
  resVec.${swizzle} /= static_cast<${type}>(${test_value_2});
  if (!check_vector_values(resVec, resArr)) {
    resAcc[0] = false;
  }
""")

specific_return_type_test_template = Template("""
  /** Tests each logical and relational operator available to vector types
   */
  auto testVec1 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  cl::sycl::vec<${ret_type}, ${size}> resVec;
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
  auto testVec1 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  cl::sycl::vec<${type}, ${size}> resVec;
  ${type} resArr[${size}];

  // Bitwise operations
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
  auto testVec1 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  cl::sycl::vec<${type}, ${size}> resVec;
  ${type} resArr[${size}];

  // Assignment operations
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
""")

non_fp_arithmetic_test_template = Template("""
  /** Performs a test of each vector operator not available to floating point
   *  types, on a given type, size, and two given values of that type
   */
  auto testVec1 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_1}));
  auto testVec2 = cl::sycl::vec<${type}, ${size}>(static_cast<${type}>(${test_value_2}));
  cl::sycl::vec<${type}, ${size}> resVec;
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
""")


def get_swizzle(size):
    return 'template swizzle<' + ', '.join(
        Data.swizzle_elem_list_dict[size]) + '>()'


def generate_all_type_test(type_str, size):
    test_string = all_type_test_template.substitute(
        type=type_str,
        size=str(size),
        swizzle=get_swizzle(size),
        test_value_1=1,
        test_value_2=2)
    return wrap_with_kernel(type_str,
                            'VEC_ALL_TYPE_OPERATOR_KERNEL_' + type_str.replace(
                                'cl::sycl::', '').replace(' ', '') + str(size),
                            'All types operator test, cl::sycl::vec<' +
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
        type_str,
        'VEC_SPECIFIC_RETURN_TYPE_OPERATOR_KERNEL_' + type_str.replace(
            'cl::sycl::', '').replace(' ', '') + str(size),
        'Specific return type operator test, cl::sycl::vec<' + type_str + ', '
        + str(size) + '>', test_string)


def generate_non_fp_bitwise_test(type_str, size):
    test_string = non_fp_bitwise_test_template.substitute(
        type=type_str,
        size=str(size),
        swizzle=get_swizzle(size),
        test_value_1=1,
        test_value_2=2)
    return wrap_with_kernel(
        type_str, 'VEC_NON_FP_BITWISE_OPERATOR_KERNEL_' + type_str.replace(
            'cl::sycl::', '').replace(' ', '') + str(size),
        'Non FP bitwise operator test, cl::sycl::vec<' + type_str + ', ' +
        str(size) + '>', test_string)


def generate_non_fp_assignment_test(type_str, size):
    test_string = non_fp_assignment_test_template.substitute(
        type=type_str,
        size=str(size),
        swizzle=get_swizzle(size),
        test_value_1=1,
        test_value_2=2)
    return wrap_with_kernel(
        type_str, 'VEC_NON_FP_ASSIGNMENT_OPERATOR_KERNEL_' + type_str.replace(
            'cl::sycl::', '').replace(' ', '') + str(size),
        'Non FP assignment operator test, cl::sycl::vec<' + type_str + ', ' +
        str(size) + '>', test_string)


def generate_non_fp_arithmetic_test(type_str, size):
    test_string = non_fp_arithmetic_test_template.substitute(
        type=type_str,
        size=str(size),
        swizzle=get_swizzle(size),
        test_value_1=1,
        test_value_2=2)
    return wrap_with_kernel(
        type_str, 'VEC_NON_FP_ARITHMETIC_OPERATOR_KERNEL_' + type_str.replace(
            'cl::sycl::', '').replace(' ', '') + str(size),
        'Non FP arithmetic operator test, cl::sycl::vec<' + type_str + ', ' +
        str(size) + '>', test_string)


def generate_operator_tests(input_file, output_file):
    """"""

    # Test with type_str='char'
    test_str = ''
    test_func_str = ''
    func_calls = ''
    for size in Data.standard_sizes:
        test_str = generate_all_type_test('char', size)
        test_func_str += wrap_with_test_func(TEST_NAME + '_ALL_TYPES', 'char',
                                             test_str, str(size))
        func_calls += make_func_call(TEST_NAME + '_ALL_TYPES', 'char',
                                     str(size))
        test_str = generate_non_fp_assignment_test('char', size)
        test_func_str += wrap_with_test_func(TEST_NAME + '_NON_FP_ASSIGNMENT',
                                             'char', test_str, str(size))
        func_calls += make_func_call(TEST_NAME + '_NON_FP_ASSIGNMENT', 'char',
                                     str(size))
        test_str = generate_non_fp_bitwise_test('char', size)
        test_func_str += wrap_with_test_func(TEST_NAME + '_NON_FP_BITWISE',
                                             'char', test_str, str(size))
        func_calls += make_func_call(TEST_NAME + '_NON_FP_BITWISE', 'char',
                                     str(size))
        test_str = generate_non_fp_arithmetic_test('char', size)
        test_func_str += wrap_with_test_func(TEST_NAME + '_NON_FP_ARITHMETIC',
                                             'char', test_str, str(size))
        func_calls += make_func_call(TEST_NAME + '_NON_FP_ARITHMETIC', 'char',
                                     str(size))
    write_source_file(test_func_str, func_calls, TEST_NAME, input_file,
                      output_file, 'char')

    for type_name in Data.standard_types:
        for sign in Data.signs:
            if (type_name == 'float' or type_name == 'double'
                    or type_name == 'cl::sycl::half') and sign is False:
                continue
            type_str = Data.standard_type_dict[(sign, type_name)]
            test_func_str = ''
            func_calls = ''
            for size in Data.standard_sizes:
                test_str = generate_all_type_test(type_str, size)
                test_func_str += wrap_with_test_func(TEST_NAME + '_ALL_TYPES',
                                                     type_str, test_str,
                                                     str(size))
                func_calls += make_func_call(TEST_NAME + '_ALL_TYPES',
                                             type_str, str(size))
                if (type_name != 'float' and type_name != 'double'
                        and type_name != 'cl::sycl::half'):
                    test_str = generate_non_fp_assignment_test(type_str, size)
                    test_func_str += wrap_with_test_func(
                        TEST_NAME + '_NON_FP_ASSIGNMENT', type_str, test_str,
                        str(size))
                    func_calls += make_func_call(
                        TEST_NAME + '_NON_FP_ASSIGNMENT', type_str, str(size))
                    test_str = generate_non_fp_bitwise_test(type_str, size)
                    test_func_str += wrap_with_test_func(
                        TEST_NAME + '_NON_FP_BITWISE', type_str, test_str,
                        str(size))
                    func_calls += make_func_call(TEST_NAME + '_NON_FP_BITWISE',
                                                 type_str, str(size))
                    test_str = generate_non_fp_arithmetic_test(type_str, size)
                    test_func_str += wrap_with_test_func(
                        TEST_NAME + '_NON_FP_ARITHMETIC', type_str, test_str,
                        str(size))
                    func_calls += make_func_call(
                        TEST_NAME + '_NON_FP_ARITHMETIC', type_str, str(size))
            write_source_file(test_func_str, func_calls, TEST_NAME, input_file,
                              output_file, type_str)

    for type_name in Data.opencl_types:
        for sign in Data.signs:
            if (type_name == 'cl::sycl::cl_float'
                    or type_name == 'cl::sycl::cl_double'
                    or type_name == 'cl::sycl::cl_half') and sign is False:
                continue
            type_str = Data.opencl_type_dict[(sign, type_name)]
            test_func_str = ''
            func_calls = ''
            for size in Data.standard_sizes:
                test_str = generate_all_type_test(type_str, size)
                test_func_str += wrap_with_test_func(TEST_NAME + '_ALL_TYPES',
                                                     type_str, test_str,
                                                     str(size))
                func_calls += make_func_call(TEST_NAME + '_ALL_TYPES',
                                             type_str, str(size))
                test_str = generate_all_types_specific_return_type_test(
                    type_str, size)
                test_func_str += wrap_with_test_func(
                    TEST_NAME + '_SPECIFIC_RETURN_TYPES', type_str, test_str,
                    str(size))
                func_calls += make_func_call(
                    TEST_NAME + '_SPECIFIC_RETURN_TYPES', type_str, str(size))
                if (type_name != 'cl::sycl::cl_float'
                        and type_name != 'cl::sycl::cl_double'
                        and type_name != 'cl::sycl::cl_half'):
                    test_str = generate_non_fp_assignment_test(type_str, size)
                    test_func_str += wrap_with_test_func(
                        TEST_NAME + '_NON_FP_ASSIGNMENT', type_str, test_str,
                        str(size))
                    func_calls += make_func_call(
                        TEST_NAME + '_NON_FP_ASSIGNMENT', type_str, str(size))
                    test_str = generate_non_fp_bitwise_test(type_str, size)
                    test_func_str += wrap_with_test_func(
                        TEST_NAME + '_NON_FP_BITWISE', type_str, test_str,
                        str(size))
                    func_calls += make_func_call(TEST_NAME + '_NON_FP_BITWISE',
                                                 type_str, str(size))
                    test_str = generate_non_fp_arithmetic_test(type_str, size)
                    test_func_str += wrap_with_test_func(
                        TEST_NAME + '_NON_FP_ARITHMETIC', type_str, test_str,
                        str(size))
                    func_calls += make_func_call(
                        TEST_NAME + '_NON_FP_ARITHMETIC', type_str, str(size))
            write_source_file(test_func_str, func_calls, TEST_NAME, input_file,
                              output_file, type_str)


def main():
    generate_operator_tests('../common/vector.template',
                            'vector_operators.cpp')


if __name__ == '__main__':
    main()
