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

#include "group_async_work_group_copy_common.h"

#define TEST_NAME group_async_work_group_copy_core

using namespace sycl_cts;
using namespace group_async_work_group_copy;

namespace TEST_NAMESPACE {

class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    check_type<size_t>{}(log, "size_t");
    for_type_and_vectors<check_type, bool>(log,
        "bool");
    for_type_and_vectors<check_type, char>(log,
        "char");
    for_type_and_vectors<check_type, signed char>(log,
        "signed char");
    for_type_and_vectors<check_type, unsigned char>(log,
        "unsigned char");
    for_type_and_vectors<check_type, short int>(log,
        "short");
    for_type_and_vectors<check_type, unsigned short int>(log,
        "unsigned short");
    for_type_and_vectors<check_type, int>(log,
        "int");
    for_type_and_vectors<check_type, unsigned int>(log,
        "unsigned int");
    for_type_and_vectors<check_type, long int>(log,
        "long");
    for_type_and_vectors<check_type, unsigned long int>(log,
        "unsigned long");
    for_type_and_vectors<check_type, long long int>(log,
        "long long");
    for_type_and_vectors<check_type, unsigned long long int>(log,
        "unsigned long long");
    for_type_and_vectors<check_type, float>(log,
        "float");

    for_type_and_vectors<check_type, sycl::byte>(log,
        "sycl::byte");
    for_type_and_vectors<check_type, sycl::cl_bool>(log,
        "sycl::cl_bool");
    for_type_and_vectors<check_type, sycl::cl_char>(log,
        "sycl::cl_char");
    for_type_and_vectors<check_type, sycl::cl_uchar>(log,
        "sycl::cl_uchar");
    for_type_and_vectors<check_type, sycl::cl_short>(log,
        "sycl::cl_short");
    for_type_and_vectors<check_type, sycl::cl_ushort>(log,
        "sycl::cl_ushort");
    for_type_and_vectors<check_type, sycl::cl_int>(log,
        "sycl::cl_int");
    for_type_and_vectors<check_type, sycl::cl_uint>(log,
        "sycl::cl_uint");
    for_type_and_vectors<check_type, sycl::cl_long>(log,
        "sycl::cl_long");
    for_type_and_vectors<check_type, sycl::cl_ulong>(log,
        "sycl::cl_ulong");
    for_type_and_vectors<check_type, sycl::cl_float>(log,
        "sycl::cl_float");

#ifdef INT8_MAX
    if (!std::is_same<sycl::cl_char, std::int8_t>::value)
      for_type_and_vectors<check_type, std::int8_t>(log, "std::int8_t");
#endif
#ifdef INT16_MAX
    if (!std::is_same<sycl::cl_short, std::int16_t>::value)
      for_type_and_vectors<check_type, std::int16_t>(log, "std::int16_t");
#endif
#ifdef INT32_MAX
    if (!std::is_same<sycl::cl_int, std::int32_t>::value)
      for_type_and_vectors<check_type, std::int32_t>(log, "std::int32_t");
#endif
#ifdef INT64_MAX
    if (!std::is_same<sycl::cl_long, std::int64_t>::value)
      for_type_and_vectors<check_type, std::int64_t>(log, "std::int64_t");
#endif
#ifdef UINT8_MAX
    if (!std::is_same<sycl::cl_uchar, std::uint8_t>::value)
      for_type_and_vectors<check_type, std::uint8_t>(log, "std::uint8_t");
#endif
#ifdef UINT16_MAX
    if (!std::is_same<sycl::cl_ushort, std::uint16_t>::value)
      for_type_and_vectors<check_type, std::uint16_t>(log, "std::uint16_t");
#endif
#ifdef UINT32_MAX
    if (!std::is_same<sycl::cl_uint, std::uint32_t>::value)
      for_type_and_vectors<check_type, std::uint32_t>(log, "std::uint32_t");
#endif
#ifdef UINT64_MAX
    if (!std::is_same<sycl::cl_ulong, std::uint64_t>::value)
      for_type_and_vectors<check_type, std::uint64_t>(log, "std::uint64_t");
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
