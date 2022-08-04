/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
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

#include "nd_item_async_work_group_copy_common.h"

#define TEST_NAME nd_item_async_work_group_copy_core

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
    check_type<size_t>(log);
    check_type_and_vec<bool>(log);
    check_type_and_vec<char>(log);
    check_type_and_vec<signed char>(log);
    check_type_and_vec<unsigned char>(log);
    check_type_and_vec<short int>(log);
    check_type_and_vec<unsigned short int>(log);
    check_type_and_vec<int>(log);
    check_type_and_vec<unsigned int>(log);
    check_type_and_vec<long int>(log);
    check_type_and_vec<unsigned long int>(log);
    check_type_and_vec<long long int>(log);
    check_type_and_vec<unsigned long long int>(log);
    check_type_and_vec<float>(log);

    check_type_and_vec<cl::sycl::byte>(log);
    check_type_and_vec<cl::sycl::cl_bool>(log);
    check_type_and_vec<cl::sycl::cl_char>(log);
    check_type_and_vec<cl::sycl::cl_uchar>(log);
    check_type_and_vec<cl::sycl::cl_short>(log);
    check_type_and_vec<cl::sycl::cl_ushort>(log);
    check_type_and_vec<cl::sycl::cl_int>(log);
    check_type_and_vec<cl::sycl::cl_uint>(log);
    check_type_and_vec<cl::sycl::cl_long>(log);
    check_type_and_vec<cl::sycl::cl_ulong>(log);
    check_type_and_vec<cl::sycl::cl_float>(log);

#ifdef INT8_MAX
    if (!std::is_same<cl::sycl::cl_char, std::int8_t>::value)
      check_type_and_vec<std::int8_t>(log);
#endif
#ifdef INT16_MAX
    if (!std::is_same<cl::sycl::cl_short, std::int16_t>::value)
      check_type_and_vec<std::int16_t>(log);
#endif
#ifdef INT32_MAX
    if (!std::is_same<cl::sycl::cl_int, std::int32_t>::value)
      check_type_and_vec<std::int32_t>(log);
#endif
#ifdef INT64_MAX
    if (!std::is_same<cl::sycl::cl_long, std::int64_t>::value)
      check_type_and_vec<std::int64_t>(log);
#endif
#ifdef UINT8_MAX
    if (!std::is_same<cl::sycl::cl_uchar, std::uint8_t>::value)
      check_type_and_vec<std::uint8_t>(log);
#endif
#ifdef UINT16_MAX
    if (!std::is_same<cl::sycl::cl_ushort, std::uint16_t>::value)
      check_type_and_vec<std::uint16_t>(log);
#endif
#ifdef UINT32_MAX
    if (!std::is_same<cl::sycl::cl_uint, std::uint32_t>::value)
      check_type_and_vec<std::uint32_t>(log);
#endif
#ifdef UINT64_MAX
    if (!std::is_same<cl::sycl::cl_ulong, std::uint64_t>::value)
      check_type_and_vec<std::uint64_t>(log);
#endif
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
