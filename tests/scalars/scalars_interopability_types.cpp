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

#include "../common/common.h"

#define TEST_NAME scalars_interopability_types

namespace scalars_interopability_types__ {
using namespace sycl_cts;

/** Test SYCL OpenCL interop scalar data types are of the minimum sizes and are
 *  correctly signed/unsigned
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  std::string errorStr = std::string(
      "The following device type does not have the correct ");

  /** execute the test
   */
  void run(util::logger &log) override {
    {
      // Integral Interop Data Types
      if (!check_type_min_size<cl_bool>(1)) {
        FAIL(log,
             "The following host type does not have the correct size: cl_bool");
      }
      check_type_min_size_sign_log<sycl::cl_bool>(log, 1, false,
                                                      "sycl::cl_bool");
      check_type_min_size_sign_log<sycl::cl_char>(log, 1, true,
                                                      "sycl::cl_char");
      check_type_min_size_sign_log<sycl::cl_uchar>(log, 1, false,
                                                       "sycl::cl_uchar");
      check_type_min_size_sign_log<sycl::cl_short>(log, 2, true,
                                                       "sycl::cl_short");
      check_type_min_size_sign_log<sycl::cl_ushort>(log, 2, false,
                                                        "sycl::cl_ushort");
      check_type_min_size_sign_log<sycl::cl_int>(log, 4, true,
                                                     "sycl::cl_int");
      check_type_min_size_sign_log<sycl::cl_uint>(log, 4, false,
                                                      "sycl::cl_uint");
      check_type_min_size_sign_log<sycl::cl_long>(log, 8, true,
                                                      "sycl::cl_long");
      check_type_min_size_sign_log<sycl::cl_ulong>(log, 8, false,
                                                       "sycl::cl_ulong");

      // Floating Point Interop Data Types
      check_type_min_size_sign_log<sycl::cl_half>(log, 2, true,
                                                      "sycl::cl_half");
      check_type_min_size_sign_log<sycl::cl_float>(log, 4, true,
                                                       "sycl::cl_float");
      check_type_min_size_sign_log<sycl::cl_double>(log, 8, true,
                                                        "sycl::cl_double");

      auto myQueue = util::get_cts_object::queue();

      bool signResults[11];
      bool sizeResults[12];
      {
        sycl::buffer<bool, 1> bufSignResult(signResults,
                                                sycl::range<1>(11));
        sycl::buffer<bool, 1> bufSizeResult(sizeResults,
                                                sycl::range<1>(12));

        myQueue.submit([&](sycl::handler &cgh) {
          auto accSignResult =
              bufSignResult.get_access<sycl::access_mode::read_write>(cgh);
          auto accSizeResult =
              bufSizeResult.get_access<sycl::access_mode::read_write>(cgh);

          cgh.single_task<TEST_NAME>([=]() {
            // Integral Interop Data Types
            // signs
            accSignResult[0] = check_type_sign<sycl::cl_char>(true);
            accSignResult[1] = check_type_sign<sycl::cl_uchar>(false);
            accSignResult[2] = check_type_sign<sycl::cl_short>(true);
            accSignResult[3] = check_type_sign<sycl::cl_ushort>(false);
            accSignResult[4] = check_type_sign<sycl::cl_int>(true);
            accSignResult[5] = check_type_sign<sycl::cl_uint>(false);
            accSignResult[6] = check_type_sign<sycl::cl_long>(true);
            accSignResult[7] = check_type_sign<sycl::cl_ulong>(false);

            // sizes
            accSizeResult[0] = check_type_min_size<sycl::cl_bool>(1);
            accSizeResult[1] = check_type_min_size<sycl::cl_char>(1);
            accSizeResult[2] = check_type_min_size<sycl::cl_uchar>(1);
            accSizeResult[3] = check_type_min_size<sycl::cl_short>(2);
            accSizeResult[4] = check_type_min_size<sycl::cl_ushort>(2);
            accSizeResult[5] = check_type_min_size<sycl::cl_int>(4);
            accSizeResult[6] = check_type_min_size<sycl::cl_uint>(4);
            accSizeResult[7] = check_type_min_size<sycl::cl_long>(8);
            accSizeResult[8] = check_type_min_size<sycl::cl_ulong>(8);

            // Floating Point Interop Data Types
            // signs
            accSignResult[8] = check_type_sign<sycl::cl_half>(true);
            accSignResult[9] = check_type_sign<sycl::cl_float>(true);
            accSignResult[10] = check_type_sign<sycl::cl_double>(true);

            // sizes
            accSizeResult[9] = check_type_min_size<sycl::cl_half>(2);
            accSizeResult[10] = check_type_min_size<sycl::cl_float>(4);
            accSizeResult[11] = check_type_min_size<sycl::cl_double>(8);

          });
        });
      }

      // signs
      if (!signResults[0]) {
        FAIL(log, errorStr + "sign: sycl::cl_char");
      }
      if (!signResults[1]) {
        FAIL(log, errorStr + "sign: sycl::cl_uchar");
      }
      if (!signResults[2]) {
        FAIL(log, errorStr + "sign: sycl::cl_short");
      }
      if (!signResults[3]) {
        FAIL(log, errorStr + "sign: sycl::cl_ushort");
      }
      if (!signResults[4]) {
        FAIL(log, errorStr + "sign: sycl::cl_int");
      }
      if (!signResults[5]) {
        FAIL(log, errorStr + "sign: sycl::cl_uint");
      }
      if (!signResults[6]) {
        FAIL(log, errorStr + "sign: sycl::cl_long");
      }
      if (!signResults[7]) {
        FAIL(log, errorStr + "sign: sycl::cl_ulong");
      }
      if (!signResults[8]) {
        FAIL(log, errorStr + "sign: sycl::cl_half");
      }
      if (!signResults[9]) {
        FAIL(log, errorStr + "sign: sycl::cl_float");
      }
      if (!signResults[10]) {
        FAIL(log, errorStr + "sign: sycl::cl_double");
      }

      // sizes
      if (!sizeResults[0]) {
        FAIL(log, errorStr + "size: sycl::cl_bool");
      }
      if (!sizeResults[1]) {
        FAIL(log, errorStr + "size: sycl::cl_char");
      }
      if (!sizeResults[2]) {
        FAIL(log, errorStr + "size: sycl::cl_uchar");
      }
      if (!sizeResults[3]) {
        FAIL(log, errorStr + "size: sycl::cl_short");
      }
      if (!sizeResults[4]) {
        FAIL(log, errorStr + "size: sycl::cl_ushort");
      }
      if (!sizeResults[5]) {
        FAIL(log, errorStr + "size: sycl::cl_int");
      }
      if (!sizeResults[6]) {
        FAIL(log, errorStr + "size: sycl::cl_uint");
      }
      if (!sizeResults[7]) {
        FAIL(log, errorStr + "size: sycl::cl_long");
      }
      if (!sizeResults[8]) {
        FAIL(log, errorStr + "size: sycl::cl_ulong");
      }
      if (!sizeResults[9]) {
        FAIL(log, errorStr + "size: sycl::cl_half");
      }
      if (!sizeResults[10]) {
        FAIL(log, errorStr + "size: sycl::cl_float");
      }
      if (!sizeResults[11]) {
        FAIL(log, errorStr + "size: sycl::cl_double");
      }

      myQueue.wait_and_throw();
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace scalars_interopability_types__ */
