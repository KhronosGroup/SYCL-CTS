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
#include "helpers.h"

#define TEST_NAME scalars_interopability_types

namespace scalars_interopability_types__ {
using namespace sycl_cts;

/** Kernel names for fp16 and fp64 types tests for devices which supports these
 * types */
class scalars_interopability_fp16;
class scalars_interopability_fp64;

/** Test SYCL OpenCL interop scalar data types are of the minimum sizes and are
 *  correctly signed/unsigned
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  std::string errorStr =
      std::string("The following device type does not have the correct ");

  /** execute the test
   */
  void run(util::logger& log) override {
    {
      auto myQueue = util::get_cts_object::queue();

      auto device = myQueue.get_device();
      bool device_supports_fp16 = device.has(sycl::aspect::fp16);
      bool device_supports_fp64 = device.has(sycl::aspect::fp64);

      // Integral Interop Data Types
      if (!check_type_min_size<cl_bool>(1)) {
        FAIL(log,
             "The following host type does not have the correct size: cl_bool");
      }
      check_type_min_size_sign_log<sycl::opencl::cl_bool>(
          log, 1, false, "sycl::opencl::cl_bool");
      check_type_min_size_sign_log<sycl::opencl::cl_char>(
          log, 1, true, "sycl::opencl::cl_char");
      check_type_min_size_sign_log<sycl::opencl::cl_uchar>(
          log, 1, false, "sycl::opencl::cl_uchar");
      check_type_min_size_sign_log<sycl::opencl::cl_short>(
          log, 2, true, "sycl::opencl::cl_short");
      check_type_min_size_sign_log<sycl::opencl::cl_ushort>(
          log, 2, false, "sycl::opencl::cl_ushort");
      check_type_min_size_sign_log<sycl::opencl::cl_int>(
          log, 4, true, "sycl::opencl::cl_int");
      check_type_min_size_sign_log<sycl::opencl::cl_uint>(
          log, 4, false, "sycl::opencl::cl_uint");
      check_type_min_size_sign_log<sycl::opencl::cl_long>(
          log, 8, true, "sycl::opencl::cl_long");
      check_type_min_size_sign_log<sycl::opencl::cl_ulong>(
          log, 8, false, "sycl::opencl::cl_ulong");

      // Floating Point Interop Data Types
      check_type_min_size_sign_log<sycl::opencl::cl_half>(
          log, 2, true, "sycl::opencl::cl_half");
      check_type_min_size_sign_log<sycl::opencl::cl_float>(
          log, 4, true, "sycl::opencl::cl_float");
      check_type_min_size_sign_log<sycl::opencl::cl_double>(
          log, 8, true, "sycl::opencl::cl_double");

      bool signResults[11];
      bool sizeResults[12];
      {
        sycl::buffer<bool, 1> bufSignResult(signResults, sycl::range<1>(11));
        sycl::buffer<bool, 1> bufSizeResult(sizeResults, sycl::range<1>(12));

        myQueue.submit([&](sycl::handler& cgh) {
          auto accSignResult =
              bufSignResult.get_access<sycl::access_mode::read_write>(cgh);
          auto accSizeResult =
              bufSizeResult.get_access<sycl::access_mode::read_write>(cgh);

          cgh.single_task<TEST_NAME>([=] {
            // Integral Interop Data Types
            // signs
            accSignResult[0] = check_type_sign<sycl::opencl::cl_char>(true);
            accSignResult[1] = check_type_sign<sycl::opencl::cl_uchar>(false);
            accSignResult[2] = check_type_sign<sycl::opencl::cl_short>(true);
            accSignResult[3] = check_type_sign<sycl::opencl::cl_ushort>(false);
            accSignResult[4] = check_type_sign<sycl::opencl::cl_int>(true);
            accSignResult[5] = check_type_sign<sycl::opencl::cl_uint>(false);
            accSignResult[6] = check_type_sign<sycl::opencl::cl_long>(true);
            accSignResult[7] = check_type_sign<sycl::opencl::cl_ulong>(false);

            // sizes
            accSizeResult[0] = check_type_min_size<sycl::opencl::cl_bool>(1);
            accSizeResult[1] = check_type_min_size<sycl::opencl::cl_char>(1);
            accSizeResult[2] = check_type_min_size<sycl::opencl::cl_uchar>(1);
            accSizeResult[3] = check_type_min_size<sycl::opencl::cl_short>(2);
            accSizeResult[4] = check_type_min_size<sycl::opencl::cl_ushort>(2);
            accSizeResult[5] = check_type_min_size<sycl::opencl::cl_int>(4);
            accSizeResult[6] = check_type_min_size<sycl::opencl::cl_uint>(4);
            accSizeResult[7] = check_type_min_size<sycl::opencl::cl_long>(8);
            accSizeResult[8] = check_type_min_size<sycl::opencl::cl_ulong>(8);

            // Floating Point Interop Data Type
            // sign
            accSignResult[9] = check_type_sign<sycl::opencl::cl_float>(true);

            // size
            accSizeResult[10] = check_type_min_size<sycl::opencl::cl_float>(4);
          });
        });

        if (device_supports_fp16) {
          myQueue.submit([&](sycl::handler& cgh) {
            auto accSignResult =
                bufSignResult.get_access<sycl::access_mode::read_write>(cgh);
            auto accSizeResult =
                bufSizeResult.get_access<sycl::access_mode::read_write>(cgh);

            cgh.single_task<scalars_interopability_fp16>([=] {
              // Floating Point 16 Interop Data Type
              // sign
              accSignResult[8] = check_type_sign<sycl::opencl::cl_half>(true);

              // size
              accSizeResult[9] = check_type_min_size<sycl::opencl::cl_half>(2);
            });
          });
        }

        if (device_supports_fp64) {
          myQueue
              .submit([&](sycl::handler& cgh) {
                auto accSignResult =
                    bufSignResult.get_access<sycl::access_mode::read_write>(
                        cgh);
                auto accSizeResult =
                    bufSizeResult.get_access<sycl::access_mode::read_write>(
                        cgh);

                cgh.single_task<scalars_interopability_fp64>([=] {
                  // Floating Point 64 Interop Data Type
                  // sign
                  accSignResult[10] =
                      check_type_sign<sycl::opencl::cl_double>(true);

                  // size
                  accSizeResult[11] =
                      check_type_min_size<sycl::opencl::cl_double>(8);
                });
              })
              .wait_and_throw();
        }
      }

      // signs
      if (!signResults[0]) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_char");
      }
      if (!signResults[1]) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_uchar");
      }
      if (!signResults[2]) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_short");
      }
      if (!signResults[3]) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_ushort");
      }
      if (!signResults[4]) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_int");
      }
      if (!signResults[5]) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_uint");
      }
      if (!signResults[6]) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_long");
      }
      if (!signResults[7]) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_ulong");
      }
      if (!signResults[8] && device_supports_fp16) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_half");
      }
      if (!signResults[9]) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_float");
      }
      if (!signResults[10] && device_supports_fp64) {
        FAIL(log, errorStr + "sign: sycl::opencl::cl_double");
      }

      // sizes
      if (!sizeResults[0]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_bool");
      }
      if (!sizeResults[1]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_char");
      }
      if (!sizeResults[2]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_uchar");
      }
      if (!sizeResults[3]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_short");
      }
      if (!sizeResults[4]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_ushort");
      }
      if (!sizeResults[5]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_int");
      }
      if (!sizeResults[6]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_uint");
      }
      if (!sizeResults[7]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_long");
      }
      if (!sizeResults[8]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_ulong");
      }
      if (!sizeResults[9] && device_supports_fp16) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_half");
      }
      if (!sizeResults[10]) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_float");
      }
      if (!sizeResults[11] && device_supports_fp64) {
        FAIL(log, errorStr + "size: sycl::opencl::cl_double");
      }

      myQueue.wait_and_throw();
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace scalars_interopability_types__ */
