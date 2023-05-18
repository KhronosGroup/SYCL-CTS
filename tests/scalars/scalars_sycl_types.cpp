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

#include <array>

#define TEST_NAME scalars_sycl_types

namespace scalars_sycl_types__ {
using namespace sycl_cts;

/** Kernel names for fp16 and fp64 types tests for devices which supports these
 * types */
class scalars_sycl_fp16;
class scalars_sycl_fp64;

/** Test SYCL scalar data types are of the minimum sizes and are correctly
 *  signed/unsigned
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  std::string errorStr =
      std::string("The following device type does not have the correct ");

  /** execute the test
   */
  void run(util::logger &log) override {
    {
      unsigned int host_size_t_size = sizeof(size_t);

      auto myQueue = util::get_cts_object::queue();

      auto device = myQueue.get_device();
      bool device_supports_fp16 = device.has(sycl::aspect::fp16);
      bool device_supports_fp64 = device.has(sycl::aspect::fp64);

      // SYCL Integral Data Types
      if (!check_type_min_size<char>(1)) {
        FAIL(log,
             "The following host type does not have the correct size: char");
      }
      check_type_min_size_sign_log<unsigned char>(log, 1, false,
                                                  "unsigned char");
      check_type_min_size_sign_log<signed char>(log, 1, true, "signed char");
      check_type_min_size_sign_log<unsigned short int>(log, 2, false,
                                                       "unsigned short int");
      check_type_min_size_sign_log<short int>(log, 2, true, "short int");
      check_type_min_size_sign_log<unsigned int>(log, 2, false, "unsigned int");
      check_type_min_size_sign_log<int>(log, 2, true, "int");
      check_type_min_size_sign_log<unsigned long int>(log, 4, false,
                                                      "unsigned long int");
      check_type_min_size_sign_log<long int>(log, 4, true, "long int");
      check_type_min_size_sign_log<unsigned long long int>(
          log, 8, false, "unsigned long long int");
      check_type_min_size_sign_log<long long int>(log, 8, true,
                                                  "long long int");
      check_type_min_size_sign_log<size_t>(log, host_size_t_size, false,
                                           "size_t");

      // SYCL Floating Point Data Types
      check_type_min_size_sign_log<sycl::half>(log, 2, true, "sycl::half");
      check_type_min_size_sign_log<float>(log, 4, true, "float");
      check_type_min_size_sign_log<double>(log, 8, true, "double");

      std::array<bool, 15> signResults;
      std::array<bool, 16> sizeResults;
      {
        sycl::buffer<bool, 1> bufSignResult(signResults.data(),
                                            sycl::range<1>(signResults.size()));
        sycl::buffer<bool, 1> bufSizeResult(sizeResults.data(),
                                            sycl::range<1>(sizeResults.size()));

        myQueue.submit([&](sycl::handler &cgh) {
          auto accSignResult =
              bufSignResult.get_access<sycl::access_mode::read_write>(cgh);
          auto accSizeResult =
              bufSizeResult.get_access<sycl::access_mode::read_write>(cgh);

          cgh.single_task<TEST_NAME>([=] {
            // SYCL Integral Data Types
            // signs
            accSignResult[0] = check_type_sign<unsigned char>(false);
            accSignResult[1] = check_type_sign<signed char>(true);
            accSignResult[2] = check_type_sign<unsigned short int>(false);
            accSignResult[3] = check_type_sign<short int>(true);
            accSignResult[4] = check_type_sign<unsigned int>(false);
            accSignResult[5] = check_type_sign<int>(true);
            accSignResult[6] = check_type_sign<unsigned long int>(false);
            accSignResult[7] = check_type_sign<long int>(true);
            accSignResult[8] = check_type_sign<unsigned long long int>(false);
            accSignResult[9] = check_type_sign<long long int>(true);
            accSignResult[10] = check_type_sign<size_t>(false);
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS  // sycl::byte is deprecated
            accSignResult[11] = check_type_sign<sycl::byte>(false);
#endif
            accSignResult[13] = check_type_sign<float>(true);

            // sizes
            accSizeResult[0] = check_type_min_size<char>(1);
            accSizeResult[1] = check_type_min_size<unsigned char>(1);
            accSizeResult[2] = check_type_min_size<signed char>(1);
            accSizeResult[3] = check_type_min_size<unsigned short int>(2);
            accSizeResult[4] = check_type_min_size<short int>(2);
            accSizeResult[5] = check_type_min_size<unsigned int>(2);
            accSizeResult[6] = check_type_min_size<int>(2);
            accSizeResult[7] = check_type_min_size<unsigned long int>(4);
            accSizeResult[8] = check_type_min_size<long int>(4);
            accSizeResult[9] = check_type_min_size<unsigned long long int>(8);
            accSizeResult[10] = check_type_min_size<long long int>(8);
            accSizeResult[11] = check_type_min_size<size_t>(host_size_t_size);
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS  // sycl::byte is deprecated
            accSizeResult[12] = check_type_min_size<sycl::byte>(1);
#endif
            accSizeResult[14] = check_type_min_size<float>(4);
          });
        });

        if (device_supports_fp16) {
          myQueue.submit([&](sycl::handler &cgh) {
            auto accSignResult =
                bufSignResult.get_access<sycl::access_mode::read_write>(cgh);
            auto accSizeResult =
                bufSizeResult.get_access<sycl::access_mode::read_write>(cgh);

            cgh.single_task<scalars_sycl_fp16>([=] {
              // Floating Point 16 Interop Data Type
              // sign
              accSignResult[12] = check_type_sign<sycl::half>(true);

              // size
              accSizeResult[13] = check_type_min_size<sycl::half>(2);
            });
          });
        }

        if (device_supports_fp64) {
          myQueue
              .submit([&](sycl::handler &cgh) {
                auto accSignResult =
                    bufSignResult.get_access<sycl::access_mode::read_write>(
                        cgh);
                auto accSizeResult =
                    bufSizeResult.get_access<sycl::access_mode::read_write>(
                        cgh);

                cgh.single_task<scalars_sycl_fp64>([=] {
                  // Floating Point 64 Interop Data Type
                  // sign
                  accSignResult[14] = check_type_sign<double>(true);

                  // size
                  accSizeResult[15] = check_type_min_size<double>(8);
                });
              })
              .wait_and_throw();
        }
      }

      // signs
      if (!signResults[0]) {
        FAIL(log, errorStr + "sign: unsigned char");
      }
      if (!signResults[1]) {
        FAIL(log, errorStr + "sign: signed char");
      }
      if (!signResults[2]) {
        FAIL(log, errorStr + "sign: unsigned short int");
      }
      if (!signResults[3]) {
        FAIL(log, errorStr + "sign: short int");
      }
      if (!signResults[4]) {
        FAIL(log, errorStr + "sign: unsigned int");
      }
      if (!signResults[5]) {
        FAIL(log, errorStr + "sign: int");
      }
      if (!signResults[6]) {
        FAIL(log, errorStr + "sign: unsigned long int");
      }
      if (!signResults[7]) {
        FAIL(log, errorStr + "sign: long int");
      }
      if (!signResults[8]) {
        FAIL(log, errorStr + "sign: unsigned long long int");
      }
      if (!signResults[9]) {
        FAIL(log, errorStr + "sign: long long int");
      }
      if (!signResults[10]) {
        FAIL(log, errorStr + "sign: size_t");
      }
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS  // sycl::byte is deprecated
      if (!signResults[11]) {
        FAIL(log, errorStr + "sign: sycl::byte");
      }
#endif
      if (!signResults[12] && device_supports_fp16) {
        FAIL(log, errorStr + "sign: sycl::half");
      }
      if (!signResults[13]) {
        FAIL(log, errorStr + "sign: float");
      }
      if (!signResults[14] && device_supports_fp64) {
        FAIL(log, errorStr + "sign: double");
      }

      // sizes
      if (!sizeResults[0]) {
        FAIL(log, errorStr + "size: char");
      }
      if (!sizeResults[1]) {
        FAIL(log, errorStr + "size: unsigned char");
      }
      if (!sizeResults[2]) {
        FAIL(log, errorStr + "size: signed char");
      }
      if (!sizeResults[3]) {
        FAIL(log, errorStr + "size: unsigned short int");
      }
      if (!sizeResults[4]) {
        FAIL(log, errorStr + "size: short int");
      }
      if (!sizeResults[5]) {
        FAIL(log, errorStr + "size: unsigned int");
      }
      if (!sizeResults[6]) {
        FAIL(log, errorStr + "size: int");
      }
      if (!sizeResults[7]) {
        FAIL(log, errorStr + "size: unsigned long int");
      }
      if (!sizeResults[8]) {
        FAIL(log, errorStr + "size: long int");
      }
      if (!sizeResults[9]) {
        FAIL(log, errorStr + "size: unsigned long long int");
      }
      if (!sizeResults[10]) {
        FAIL(log, errorStr + "size: long long int");
      }
      if (!sizeResults[11]) {
        FAIL(log, errorStr + "size: size_t");
      }
#if SYCL_CTS_ENABLE_DEPRECATED_FEATURES_TESTS  // sycl::byte is deprecated
      if (!sizeResults[12]) {
        FAIL(log, errorStr + "size: sycl::byte");
      }
#endif
      if (!sizeResults[13] && device_supports_fp16) {
        FAIL(log, errorStr + "size: sycl::half");
      }
      if (!sizeResults[14]) {
        FAIL(log, errorStr + "size: float");
      }
      if (!sizeResults[15] && device_supports_fp64) {
        FAIL(log, errorStr + "size: double");
      }

      myQueue.wait_and_throw();
    }

    // Check sycl::half limits specialization
    {
      INFO("Check that std::numeric_limits is specialized for sycl::half type");
      CHECK(std::numeric_limits<sycl::half>::is_specialized);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace scalars_sycl_types__ */
