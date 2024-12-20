/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
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

#include <algorithm>
#include <array>
#include <string>

#define TEST_NAME device_event_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class device_event_wait;

/** tests the api for sycl::device_event
 */
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
    {
      /** check wait() member function
       */
      auto testQueue = util::get_cts_object::queue();

      constexpr size_t bufferSize = 512;
      constexpr size_t sampleIndex = bufferSize / 2;
      constexpr int referenceValue = 1234;

      std::array<int, bufferSize> data;
      std::fill(data.begin(), data.end(), referenceValue);

      bool error = false;
      {
        sycl::range<1> range(1);
        sycl::range<1> dataRange(bufferSize);
        sycl::buffer<int, 1> buf(data.data(), dataRange);
        sycl::buffer<bool, 1> errBuf(&error, range);

        testQueue.submit([&](sycl::handler &cgh) {

          auto globalAcc = buf.get_access<sycl::access_mode::read_write>(cgh);
          auto errorAcc = errBuf.get_access<sycl::access_mode::write>(cgh);
          auto localAcc = sycl::local_accessor<int, 1>(dataRange, cgh);

          cgh.parallel_for<class device_event_wait>(
              sycl::nd_range<1>(range, range),
              [=](sycl::nd_item<1> ndItem) {
// FIXME: re-enable when sycl::access::decorated and get_multi_ptr is
// implemented
#if !SYCL_CTS_COMPILING_WITH_ADAPTIVECPP
                // Run asynchronous copy for full buffer
                sycl::device_event deviceEvent = ndItem.async_work_group_copy(
                    localAcc
                        .template get_multi_ptr<sycl::access::decorated::yes>(),
                    globalAcc
                        .template get_multi_ptr<sycl::access::decorated::yes>(),
                    bufferSize);

                deviceEvent.wait();
#endif
                // Check sample was updated
                if (localAcc[sampleIndex] != referenceValue) {
                  errorAcc[0] = true;
                }
              });
        });
      }
      if (error) {
        FAIL(log, "sycl::device_event async_work_group_copy failed");
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
