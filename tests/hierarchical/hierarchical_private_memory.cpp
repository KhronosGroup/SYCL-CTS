/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
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

#define TEST_NAME hierarchical_private_memory

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test cl::sycl::range::get(int index) return size_t
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
    try {
      constexpr unsigned globalRange1d = 6;
      constexpr unsigned globalRange2d = 2;
      constexpr unsigned local = globalRange2d;
      std::vector<size_t> data(globalRange1d * globalRange2d);

      auto myQueue = util::get_cts_object::queue();
      // using this scope we ensure that the buffer will update the host values
      // after the wait_and_throw
      {
        cl::sycl::buffer<size_t, 1> buf(
            data.data(), cl::sycl::range<1>(globalRange1d * globalRange2d));

        myQueue.submit([&](cl::sycl::handler &cgh) {
          auto accessor =
              buf.template get_access<cl::sycl::access::mode::read_write>(cgh);
          cl::sycl::range<2> globalRange(globalRange1d, globalRange2d);
          cl::sycl::range<2> localRange(local, local);
          auto groupRange = globalRange / localRange;
          cgh.parallel_for_work_group<class hierarchical_private_memory>(
              groupRange, localRange, [=](cl::sycl::group<2> group_pid) {
                cl::sycl::private_memory<size_t, 2> priv(group_pid);

                group_pid.parallel_for_work_item(
                    [&](cl::sycl::h_item<2> itemID) {
                      priv(itemID) = itemID.get_global().get_linear_id();
                    });

                group_pid.parallel_for_work_item([&](
                    cl::sycl::h_item<2> itemID) {
                  accessor[itemID.get_global().get_linear_id()] = priv(itemID);
                });
              });
        });
        myQueue.wait_and_throw();
      }

      for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != i) {
          cl::sycl::string_class errorMessage =
              cl::sycl::string_class("Value for global id ") +
              std::to_string(i) + cl::sycl::string_class(" was not correct (") +
              std::to_string(data[i]) + cl::sycl::string_class(" instead of ") +
              std::to_string(i) + ")";
          FAIL(log, errorMessage);
        }
      }

    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAMESPACE
