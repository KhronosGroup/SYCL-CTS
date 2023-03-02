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

#define TEST_NAME queue_api

namespace TEST_NAMESPACE {

using namespace sycl_cts;

class queue_api_0;
class queue_api_1;
class queueNoWait;

/** test the api for sycl::queue
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
    {
      /** check get_context() member function
       */
      {
        auto queue = util::get_cts_object::queue(cts_selector);

        auto context = queue.get_context();
        check_return_type<sycl::context>(log, context,
                                         "sycl::queue::get_context()");
      }

      /** check get_device() member function
       */
      {
        auto queue = util::get_cts_object::queue(cts_selector);

        auto device = queue.get_device();
        check_return_type<sycl::device>(log, device,
                                        "sycl::queue::get_device()");
      }

      /** check submit(command_group_scope) member function
       */
      {
        auto queue = util::get_cts_object::queue(cts_selector);

        auto event = queue.submit([&](sycl::handler &handler) {
          handler.single_task<class queue_api_0>([=]() {});
        });
        check_return_type<sycl::event>(
            log, event, "sycl::queue::submit(command_group_scope)");
      }
      /** check submit(command_group_scope, queue) member function
       */
      {
        auto queue = util::get_cts_object::queue(cts_selector);

        auto secondaryQueue = util::get_cts_object::queue();
        auto event = queue.submit(
            [&](sycl::handler &handler) {
              handler.single_task<class queue_api_1>([=]() {});
            },
            secondaryQueue);
        check_return_type<sycl::event>(
            log, event, "sycl::queue::submit(command_group_scope, queue)");
        queue.wait_and_throw();
      }

      /** check wait() member function
       */
      {
        auto queue = util::get_cts_object::queue(cts_selector);

        queue.wait();
      }

      /** check wait_and_throw() member function
       */
      {
        auto queue = util::get_cts_object::queue(cts_selector);

        queue.wait_and_throw();
      }

      /** check throw_asynchronous() member function
       */
      {
        auto queue = util::get_cts_object::queue(cts_selector);

        queue.throw_asynchronous();
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
