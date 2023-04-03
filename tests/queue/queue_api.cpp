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
#include <thread>

namespace queue_api {

using namespace sycl_cts;

class queue_api_0;
class queue_api_1;
class queue_api_2;
class queue_api_3;

TEST_CASE("Check the api for sycl::queue", "[queue]") {
  {
    /** check get_context() member function
     */
    {
      auto queue = util::get_cts_object::queue(cts_selector);

      auto context = queue.get_context();
      check_return_type<sycl::context>(context, "sycl::queue::get_context()");
    }

    /** check get_device() member function
     */
    {
      auto queue = util::get_cts_object::queue(cts_selector);

      auto device = queue.get_device();
      check_return_type<sycl::device>(device, "sycl::queue::get_device()");
    }

    /** check submit(command_group_scope) member function
     */
    {
      auto queue = util::get_cts_object::queue(cts_selector);

      auto event = queue.submit([&](sycl::handler &handler) {
        handler.single_task<class queue_api_0>([=] {});
      });
      check_return_type<sycl::event>(
          event, "sycl::queue::submit(command_group_scope)");
    }

    /** check submit(command_group_scope, queue) member function
     */
    {
      auto queue = util::get_cts_object::queue(cts_selector);

      auto secondaryQueue = util::get_cts_object::queue();
      auto event = queue.submit(
          [&](sycl::handler &handler) {
            handler.single_task<class queue_api_1>([=] {});
          },
          secondaryQueue);
      check_return_type<sycl::event>(
          event, "sycl::queue::submit(command_group_scope, queue)");
      queue.wait_and_throw();
    }

    /** check that command group function object is invoked synchronously
        within same thread as call to sycl::queue::submit(cgf) or
        sycl::queue::submit(cgf, secondaryQueue)
     */
    {
      std::thread::id cgf_thread_id;

      auto queue = util::get_cts_object::queue(cts_selector);

      queue
          .submit([&cgf_thread_id](sycl::handler &handler) {
            cgf_thread_id = std::this_thread::get_id();
            handler.single_task<class queue_api_2>([=] {});
          })
          .wait();

      CHECK(cgf_thread_id == std::this_thread::get_id());
    }

    {
      std::thread::id cgf_thread_id;

      auto queue = util::get_cts_object::queue(cts_selector);
      auto secondaryQueue = util::get_cts_object::queue();

      queue
          .submit(
              [&cgf_thread_id](sycl::handler &handler) {
                cgf_thread_id = std::this_thread::get_id();
                handler.single_task<class queue_api_3>([=] {});
              },
              secondaryQueue)
          .wait();

      CHECK(cgf_thread_id == std::this_thread::get_id());
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

} /* namespace queue_api */
