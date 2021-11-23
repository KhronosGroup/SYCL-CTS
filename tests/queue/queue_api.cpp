/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
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
      /** check is_host() member function
       */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto isHost = queue.is_host();
        check_return_type<bool>(log, isHost, "sycl::queue::is_host()");
      }

      /** check get_context() member function
       */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto context = queue.get_context();
        check_return_type<sycl::context>(log, context,
                                             "sycl::queue::get_context()");
      }

      /** check get_device() member function
       */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto device = queue.get_device();
        check_return_type<sycl::device>(log, device,
                                            "sycl::queue::get_device()");
      }

      /** check submit(command_group_scope) member function
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto event = queue.submit([&](sycl::handler &handler) {
          handler.single_task<class queue_api_0>([=]() {});
        });
        check_return_type<sycl::event>(
            log, event, "sycl::queue::submit(command_group_scope)");
      }
      /** check submit(command_group_scope, queue) member function
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

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
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        queue.wait();
      }

      /** check wait_and_throw() member function
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        queue.wait_and_throw();
      }

      /** check throw_asynchronous() member function
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        queue.throw_asynchronous();
      }
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
