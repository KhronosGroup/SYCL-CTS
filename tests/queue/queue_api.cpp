/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_api

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test the api for cl::sycl::queue
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
    try {

      /** check is_host() member function
       */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto isHost = queue.is_host();
        check_return_type<bool>(log, isHost, "cl::sycl::queue::is_host()");
      }

      /** check get_context() member function
       */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto context = queue.get_context();
        check_return_type<cl::sycl::context>(log, context,
                                             "cl::sycl::queue::get_context()");
      }

      /** check get_device() member function
       */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto device = queue.get_device();
        check_return_type<cl::sycl::device>(log, device,
                                            "cl::sycl::queue::get_device()");
      }

      /** check get() member function
       */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);
        if (!selector.is_host()) {
          auto clQueueObject = queue.get();
          check_return_type<cl_command_queue>(log, clQueueObject,
                                              "cl::sycl::queue::get()");
        }
      }

      /** check submit(command_group_scope) member function
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto event = queue.submit([&](cl::sycl::handler &handler) {
          handler.single_task<class queue_api_0>([=]() {});
        });
        check_return_type<cl::sycl::event>(
            log, event, "cl::sycl::queue::submit(command_group_scope)");
      }
      /** check submit(command_group_scope, queue) member function
      */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        auto secondaryQueue = util::get_cts_object::queue();
        auto event = queue.submit(
            [&](cl::sycl::handler &handler) {
              handler.single_task<class queue_api_1>([=]() {});
            },
            secondaryQueue);
        check_return_type<cl::sycl::event>(
            log, event, "cl::sycl::queue::submit(command_group_scope, queue)");
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

      /* kernel does nothing, but this checks the case where nothing
       * waits on the queue finishing */
      {
        cts_selector selector;
        auto queue = util::get_cts_object::queue(selector);

        queue.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task<class queueNoWait>([=] { int i = 0; });
        });
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
