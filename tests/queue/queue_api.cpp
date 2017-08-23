/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME queue_api

namespace queue_api__ {
using namespace sycl_cts;

/** tests the api for cl::sycl::queue
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  virtual void run(util::logger &log) override {
    try {
      cts_selector selector;
      auto queue = util::get_cts_object::queue(selector);

      /** check is_host() method
       */
      {
        auto isHost = queue.is_host();
        check_return_type<bool>(log, isHost, "cl::sycl::queue::is_host()");
      }

      /** check get_context() method
       */
      {
        auto context = queue.get_context();
        check_return_type<cl::sycl::context>(log, context,
                                             "cl::sycl::queue::get_context()");
      }

      /** check get_device() method
       */
      {
        auto device = queue.get_device();
        check_return_type<cl::sycl::device>(log, device,
                                            "cl::sycl::queue::get_device()");
      }

      /** check get method on non-host devices
       */
      if (!selector.is_host()) {
        auto clQueueObject = queue.get();
        check_return_type<cl_command_queue>(log, clQueueObject,
                                            "cl::sycl::queue::get()");
      }

      /** check submit(command_group_scope) method
      */
      {
        auto event = queue.submit([&](cl::sycl::handler &handler) {
          handler.single_task<class queue_api_0>([=]() {});
        });
        check_return_type<cl::sycl::event>(
            log, event, "cl::sycl::queue::submit(command_group_scope)");
      }
      /** check submit(command_group_scope, queue) method
      */
      auto secondaryQueue = util::get_cts_object::queue();
      auto event = queue.submit(
          [&](cl::sycl::handler &handler) {
            handler.single_task<class queue_api_1>([=]() {});
          },
          secondaryQueue);
      check_return_type<cl::sycl::event>(
          log, event, "cl::sycl::queue::submit(command_group_scope, queue)");

      /** check wait() method
      */
      queue.wait();

      /** check wait_and_throw() method
      */
      queue.wait_and_throw();

      /** check throw_asynchronous() method
      */
      queue.throw_asynchronous();

      /* Kernel does nothing, but this checks the case where nothing
       * waits on the queue finishing */
      queue.submit([&](cl::sycl::handler &cgh) {
        cgh.single_task<class queueNoWait>([=] { int i = 0; });
      });
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace queue_api__ */
