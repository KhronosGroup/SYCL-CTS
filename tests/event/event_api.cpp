/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_api

namespace TEST_NAMESPACE {

using namespace sycl_cts;

class event_api_kernel_0;
class event_api_kernel_1;
class event_api_kernel_2;
class event_api_kernel_3;
class event_api_kernel_4;
class event_api_kernel_5;
class event_api_kernel_6;

/** test the api for sycl::event
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
#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
    /** check get()
     */
    {
      auto queue = util::get_cts_object::queue();
      auto event = get_queue_event<class event_api_kernel_0>(queue);

      if (!queue.is_host()) {
        auto evt = event.get();
        check_return_type<cl_event>(log, evt, "sycl::event::get()");
      }
      queue.wait_and_throw();
    }
#endif

    /** check is_host()
     */
    {
      auto queue = util::get_cts_object::queue();
      auto event = get_queue_event<class event_api_kernel_1>(queue);

      auto isHost = event.is_host();
      check_return_type<bool>(log, isHost, "sycl::event::is_host()");
    }

    /** check get_wait_list()
     */
    {
      auto queue = util::get_cts_object::queue();
      auto event = get_queue_event<class event_api_kernel_2>(queue);

      auto events = event.get_wait_list();
      check_return_type<std::vector<sycl::event>>(
          log, events, "sycl::event::get_wait_list()");
    }

    /** check wait()
     */
    {
      auto queue = util::get_cts_object::queue();
      auto event = get_queue_event<class event_api_kernel_3>(queue);

      event.wait();
    }

    /** check wait_and_throw()
     */
    {
      auto queue = util::get_cts_object::queue();
      auto event = get_queue_event<class event_api_kernel_4>(queue);

      event.wait_and_throw();
    }

    /** check static wait()
     */
    {
      auto queue = util::get_cts_object::queue();
      auto event = get_queue_event<class event_api_kernel_5>(queue);
      std::vector<sycl::event> eventList;
      eventList.push_back(event);

      sycl::event::wait(eventList);
    }

    /** check static wait_and_throw()
     */
    {
      auto queue = util::get_cts_object::queue();
      auto event = get_queue_event<class event_api_kernel_6>(queue);
      std::vector<sycl::event> eventList;
      eventList.push_back(event);

      sycl::event::wait_and_throw(eventList);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
