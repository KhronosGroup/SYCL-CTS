/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_api

namespace event_api__ {
using namespace sycl_cts;

/**
 * @brief Helper function to check an event profiling info parameter.
 */
template <typename returnT, cl::sycl::info::event kValue>
void check_event_get_profiling_info_param(sycl_cts::util::logger &log,
                                          const cl::sycl::event &object) {
  /** check param_traits return type
  */
  using paramTraitsType =
      typename cl::sycl::info::param_traits<cl::sycl::info::event_profiling,
                                            kValue>::return_type;
  check_return_type<returnT>(log, paramTraitsType,
                             "cl::sycl::info::param_traits<cl::sycl::info::"
                             "event_profiling, kValue>::return_type");

  /** check get_profiling_info return type
  */
  auto returnValue = object.template get_profiling_info<kValue>();
  check_return_type<returnT>(log, returnValue, "event::get_profiling_info()");
}

/**
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
      auto queue = util::get_cts_object::queue();

      cl::sycl::event event = queue.submit([&](cl::sycl::handler &handler) {
        handler.single_task<class kernel0>([=]() {});
      });

      cl::sycl::vector_class<cl::sycl::event> eventList;
      eventList.push_back(event);

      if (!queue.is_host()) {
        auto evt = event.get();
        check_return_type<cl_event>(log, evt, "get()");
      }

      /** check get_wait_list()
      */
      auto events = event.get_wait_list();
      check_return_type<cl::sycl::vector_class<cl::sycl::event>>(
          log, events, "get_wait_list()");

      /** check wait()
      */
      event.wait();

      /** check wait_and_throw()
      */
      event.wait_and_throw();

      /** check static wait()
      */
      cl::sycl::event::wait(eventList);

      /** check static wait_and_throw()
      */
      cl::sycl::event::wait_and_throw(eventList);

      check_get_info_param<cl::sycl::info::event,
                           cl::sycl::info::event_command_status,
                           cl::sycl::info::event::command_execution_status>(
          log, event);
      check_get_info_param<cl::sycl::info::event, cl::sycl::cl_uint,
                           cl::sycl::info::event::reference_count>(log, event);

      check_event_get_profiling_info_param<
          ::cl_ulong, cl::sycl::info::event_profiling::command_submit>(log,
                                                                       event);
      check_event_get_profiling_info_param<
          ::cl_ulong, cl::sycl::info::event_profiling::command_start>(log,
                                                                      event);
      check_event_get_profiling_info_param<
          ::cl_ulong, cl::sycl::info::event_profiling::command_end>(log, event);
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace event_api__ */
