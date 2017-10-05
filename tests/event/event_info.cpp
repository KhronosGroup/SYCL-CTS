/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_api

namespace TEST_NAMESPACE {

using namespace sycl_cts;

/** test the api for cl::sycl::event
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

      /** check cl::sycl::info::event_command_status
      */
      check_enum_class_value(cl::sycl::info::event_command_status::complete);
      check_enum_class_value(cl::sycl::info::event_command_status::running);
      check_enum_class_value(cl::sycl::info::event_command_status::submitted);

      /** check cl::sycl::info::event
      */
      check_enum_class_value(cl::sycl::info::event::command_execution_status);
      check_enum_class_value(cl::sycl::info::event::reference_count);

      /** check cl::sycl::info::event_profiling
      */
      check_enum_class_value(cl::sycl::info::event_profiling::command_submit);
      check_enum_class_value(cl::sycl::info::event_profiling::command_start);
      check_enum_class_value(cl::sycl::info::event_profiling::command_end);

      /** check get_info & get_profiling_info parameters
      */
      {
        auto queue = util::get_cts_object::queue();
        cl::sycl::event event = queue.submit([&](cl::sycl::handler &handler) {
          handler.single_task(dummy_functor<class TEST_NAME>());
        });

        check_get_info_param<cl::sycl::info::event,
          cl::sycl::info::event_command_status,
          cl::sycl::info::event::command_execution_status>(log, event);
        check_get_info_param<cl::sycl::info::event, cl::sycl::cl_uint,
          cl::sycl::info::event::reference_count>(log, event);
        check_get_profiling_info_param<cl::sycl::info::event_profiling,
          cl::sycl::cl_ulong, cl::sycl::info::event_profiling::command_submit>(
            log, event);
        check_get_profiling_info_param<cl::sycl::info::event_profiling,
          cl::sycl::cl_ulong, cl::sycl::info::event_profiling::command_start>(
          log, event);
        check_get_profiling_info_param<cl::sycl::info::event_profiling,
          cl::sycl::cl_ulong, cl::sycl::info::event_profiling::command_end>(
          log, event);
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

} /* namespace TEST_NAMESPACE */
