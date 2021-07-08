/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME event_info

namespace TEST_NAMESPACE {

using namespace sycl_cts;
namespace sycl = sycl;

/** test the get_info and get_profiling_info apis for sycl::event
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
      /** check sycl::info::event_command_status
       */
      check_enum_class_value(sycl::info::event_command_status::complete);
      check_enum_class_value(sycl::info::event_command_status::running);
      check_enum_class_value(sycl::info::event_command_status::submitted);

      /** check sycl::info::event
       */
      check_enum_class_value(sycl::info::event::command_execution_status);

      /** check sycl::info::event_profiling
       */
      check_enum_class_value(sycl::info::event_profiling::command_submit);
      check_enum_class_value(sycl::info::event_profiling::command_start);
      check_enum_class_value(sycl::info::event_profiling::command_end);

      /** check get_info
       */
      {
        auto queue = util::get_cts_object::queue();
        auto e1 = queue.submit(
            [&](sycl::handler &cgh) { cgh.single_task(dummy_functor()); });
        auto e2 = queue.submit([&](sycl::handler &cgh) {
          cgh.depends_on(e1);
          cgh.single_task(dummy_functor());
        });
        // Check that query returns the expected type.
        check_get_info_param<sycl::info::event,
                             sycl::info::event_command_status,
                             sycl::info::event::command_execution_status>(log,
                                                                          e2);
        e2.wait();
        // Since e2's CGF depends on e1, the latter must have completed by now.
        const auto status =
            e1.get_info<sycl::info::event::command_execution_status>();
        if (status != sycl::info::event_command_status::complete) {
          FAIL(log, "event returned unexpected command execution status");
        }
      }

      /** check get_profiling_info
       */
      {
        const sycl::device_selector &selector = cts_selector();
        static cts_async_handler asyncHandler;
        sycl::queue queue =
            sycl::queue(selector, asyncHandler,
                        {sycl::property::queue::enable_profiling()});

        sycl::event event = queue.submit([&](sycl::handler &handler) {
          handler.single_task(dummy_functor());
        });

        event.wait();

        // Check that queries return the expected type.
        check_get_profiling_info_param<
            sycl::info::event_profiling, uint64_t,
            sycl::info::event_profiling::command_submit>(log, event);
        check_get_profiling_info_param<
            sycl::info::event_profiling, uint64_t,
            sycl::info::event_profiling::command_start>(log, event);
        check_get_profiling_info_param<
            sycl::info::event_profiling, uint64_t,
            sycl::info::event_profiling::command_end>(log, event);

        const uint64_t submit_time = event.get_profiling_info<
            sycl::info::event_profiling::command_submit>();
        const uint64_t start_time = event.get_profiling_info<
            sycl::info::event_profiling::command_start>();
        const uint64_t end_time =
            event
                .get_profiling_info<sycl::info::event_profiling::command_end>();

        // While the returned values are implementation defined, we can still
        // perform some basic sanity checks.
        if (!(submit_time <= start_time)) {
          FAIL(log, "command submit time > start time");
        }
        if (!(start_time <= end_time)) {
          FAIL(log, "command start time > end time");
        }
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      sycl::string_class errorMsg =
          "a SYCL exception was caught: " + sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace TEST_NAMESPACE */
