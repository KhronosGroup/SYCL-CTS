/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_api

struct kernel_name_api {
  void operator()() const {}
};

namespace kernel_api__ {
using namespace sycl_cts;

/** test cl::sycl::kernel
 */
class TEST_NAME : public sycl_cts::util::test_base {
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
      auto ctsQueue = util::get_cts_object::queue();
      const auto isHostCtx = ctsQueue.is_host();
      auto deviceList = ctsQueue.get_context().get_devices();

      // Create kernel
      if (!is_compiler_available(deviceList)) {
        log.note(
            "online compiler is not available -- skipping test of kernel api");
      } else {
        cl::sycl::program prog(ctsQueue.get_context());
        prog.build_with_kernel_type<kernel_name_api>();
        auto k = prog.get_kernel<kernel_name_api>();
        ctsQueue.submit(
            [&](cl::sycl::handler &h) { h.single_task(kernel_name_api{}); });
        ctsQueue.wait_and_throw();

        // Check is_host()
        bool isHost = k.is_host();

        // Check get_context()
        auto cxt = k.get_context();
        check_return_type<cl::sycl::context>(log, cxt,
                                             "cl::sycl::kernel::get_context()");

        // Check get_program()
        auto prgrm = k.get_program();
        check_return_type<cl::sycl::program>(log, prgrm,
                                             "cl::sycl::kernel::get_program()");
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

} /* namespace kernel_api__ */
