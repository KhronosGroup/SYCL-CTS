/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
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

/** test sycl::kernel
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
        sycl::program prog(ctsQueue.get_context());
        prog.build_with_kernel_type<kernel_name_api>();
        auto k = prog.get_kernel<kernel_name_api>();
        ctsQueue.submit(
            [&](sycl::handler &h) { h.single_task(kernel_name_api{}); });
        ctsQueue.wait_and_throw();

        // Check is_host()
        bool isHost = k.is_host();

        // Check get_context()
        auto cxt = k.get_context();
        check_return_type<sycl::context>(log, cxt,
                                             "sycl::kernel::get_context()");

        // Check get_program()
        auto prgrm = k.get_program();
        check_return_type<sycl::program>(log, prgrm,
                                             "sycl::kernel::get_program()");
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_api__ */
