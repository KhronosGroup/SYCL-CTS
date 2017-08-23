/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME context_info

namespace contect_info__ {
using namespace sycl_cts;

/** tests the info for cl::sycl::context
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
  */
  virtual void run(util::logger &log) override {
    try {
      auto context = util::get_cts_object::context();

      /** check get_info for info::context::reference_count
       */
      {
        auto ref_count =
            context.get_info<cl::sycl::info::context::reference_count>();
        check_return_type<cl_uint>(
            log, ref_count,
            "get_info<cl::sycl::info::context::reference_count>()");
        TEST_TYPE_TRAIT(context, reference_count, context);
      }

      /** check get_info for info::context::num_devices
       */
      {
        auto num_dev = context.get_info<cl::sycl::info::context::num_devices>();
        check_return_type<cl_uint>(
            log, ref_count, "get_info<cl::sycl::info::context::num_devices>()");
        TEST_TYPE_TRAIT(context, num_devices, context);
      }

      /** check get_info for info::context::devices
       */
      {
        auto devs = context.get_info<cl::sycl::info::context::devices>();
        check_return_type<cl::sycl::vector_class<cl::sycl::device>>(
            log, devs, "get_info<cl::sycl::info::context::devices>()");
        TEST_TYPE_TRAIT(context, devices, context);
      }

      /** check get_info for info::context::gl_interop
       */
      {
        auto interop = context.get_info<cl::sycl::info::context::gl_interop>();
        check_return_type<cl::sycl::info::gl_context_interop>(
            log, interop, "get_info<cl::sycl::info::context::gl_interop>()");
        TEST_TYPE_TRAIT(context, gl_interop, context);
      }

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

} /* namespace context_info__ */
