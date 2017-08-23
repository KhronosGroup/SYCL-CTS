/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_api

namespace kernel_api__ {
using namespace sycl_cts;

/** simple OpenCL test kernel
 */
cl::sycl::string_class kernel_source = R"(
__kernel void sample(__global float * input)
{
    int x = get_global_id(0);
}
)";

/** test cl::sycl::kernel
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl {
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
      cl_program clProgram = nullptr;
      if (!create_built_program(kernel_source, clProgram, log)) {
        return;
      }

      cl_kernel clKernel = nullptr;
      if (!create_kernel(clProgram, "sample", clKernel, log)) {
        return;
      }

      cl::sycl::kernel k(clKernel);

      cl_kernel krnl = k.get();

      cl_uint ref_cnt;

      int err = clGetKernelInfo(krnl, CL_KERNEL_REFERENCE_COUNT,
                                sizeof(cl_uint), &ref_cnt, nullptr);

      CHECK_VALUE(log, err, CL_SUCCESS, 0);

      if (ref_cnt <= 0) {
        FAIL(log, "ref_cnt is incorrect. Problem with obtained cl_kernel.");
      }

      // just to make sure if returned context is correct type
      auto cxt = k.get_context();
      check_return_type<cl::sycl::context>(log, cxt,
                                           "cl::sycl::kernel::get_context()");

      auto prgrm = k.get_program();
      check_return_type<cl::sycl::program>(log, prgrm,
                                           "cl::sycl::kernel::get_program()");
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

} /* namespace kernel_api__ */
