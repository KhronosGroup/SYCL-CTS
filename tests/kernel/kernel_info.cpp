/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_info

namespace kernel_info__ {
using namespace sycl_cts;

class kernel0;

/** tests the info for cl::sycl::kernel
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl {
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
      auto queue = util::get_cts_object::queue();

      cl::sycl::program program(queue.get_context());
      program.build_with_kernel_type<kernel0>();
      cl::sycl::kernel kernel = program.get_kernel<kernel0>();
      queue.submit(
          [&](cl::sycl::handler &cgh) { cgh.single_task<kernel0>([=]() {}); });

      /** check types
      */
      using kernelInfo = cl::sycl::info::kernel;

      /** initialize return values
      */
      cl_uint clUintRet;
      cl::sycl::string_class stringRet;
      size_t sizeTRet;
      cl::sycl::range<3> range3Ret;
      cl_ulong clUlongRet;

      /** check program info parameters
      */
      if (!queue.is_host()) {
        clUintRet = kernel.get_info<cl::sycl::info::kernel::reference_count>();
      }
      stringRet = kernel.get_info<cl::sycl::info::kernel::function_name>();
      clUintRet = kernel.get_info<cl::sycl::info::kernel::num_args>();
      stringRet = kernel.get_info<cl::sycl::info::kernel::attributes>();
      auto dev = util::get_cts_object::device();
      if (dev.get_info<cl::sycl::info::device::device_type>() ==
          cl::sycl::info::device_type::custom) {
        range3Ret = kernel.get_work_group_info<
            cl::sycl::info::kernel_work_group::global_work_size>(dev);
      }
      range3Ret = kernel.get_work_group_info<
          cl::sycl::info::kernel_work_group::compile_work_group_size>(dev);
      sizeTRet =
          kernel.get_work_group_info<cl::sycl::info::kernel_work_group::
                                         preferred_work_group_size_multiple>(
              dev);
      clUlongRet = kernel.get_work_group_info<
          cl::sycl::info::kernel_work_group::private_mem_size>(dev);
      sizeTRet = kernel.get_work_group_info<
          cl::sycl::info::kernel_work_group::work_group_size>(dev);

      TEST_TYPE_TRAIT(kernel, reference_count, kernel);
      TEST_TYPE_TRAIT(kernel, function_name, kernel);
      TEST_TYPE_TRAIT(kernel, num_args, kernel);
      TEST_TYPE_TRAIT(kernel, attributes, kernel);

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

} /* namespace kernel_info__ */
