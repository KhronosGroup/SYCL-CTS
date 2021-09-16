/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_info

namespace kernel_info__ {
using namespace sycl_cts;

class kernel0;

/** tests the info for sycl::kernel
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
      auto queue = util::get_cts_object::queue();
      auto deviceList = queue.get_context().get_devices();
      auto ctx = queue.get_context();

      using k_name = kernel0;
      auto kb = sycl::get_kernel_bundle<k_name,
                                        sycl::bundle_state::executable>(ctx);
      auto kernel = kb.get_kernel(sycl::get_kernel_id<k_name>());

      queue.submit([&](sycl::handler &cgh) {
        cgh.single_task<k_name>([=]() {});
      });

      /** check types
       */
      check_type_existence<sycl::info::kernel> typeCheck;

      /** initialize return values
       */
      cl_uint clUintRet;
      std::string stringRet;
      size_t sizeTRet;
      sycl::range<3> range3Ret{0, 0, 0};
      cl_ulong clUlongRet;

      // silent warnings
      (void)clUintRet;
      (void)sizeTRet;
      (void)clUlongRet;

      /** check program info parameters
       */
      if (!queue.is_host()) {
        clUintRet =
            kernel.get_info<sycl::info::kernel::reference_count>();
      }
      stringRet = kernel.get_info<sycl::info::kernel::function_name>();
      clUintRet = kernel.get_info<sycl::info::kernel::num_args>();
      stringRet = kernel.get_info<sycl::info::kernel::attributes>();
      auto dev = util::get_cts_object::device();
      if (dev.get_info<sycl::info::device::device_type>() ==
          sycl::info::device_type::custom) {
        range3Ret = kernel.get_work_group_info<
            sycl::info::kernel_work_group::global_work_size>(dev);
      }
      range3Ret = kernel.get_work_group_info<
          sycl::info::kernel_work_group::compile_work_group_size>(dev);
      sizeTRet =
          kernel.get_work_group_info<sycl::info::kernel_work_group::
                                          preferred_work_group_size_multiple>(
              dev);
      clUlongRet = kernel.get_work_group_info<
          sycl::info::kernel_work_group::private_mem_size>(dev);
      sizeTRet = kernel.get_work_group_info<
          sycl::info::kernel_work_group::work_group_size>(dev);

      TEST_TYPE_TRAIT(kernel, reference_count, kernel);
      TEST_TYPE_TRAIT(kernel, function_name, kernel);
      TEST_TYPE_TRAIT(kernel, num_args, kernel);
      TEST_TYPE_TRAIT(kernel, attributes, kernel);

      queue.wait_and_throw();
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_info__ */
