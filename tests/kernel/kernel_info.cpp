/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2017-2022 Codeplay Software LTD. All Rights Reserved.
//  Copyright (c) 2022 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "../common/common.h"

// Disable test when compiling with ComputeCpp
// ComputeCpp doesn't fully support kernel::get_info of SYCL 2020 spec
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP

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
    {
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
      clUintRet = kernel.get_info<sycl::info::kernel::num_args>();
      stringRet = kernel.get_info<sycl::info::kernel::attributes>();
      auto dev = util::get_cts_object::device();
      if (dev.get_info<sycl::info::device::device_type>() ==
          sycl::info::device_type::custom) {
        range3Ret =
            kernel
                .get_info<sycl::info::kernel_device_specific::global_work_size>(
                    dev);
      }
      range3Ret = kernel.get_info<
          sycl::info::kernel_device_specific::compile_work_group_size>(dev);
      sizeTRet = kernel.get_info<sycl::info::kernel_device_specific::
                                     preferred_work_group_size_multiple>(dev);
      clUlongRet =
          kernel.get_info<sycl::info::kernel_device_specific::private_mem_size>(
              dev);
      sizeTRet =
          kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(
              dev);

      check_get_info_param<sycl::info::kernel::num_args, uint32_t>(log, kernel);
      check_get_info_param<sycl::info::kernel::attributes, std::string>(log,
                                                                        kernel);

      queue.wait_and_throw();
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_info__ */
#endif // SYCL_CTS_COMPILING_WITH_COMPUTECPP
