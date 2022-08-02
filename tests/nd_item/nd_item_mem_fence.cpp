/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright (c) 2018-2022 Codeplay Software LTD. All Rights Reserved.
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

#define TEST_NAME nd_item_mem_fence

namespace nd_item_mem_fence__ {
using namespace sycl_cts;

class mem_fence_kernel;
void test_mem_fence(util::logger &log, cl::sycl::queue &queue) {
  /* set workspace size */
  constexpr size_t globalSize = 64;
  size_t localSize = 2;

  /* adjust work-group size */
  {
    /* check work-group size device limit */
    auto device = queue.get_device();
    auto maxDeviceWorkGroupSize =
        device.template get_info<cl::sycl::info::device::max_work_group_size>();
    localSize = (maxDeviceWorkGroupSize < localSize) ? maxDeviceWorkGroupSize
                                                     : localSize;

    /* Check work-group size kernel limit - it must be >= 2 to test
     * nd_item::mem_fence member function. To query
     * info::kernel_work_group::work_group_size property, we need obtain test
     * kernel handler, which requires online compilation
     * */
    auto devices = queue.get_context().get_devices();
    if (!is_compiler_available(devices) || !is_linker_available(devices))
      return;

    cl::sycl::program P(queue.get_context());
    P.build_with_kernel_type<mem_fence_kernel>("");
    auto kernel = P.get_kernel<mem_fence_kernel>();
    auto maxKernelWorkGroupSize = kernel.template get_work_group_info<
        cl::sycl::info::kernel_work_group::work_group_size>(device);
    localSize = (maxKernelWorkGroupSize < localSize) ? maxKernelWorkGroupSize
                                                     : localSize;
  }

  /* allocate and assign host data */

  /* init ranges*/
  cl::sycl::range<1> globalRange(globalSize);
  cl::sycl::range<1> localRange(localSize);
  cl::sycl::nd_range<1> NDRange(globalRange, localRange);

  /* run kernel to check mem_fence interface is available*/
  {
    queue.submit([&](cl::sycl::handler &cgh) {

      cgh.parallel_for<class mem_fence_kernel>(
          NDRange, [=](cl::sycl::nd_item<1> item) {

            item.mem_fence(cl::sycl::access::fence_space::local_space);
            item.mem_fence(cl::sycl::access::fence_space::global_space);
            item.mem_fence(cl::sycl::access::fence_space::global_and_local);
            item.mem_fence();

          });
    });
  }
}

/** test cl::sycl::nd_item mem_fence
*/
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
  *  @param info, test_base::info structure as output
  */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
  *  @param log, test transcript logging class
  */
  void run(util::logger &log) override {
    try {
      auto cmdQueue = util::get_cts_object::queue();

      test_mem_fence(log, cmdQueue);

      cmdQueue.wait_and_throw();
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

util::test_proxy<TEST_NAME> proxy;

} /* namespace nd_item_mem_fence__ */
