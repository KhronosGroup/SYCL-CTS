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

namespace kernel_info__ {
using namespace sycl_cts;

class kernel0;

TEST_CASE("kernel info test", "[kernel]") {
  auto queue = util::get_cts_object::queue();
  auto deviceList = queue.get_context().get_devices();
  auto ctx = queue.get_context();

  using k_name = kernel0;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernel = kb.get_kernel(sycl::get_kernel_id<k_name>());

  queue.submit([&](sycl::handler &cgh) { cgh.single_task<k_name>([=]() {}); });

  /** check program info parameters
   */
  auto clUintRet = kernel.get_info<sycl::info::kernel::num_args>();
  check_return_type<cl_uint>(
      clUintRet, "sycl::kernel::get_info<sycl::info::kernel::num_args>()");

  auto stringRet = kernel.get_info<sycl::info::kernel::attributes>();
  check_return_type<std::string>(
      stringRet, "sycl::kernel::get_info<sycl::info::kernel::attributes>()");

  auto dev = util::get_cts_object::device();
  if (dev.get_info<sycl::info::device::device_type>() ==
      sycl::info::device_type::custom) {
    auto range3Ret =
        kernel.get_info<sycl::info::kernel_device_specific::global_work_size>(
            dev);
    check_return_type<sycl::range<3>>(
        range3Ret,
        "sycl::kernel::get_info<sycl::info::kernel_device_specific::global_"
        "work_size>(dev)");
  }

  auto range3Ret = kernel.get_info<
      sycl::info::kernel_device_specific::compile_work_group_size>(dev);
  check_return_type<sycl::range<3>>(
      range3Ret,
      "sycl::kernel::get_info<sycl::info::kernel_device_specific::compile_work_"
      "group_size>(dev)");

  auto sizeTRet = kernel.get_info<
      sycl::info::kernel_device_specific::preferred_work_group_size_multiple>(
      dev);
  check_return_type<size_t>(
      sizeTRet,
      "sycl::kernel::get_info<sycl::info::kernel_device_specific::preferred_"
      "work_group_size_multiple>(dev)");

  auto clUlongRet =
      kernel.get_info<sycl::info::kernel_device_specific::private_mem_size>(
          dev);
  check_return_type<cl_ulong>(clUlongRet,
                              "sycl::kernel::get_info<sycl::info::kernel_"
                              "device_specific::private_mem_size>(dev)");

  auto anotherSizeTRet =
      kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(dev);
  check_return_type<size_t>(anotherSizeTRet,
                            "sycl::kernel::get_info<sycl::info::kernel_device_"
                            "specific::work_group_size>(dev)");

// FIXME: Reenable when struct information descriptors are implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP
  check_get_info_param<sycl::info::kernel::num_args, uint32_t>(kernel);
  check_get_info_param<sycl::info::kernel::attributes, std::string>(kernel);
#endif

  queue.wait_and_throw();
};

} /* namespace kernel_info__ */
#endif // SYCL_CTS_COMPILING_WITH_COMPUTECPP
