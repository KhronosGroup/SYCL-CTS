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

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"

// Disable test when compiling with ComputeCpp
// ComputeCpp doesn't fully support kernel::get_info of SYCL 2020 spec
#ifndef SYCL_CTS_COMPILING_WITH_COMPUTECPP

namespace kernel_info__ {
using namespace sycl_cts;

class kernel0;

TEST_CASE("Test kernel info", "[kernel]") {
  auto queue = util::get_cts_object::queue();
  auto dev = util::get_cts_object::device();
  auto ctx = queue.get_context();

  using k_name = kernel0;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernel = kb.get_kernel(sycl::get_kernel_id<k_name>());

  queue.submit([&](sycl::handler &cgh) { cgh.single_task<k_name>([=]() {}); });

  /** check program info parameters
   */
  auto incorrect_num_args_usage = [&kernel] {
    auto uint32Ret = kernel.get_info<sycl::info::kernel::num_args>();
  };

  {
    /** According to SYCL specification info::kernel::num_args descriptor may
        only be used to query a kernel that resides in a kernel bundle that was
        constructed using a backend specific interoperability function or to
        query a device built-in kernel. In other cases an exception with the
        errc::invalid error code is thrown. We attempt to use
        info::kernel::num_args descriptor with user defined kernel which resides
        in a kernel bundle that was constructed without using interoperability
        function. So we expect that exception will be thrown.
     */
    INFO(
        "Check that exception with error code \"errc::invalid\" is thrown in "
        "case of sycl::info::kernel::num_args descriptor usage with user "
        "defined kernel which resides in a kernel bundle that was constructed "
        "without using interoperability function.");
    CHECK_THROWS_MATCHES(incorrect_num_args_usage, sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  auto incorrect_global_work_size_usage = [&kernel, &dev] {
    auto range3Ret =
        kernel.get_info<sycl::info::kernel_device_specific::global_work_size>(
            dev);
  };
  /** According to SYCL specification
      info::kernel_device_specific::global_work_size descriptor may only be used
      if the device type is device_type::custom or if the kernel is a built-in
      kernel. In other cases an exception with the errc::invalid error code is
      thrown. We attempt to use info::kernel_device_specific::global_work_size
      descriptor with user defined kernel and expect that exception will be
      thrown if device type is not custom.
   */
  if (dev.get_info<sycl::info::device::device_type>() !=
      sycl::info::device_type::custom) {
    INFO(
        "Check that exception with error code \"errc::invalid\" is thrown in "
        "case of sycl::info::kernel_device_specific::global_work_size "
        "descriptor usage with user defined kernel and device which type "
        "is not custom.");
    CHECK_THROWS_MATCHES(incorrect_global_work_size_usage, sycl::exception,
                         sycl_cts::util::equals_exception(sycl::errc::invalid));
  }

  auto built_in_kernel_ids =
      dev.get_info<sycl::info::device::built_in_kernel_ids>();
  if (!built_in_kernel_ids.empty()) {
    auto built_in_kernel = kb.get_kernel(built_in_kernel_ids[0]);

    auto uint32Ret = built_in_kernel.get_info<sycl::info::kernel::num_args>();
    check_return_type<uint32_t>(
        uint32Ret, "sycl::kernel::get_info<sycl::info::kernel::num_args>()");

    auto range3Ret =
        built_in_kernel
            .get_info<sycl::info::kernel_device_specific::global_work_size>(
                dev);
    check_return_type<sycl::range<3>>(
        range3Ret,
        "sycl::kernel::get_info<sycl::info::kernel_device_specific::global_"
        "work_size>(dev) for built_in_kernel");
  }

  if (dev.get_info<sycl::info::device::device_type>() ==
      sycl::info::device_type::custom) {
    auto range3Ret =
        kernel.get_info<sycl::info::kernel_device_specific::global_work_size>(
            dev);
    check_return_type<sycl::range<3>>(
        range3Ret,
        "sycl::kernel::get_info<sycl::info::kernel_device_specific::global_"
        "work_size>(dev) for custom device");
  }

  auto stringRet = kernel.get_info<sycl::info::kernel::attributes>();
  check_return_type<std::string>(
      stringRet, "sycl::kernel::get_info<sycl::info::kernel::attributes>()");

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

  auto privateMemSizeRet =
      kernel.get_info<sycl::info::kernel_device_specific::private_mem_size>(
          dev);
  check_return_type<size_t>(privateMemSizeRet,
                            "sycl::kernel::get_info<sycl::info::kernel_"
                            "device_specific::private_mem_size>(dev)");

  auto workGroupSizeRet =
      kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(dev);
  check_return_type<size_t>(workGroupSizeRet,
                            "sycl::kernel::get_info<sycl::info::kernel_device_"
                            "specific::work_group_size>(dev)");

  auto maxNumSubGroupsRet =
      kernel.get_info<sycl::info::kernel_device_specific::max_num_sub_groups>(
          dev);
  check_return_type<uint32_t>(maxNumSubGroupsRet,
                              "sycl::kernel::get_info<sycl::info::kernel_"
                              "device_specific::max_num_sub_groups>(dev)");

  auto compileNumSubGroupsRet =
      kernel
          .get_info<sycl::info::kernel_device_specific::compile_num_sub_groups>(
              dev);
  check_return_type<uint32_t>(compileNumSubGroupsRet,
                              "sycl::kernel::get_info<sycl::info::kernel_"
                              "device_specific::compile_num_sub_groups>(dev)");

  auto maxSubGroupSizeRet =
      kernel.get_info<sycl::info::kernel_device_specific::max_sub_group_size>(
          dev);
  check_return_type<uint32_t>(maxSubGroupSizeRet,
                              "sycl::kernel::get_info<sycl::info::kernel_"
                              "device_specific::max_sub_group_size>(dev)");

  auto compileSubGroupSizeRet =
      kernel
          .get_info<sycl::info::kernel_device_specific::compile_sub_group_size>(
              dev);
  check_return_type<uint32_t>(compileSubGroupSizeRet,
                              "sycl::kernel::get_info<sycl::info::kernel_"
                              "device_specific::compile_sub_group_size>(dev)");

// FIXME: Reenable when struct information descriptors are implemented
#if !SYCL_CTS_COMPILING_WITH_HIPSYCL && !SYCL_CTS_COMPILING_WITH_COMPUTECPP
  check_get_info_param<sycl::info::kernel::num_args, uint32_t>(kernel);
  check_get_info_param<sycl::info::kernel::attributes, std::string>(kernel);
#endif

  queue.wait_and_throw();
};

} /* namespace kernel_info__ */

#endif // SYCL_CTS_COMPILING_WITH_COMPUTECPP
