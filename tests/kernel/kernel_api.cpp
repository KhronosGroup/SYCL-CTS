/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2017-2022 Codeplay Software LTD.
//  SPDX-FileCopyrightText: 2022-2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../common/common.h"

struct kernel_name_api {
  void operator()() const {}
};

namespace kernel_api__ {
using namespace sycl_cts;

TEST_CASE("Test kernel API", "[kernel]") {
  auto ctsQueue = util::get_cts_object::queue();
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  // Create kernel
  using k_name = kernel_name_api;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernel = kb.get_kernel(sycl::get_kernel_id<k_name>());
  ctsQueue.submit([&](sycl::handler &h) { h.single_task<k_name>(k_name{}); });
  ctsQueue.wait_and_throw();

  // Check get_context()
  auto cxt = kernel.get_context();
  check_return_type<sycl::context>(cxt, "sycl::kernel::get_context()");

  // Check get_backend()
  auto backend = kernel.get_backend();
  check_return_type<sycl::backend>(backend, "sycl::kernel::get_backend()");
};

} /* namespace kernel_api__ */
