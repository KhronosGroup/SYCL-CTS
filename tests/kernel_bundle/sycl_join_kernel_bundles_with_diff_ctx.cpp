/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2023 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
*******************************************************************************/

#include "../../util/sycl_exceptions.h"
#include "../common/common.h"
#include "kernel_bundle.h"

class simple_kernel1;
class simple_kernel2;

template <sycl::bundle_state bundle_state>
void check_for_state(const sycl::context& first_ctx,
                     const sycl::context& second_ctx) {
  auto first_kb =
      sycl::get_kernel_bundle<simple_kernel1, bundle_state>(first_ctx);
  auto second_kb =
      sycl::get_kernel_bundle<simple_kernel2, bundle_state>(second_ctx);

  std::vector<sycl::kernel_bundle<bundle_state>> kb_with_diff_ctx{first_kb,
                                                                  second_kb};

  auto action = [&] { sycl::join(kb_with_diff_ctx); };

  INFO("Bundle state: "
       << sycl_cts::get_cts_string::for_bundle_state<bundle_state>()
       << "check sycl::join exception errc::invalid thrown with different "
          "context");
  CHECK_THROWS_MATCHES(action(), sycl::exception,
                       sycl_cts::util::equals_exception(sycl::errc::invalid));
}

TEST_CASE("sycl::join kernel bundles with different contexts", "[sycl::join]") {
  using namespace kernels;
  using namespace sycl_cts::tests::kernel_bundle;

  const std::vector<sycl::device> devices{sycl::device::get_devices()};
  if (devices.size() < 2) {
    SKIP("Requires at least two devices");
  }

  sycl::context first_ctx(devices[0]);
  sycl::context second_ctx(devices[1]);
  sycl::queue q1{devices[0]};
  sycl::queue q2{devices[1]};

  sycl_cts::tests::kernel_bundle::define_kernel<simple_kernel1>(q1);
  sycl_cts::tests::kernel_bundle::define_kernel<simple_kernel2>(q2);

  check_for_state<sycl::bundle_state::input>(first_ctx, second_ctx);
  check_for_state<sycl::bundle_state::object>(first_ctx, second_ctx);
  check_for_state<sycl::bundle_state::executable>(first_ctx, second_ctx);
}
