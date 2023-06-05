/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
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
#include "kernel_bundle.h"
#include "kernels.h"

template <sycl::bundle_state bundle_state>
void check_for_state(const sycl::context& first_ctx,
                     const sycl::context& second_ctx) {
  auto first_kb =
      sycl::get_kernel_bundle<bundle_state>(first_ctx, first_ctx.get_devices());
  auto second_kb = sycl::get_kernel_bundle<bundle_state>(
      second_ctx, second_ctx.get_devices());

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
  {
    INFO("Requires at least two devices");
    REQUIRE(devices.size() > 1);
  }

  sycl::context first_ctx(devices[0]);
  sycl::context second_ctx(devices[1]);
  sycl::queue q{devices[0]};

  check_for_state<sycl::bundle_state::input>(first_ctx, second_ctx);
  check_for_state<sycl::bundle_state::object>(first_ctx, second_ctx);
  check_for_state<sycl::bundle_state::executable>(first_ctx, second_ctx);

  define_kernel<simple_kernel_descriptor, sycl::bundle_state::executable>(q);
  define_kernel<simple_kernel_descriptor_second,
                sycl::bundle_state::executable>(q);
}
