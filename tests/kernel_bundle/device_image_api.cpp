/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
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
#include "../common/get_cts_object.h"

#include <type_traits>

struct kernel_name;

TEST_CASE("device image api", "[device_image]") {
  // check that device_image is not user-constructible
  {
    CHECK_FALSE(std::is_default_constructible_v<
                sycl::device_image<sycl::bundle_state::input>>);
    CHECK_FALSE(std::is_default_constructible_v<
                sycl::device_image<sycl::bundle_state::object>>);
    CHECK_FALSE(std::is_default_constructible_v<
                sycl::device_image<sycl::bundle_state::executable>>);
  }

  // check that has_kernel is noexcept
  {
    sycl::device device = sycl_cts::util::get_cts_object::device();
    sycl::queue queue = sycl_cts::util::get_cts_object::queue();
    sycl::context context = queue.get_context();
    bool selector_was_called = false;  // check that the test gets executed
    bool has_kernel_noexcept = false;
    bool has_kernel_dev_noexcept = false;
    sycl::kernel_id kernel_id = sycl::get_kernel_id<kernel_name>();
    constexpr sycl::bundle_state bundle_state = sycl::bundle_state::executable;
    // force the online compilation of all kernels in the context,
    // calls the selector on all device images
    const auto bundle = sycl::get_kernel_bundle<bundle_state>(
        context, [&](const sycl::device_image<bundle_state>& device_image) {
          selector_was_called = true;
          has_kernel_noexcept = noexcept(device_image.has_kernel(kernel_id));
          has_kernel_dev_noexcept =
              noexcept(device_image.has_kernel(kernel_id, device));
          return true;
        });
    // minimal kernel with side effect as some implementations may not support
    // empty kernels
    sycl::buffer<int, 1> buffer{sycl::range<1>{1}};
    queue
        .submit([&](sycl::handler& cgh) {
          sycl::accessor acc{buffer, cgh, sycl::write_only};
          cgh.single_task<kernel_name>([=]() { acc[0] = 1; });
        })
        .wait_and_throw();
    CHECK((selector_was_called && has_kernel_noexcept &&
           has_kernel_dev_noexcept));
  }
}
