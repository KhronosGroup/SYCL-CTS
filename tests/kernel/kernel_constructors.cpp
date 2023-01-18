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

template <int a>
class test_kernel {
 public:
  void operator()() const {}
};

namespace kernel_constructors__ {
using namespace sycl_cts;

TEST_CASE("Test copy constructor", "[kernel]") {
  cts_selector ctsSelector;
  auto ctsQueue = util::get_cts_object::queue(ctsSelector);
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  using k_name = test_kernel<0>;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());

  sycl::kernel kernelB(kernelA);

  ctsQueue.submit([&](sycl::handler &cgh) { cgh.single_task(k_name()); });

#ifdef SYCL_BACKEND_OPENCL
  if (ctsQueue.get_backend() == sycl::backend::opencl) {
    auto iopKernelA = sycl::get_native<sycl::backend::opencl>(kernelA);
    auto iopKernelB = sycl::get_native<sycl::backend::opencl>(kernelB);
    if (!ctsSelector.is_host() && (iopKernelA != iopKernelB)) {
      FAIL(
          "kernel was not constructed correctly. (contains different "
          "OpenCL kernel object)");
    }
  }
#endif

  ctsQueue.wait_and_throw();
}

TEST_CASE("Test assignment operator", "[kernel]") {
  cts_selector ctsSelector;
  auto ctsQueue = util::get_cts_object::queue(ctsSelector);
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  using k_name = test_kernel<1>;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());

  ctsQueue.submit([&](sycl::handler &cgh) { cgh.single_task(k_name()); });

  sycl::kernel kernelB = kernelA;

#ifdef SYCL_BACKEND_OPENCL
  if (ctsQueue.get_backend() == sycl::backend::opencl) {
    auto iopKernelA = sycl::get_native<sycl::backend::opencl>(kernelA);
    auto iopKernelB = sycl::get_native<sycl::backend::opencl>(kernelB);
    if (!ctsSelector.is_host() && (iopKernelA != iopKernelB)) {
      FAIL(
          "kernel was not constructed correctly. (contains different "
          "OpenCL kernel object)");
    }
  }
#endif

  ctsQueue.wait_and_throw();
}

TEST_CASE("Test move constructor", "[kernel]") {
  cts_selector ctsSelector;
  auto ctsQueue = util::get_cts_object::queue(ctsSelector);
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  using k_name = test_kernel<2>;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());

  ctsQueue.submit([&](sycl::handler &cgh) { cgh.single_task(k_name()); });

  sycl::kernel kernelB(std::move(kernelA));

  ctsQueue.wait_and_throw();
}

TEST_CASE("Test move assignment operator", "[kernel]") {
  cts_selector ctsSelector;
  auto ctsQueue = util::get_cts_object::queue(ctsSelector);
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  using k_name = test_kernel<3>;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());
  ctsQueue.submit([&](sycl::handler &cgh) { cgh.single_task(k_name()); });

  sycl::kernel kernelB = std::move(kernelA);

  ctsQueue.wait_and_throw();
}

TEST_CASE("Test equality operator", "[kernel]") {
  cts_selector ctsSelector;
  auto ctsQueue = util::get_cts_object::queue(ctsSelector);
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  using k_name4 = test_kernel<4>;
  auto kbA =
      sycl::get_kernel_bundle<k_name4, sycl::bundle_state::executable>(ctx);
  auto kernelA = kbA.get_kernel(sycl::get_kernel_id<k_name4>());
  ctsQueue.submit([&](sycl::handler &cgh) { cgh.single_task(k_name4()); });
  sycl::kernel kernelB(kernelA);

  using k_name5 = test_kernel<5>;
  auto kbC =
      sycl::get_kernel_bundle<k_name5, sycl::bundle_state::executable>(ctx);
  auto kernelC = kbC.get_kernel(sycl::get_kernel_id<k_name5>());
  ctsQueue.submit([&](sycl::handler &cgh) { cgh.single_task(k_name5()); });
  kernelC = (kernelA);

  using k_name6 = test_kernel<6>;
  auto kbD =
      sycl::get_kernel_bundle<k_name6, sycl::bundle_state::executable>(ctx);
  auto kernelD = kbD.get_kernel(sycl::get_kernel_id<k_name6>());

  ctsQueue.submit([&](sycl::handler &cgh) { cgh.single_task(k_name6()); });

  if (!ctsSelector.is_host()) {
#ifdef SYCL_BACKEND_OPENCL
    if (ctsQueue.get_backend() == sycl::backend::opencl) {
      auto iopKernelA = sycl::get_native<sycl::backend::opencl>(kernelA);
      auto iopKernelB = sycl::get_native<sycl::backend::opencl>(kernelB);
      auto iopKernelC = sycl::get_native<sycl::backend::opencl>(kernelC);
      auto iopCtxA =
          sycl::get_native<sycl::backend::opencl>(kernelA.get_context());
      auto iopCtxB =
          sycl::get_native<sycl::backend::opencl>(kernelB.get_context());
      auto iopCtxC =
          sycl::get_native<sycl::backend::opencl>(kernelC.get_context());
      auto iopProgA =
          sycl::get_native<sycl::backend::opencl>(kernelA.get_kernel_bundle());
      auto iopProgB =
          sycl::get_native<sycl::backend::opencl>(kernelB.get_kernel_bundle());
      auto iopProgC =
          sycl::get_native<sycl::backend::opencl>(kernelC.get_kernel_bundle());

      if (kernelA == kernelB && (iopKernelA != iopKernelB ||
                                 iopCtxA != iopCtxB || iopProgA != iopProgB)) {
        FAIL(
            "kernel equality does not work correctly (copy "
            "constructed)");
      }
      if (kernelA == kernelC && (iopKernelA != iopKernelC ||
                                 iopCtxA != iopCtxC || iopProgA != iopProgC)) {
        FAIL("kernel equality does not work correctly (copy assigned)");
      }
    }
#endif
    if (kernelA != kernelB) {
      FAIL(
          "kernel non-equality does not work correctly"
          "(copy constructed)");
    }
    if (kernelA != kernelC) {
      FAIL(
          "kernel non-equality does not work correctly"
          "(copy assigned)");
    }
    if (kernelC == kernelD) {
      FAIL(
          "kernel equality does not work correctly"
          "(comparing same)");
    }
    if (!(kernelC != kernelD)) {
      FAIL(
          "kernel non-equality does not work correctly"
          "(comparing same)");
    }
  }

  ctsQueue.wait_and_throw();
}

TEST_CASE("Test hashing", "[kernel]") {
  auto ctsQueue = util::get_cts_object::queue();
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  using k_name = test_kernel<7>;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());
  ctsQueue.submit([&](sycl::handler &cgh) { cgh.single_task(k_name()); });

  sycl::kernel kernelB = kernelA;

  std::hash<sycl::kernel> hasher;

  if (hasher(kernelA) != hasher(kernelB)) {
    FAIL(
        "kernel hashing does not work correctly (hashing of equal "
        "failed)");
  }

  ctsQueue.wait_and_throw();
}

} /* namespace kernel_constructors__ */
