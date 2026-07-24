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

template <int a>
class test_kernel {
 public:
  void operator()() const {}
};

namespace kernel_constructors__ {
using namespace sycl_cts;

TEST_CASE("Test copy constructor", "[kernel]") {
  auto ctsQueue = util::get_cts_object::queue(cts_selector);
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  using k_name = test_kernel<0>;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());

  sycl::kernel kernelB(kernelA);

  ctsQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<k_name>(k_name()); });

#ifdef SYCL_BACKEND_OPENCL
  if (ctsQueue.get_backend() == sycl::backend::opencl) {
    auto iopKernelA = sycl::get_native<sycl::backend::opencl>(kernelA);
    auto iopKernelB = sycl::get_native<sycl::backend::opencl>(kernelB);

    INFO(
        "kernel was not constructed correctly. (contains different "
        "OpenCL kernel object)");
    CHECK(iopKernelA == iopKernelB);
  }
#endif

  ctsQueue.wait_and_throw();
}

TEST_CASE("Test assignment operator", "[kernel]") {
  auto ctsQueue = util::get_cts_object::queue(cts_selector);
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  using k_name = test_kernel<1>;
  auto kb =
      sycl::get_kernel_bundle<k_name, sycl::bundle_state::executable>(ctx);
  auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());

  ctsQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<k_name>(k_name()); });

  sycl::kernel kernelB = kernelA;

#ifdef SYCL_BACKEND_OPENCL
  if (ctsQueue.get_backend() == sycl::backend::opencl) {
    auto iopKernelA = sycl::get_native<sycl::backend::opencl>(kernelA);
    auto iopKernelB = sycl::get_native<sycl::backend::opencl>(kernelB);

    INFO(
        "kernel was not constructed correctly. (contains different "
        "OpenCL kernel object)");
    CHECK(iopKernelA == iopKernelB);
  }
#endif

  ctsQueue.wait_and_throw();
}

TEST_CASE("Test equality operator", "[kernel]") {
  auto ctsQueue = util::get_cts_object::queue(cts_selector);
  auto deviceList = ctsQueue.get_context().get_devices();
  auto ctx = ctsQueue.get_context();

  using k_name4 = test_kernel<4>;
  auto kbA =
      sycl::get_kernel_bundle<k_name4, sycl::bundle_state::executable>(ctx);
  auto kernelA = kbA.get_kernel(sycl::get_kernel_id<k_name4>());
  ctsQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<k_name4>(k_name4()); });
  sycl::kernel kernelB(kernelA);

  using k_name5 = test_kernel<5>;
  auto kbC =
      sycl::get_kernel_bundle<k_name5, sycl::bundle_state::executable>(ctx);
  auto kernelC = kbC.get_kernel(sycl::get_kernel_id<k_name5>());
  ctsQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<k_name5>(k_name5()); });
  kernelC = (kernelA);

  using k_name6 = test_kernel<6>;
  auto kbD =
      sycl::get_kernel_bundle<k_name6, sycl::bundle_state::executable>(ctx);
  auto kernelD = kbD.get_kernel(sycl::get_kernel_id<k_name6>());

  ctsQueue.submit(
      [&](sycl::handler &cgh) { cgh.single_task<k_name6>(k_name6()); });

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
    {
      INFO(
          "kernel equality does not work correctly (copy "
          "constructed)");
      CHECK(
          (kernelA != kernelB || (iopKernelA == iopKernelB &&
                                  iopCtxA == iopCtxB && iopProgA == iopProgB)));
    }
    {
      INFO("kernel equality does not work correctly (copy assigned)");
      CHECK(
          (kernelA != kernelC || (iopKernelA == iopKernelC &&
                                  iopCtxA == iopCtxC && iopProgA == iopProgC)));
    }
  }
#endif

  ctsQueue.wait_and_throw();
}

TEST_CASE("Test default constructor is deleted", "[kernel]") {
  STATIC_CHECK(!std::is_default_constructible_v<sycl::kernel>);
}

} /* namespace kernel_constructors__ */
