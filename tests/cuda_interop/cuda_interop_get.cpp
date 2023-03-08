/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
*******************************************************************************/

#include "cuda_helper.hpp"

using namespace sycl_cts::util;

TEST_CASE("CUDA interop get test") {
#ifdef SYCL_BACKEND_CUDA
  auto queue = get_cts_object::queue();

  INFO("Checking queue is using CUDA backend");
  REQUIRE(queue.get_backend() == sycl::backend::cuda);

  /** check get_native() for device
   */
  {
    auto device = get_cts_object::device(cts_selector);
    auto interopDevice = sycl::get_native<sycl::backend::cuda>(device);
    check_return_type<CUdevice>(interopDevice, "get_native(device)");
    int n_devices;
    CUDA_CHECK(cuDeviceGetCount(&n_devices));

    INFO("Checking get_native(device) returned valid CUdevice");
    bool is_valid_device = (interopDevice >= 0) && (interopDevice < n_devices);
    CHECK(is_valid_device);
  }

  /** check get_native() for context
   */
  {
    auto context = get_cts_object::context(cts_selector);
    auto interopContext = sycl::get_native<sycl::backend::cuda>(context);
    check_return_type<std::vector<CUcontext>>(interopContext,
                                              "get_native(context)");

    INFO(
        "Checking get_native(Context) returned a valid std::vector<CUcontext>");
    CHECK(interopContext.size() != 0);
  }

  /** check get_native() for queue
   */
  {
    auto ctsQueue = get_cts_object::queue(cts_selector);
    auto interopQueue = sycl::get_native<sycl::backend::cuda>(ctsQueue);
    check_return_type<CUstream>(interopQueue, "get_native(queue)");
  }

  /** check get_native() for event
   */
  {
    auto ctsQueue = get_cts_object::queue(cts_selector);

    sycl::event event = ctsQueue.submit([&](sycl::handler &cgh) {
      cgh.single_task<class event_kernel>([] {});
    });

    auto interopEvent = sycl::get_native<sycl::backend::cuda>(event);
    check_return_type<CUevent>(interopEvent, "get_native(event)");
  }
#else
  SKIP("CUDA backend is not supported");
#endif  // SYCL_BACKEND_CUDA
};
