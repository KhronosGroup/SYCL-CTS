/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2022 The Khronos Group Inc.
//
*******************************************************************************/

#include "cuda_helper.hpp"

using namespace sycl_cts::util;

TEST_CASE("CUDA interop constructor test") {
#ifdef SYCL_BACKEND_CUDA
  auto queue = get_cts_object::queue();

  INFO("Checking queue is using CUDA backend");
  REQUIRE(queue.get_backend() == sycl::backend::cuda);

  /** create native objects
   */
  CUdevice m_cu_device;
  CUstream m_cu_stream;
  CUevent m_cu_event;

  const auto ctsContext = queue.get_context();
  CUDA_CHECK(cuDeviceGet(&m_cu_device, 0));

  /** check make_device (CUdevice)
   */
  {
    sycl::device device = sycl::make_device<sycl::backend::cuda>(m_cu_device);

    CUdevice interopDeviceID = sycl::get_native<sycl::backend::cuda>(device);

    INFO("Checking device was constructed correctly");
    CHECK(interopDeviceID == m_cu_device);
  }

  /** check make_context (CUcontext)
   */
  {
    CUcontext curr_context, m_cu_context;
    CUDA_CHECK(cuCtxGetCurrent(&curr_context));
    CUDA_CHECK(cuCtxCreate(&m_cu_context, CU_CTX_MAP_HOST, m_cu_device));
    CUDA_CHECK(cuCtxSetCurrent(curr_context));

    sycl::context context =
        sycl::make_context<sycl::backend::cuda>(m_cu_context);

    auto interopContext = sycl::get_native<sycl::backend::cuda>(context);

    INFO("Checking context was constructed correctly");
    bool found_m_cu_context = false;
    for (const auto &ctx : interopContext) {
      if (ctx == m_cu_context) found_m_cu_context = true;
    }

    CHECK(found_m_cu_context);
  }

  /** check make_context (CUcontext, async_handler)
   */
  {
    CUcontext curr_context, m_cu_context;
    CUDA_CHECK(cuCtxGetCurrent(&curr_context));
    CUDA_CHECK(cuCtxCreate(&m_cu_context, CU_CTX_MAP_HOST, m_cu_device));
    CUDA_CHECK(cuCtxSetCurrent(curr_context));

    cts_async_handler asyncHandler;
    sycl::context context =
        sycl::make_context<sycl::backend::cuda>(m_cu_context, asyncHandler);

    auto interopContext = sycl::get_native<sycl::backend::cuda>(context);

    INFO("Checking context was constructed correctly");
    bool found_m_cu_context = false;
    for (const auto &ctx : interopContext) {
      if (ctx == m_cu_context) found_m_cu_context = true;
    }

    CHECK(found_m_cu_context);
  }

  /** check make_queue (CUstream, const context&)
   */
  {
    CUDA_CHECK(cuStreamCreate(&m_cu_stream, CU_STREAM_DEFAULT));

    sycl::queue queue =
        sycl::make_queue<sycl::backend::cuda>(m_cu_stream, ctsContext);

    auto interopQueue = sycl::get_native<sycl::backend::cuda>(queue);
    INFO("Checking queue was constructed correctly");
    CHECK(interopQueue == m_cu_stream);

    sycl::queue queueCopy(queue);
    auto interopQueueCopy = sycl::get_native<sycl::backend::cuda>(queueCopy);
    INFO("Checking queue was copied correctly");
    CHECK(interopQueue == interopQueueCopy);
  }

  /** check make_queue (CUstream, const context&, async_handler)
   */
  {
    CUDA_CHECK(cuStreamCreate(&m_cu_stream, CU_STREAM_DEFAULT));
    cts_async_handler asyncHandler;
    sycl::queue queue = sycl::make_queue<sycl::backend::cuda>(
        m_cu_stream, ctsContext, asyncHandler);

    auto interopQueue = sycl::get_native<sycl::backend::cuda>(queue);
    INFO("Checking queue was constructed correctly");
    CHECK(interopQueue == m_cu_stream);
  }

  /** check make_event (CUevent, const context&)
   */
  {
    CUevent cuEvent;
    CUDA_CHECK(cuEventCreate(&cuEvent, CU_EVENT_DEFAULT));

    sycl::event event =
        sycl::make_event<sycl::backend::cuda>(cuEvent, ctsContext);

    CUevent interopEvent = sycl::get_native<sycl::backend::cuda>(event);

    INFO("Checking event was constructed correctly");
    CHECK(interopEvent == cuEvent);
  }
#else
  SKIP("CUDA backend is not supported");
#endif  // SYCL_BACKEND_CUDA
}
