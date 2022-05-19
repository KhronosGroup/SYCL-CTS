/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_BACKEND_CUDA
#include "../../util/test_base_cuda.h"

#endif

#define TEST_NAME cuda_interop_constructors

namespace cuda_interop_constructors__ {
using namespace sycl_cts;

/** tests the constructors for CUDA inter-op
 */
class TEST_NAME :
#ifdef SYCL_BACKEND_CUDA
    public util::test_base_cuda
#else
    public util::test_base
#endif
{
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
#ifdef SYCL_BACKEND_CUDA
    {
      auto queue = util::get_cts_object::queue();
      if (queue.get_backend() != sycl::backend::cuda) {
        WARN(
            "CUDA interoperability part is not supported on non-CUDA "
            "backend types");
        return;
      }

      cts_selector ctsSelector;
      const auto ctsContext = util::get_cts_object::context(ctsSelector);
      auto res = cuDeviceGet(&m_cu_device, 0);
      m_cu_platform.push_back(m_cu_device);

      CUcontext curr_context;
      assert(0 == cuCtxGetCurrent(&curr_context));
      cuCtxCreate(&m_cu_context, CU_CTX_MAP_HOST, m_cu_device);
      assert(0 == cuCtxSetCurrent(curr_context));

      /** check make_platform (std::vector<CUdevice>)
       */
      {
        sycl::platform platform =
            sycl::make_platform<sycl::backend::cuda>(m_cu_platform);

        std::vector<CUdevice> interopPlatformID =
            sycl::get_native<sycl::backend::cuda>(platform);
        if (interopPlatformID != m_cu_platform) {
          FAIL(log, "platform was not constructed correctly");
        }
      }

      /** check make_device (CUdevice)
       */
      {
        sycl::device device =
            sycl::make_device<sycl::backend::cuda>(m_cu_device);

        CUdevice interopDeviceID =
            sycl::get_native<sycl::backend::cuda>(device);
        if (interopDeviceID != m_cu_device) {
          FAIL(log, "device was not constructed correctly");
        }
      }

      /** check make_context (CUcontext)
       */
      {
        CUcontext curr_context, m_cu_context;
        assert(0 == cuCtxGetCurrent(&curr_context));
        cuCtxCreate(&m_cu_context, CU_CTX_MAP_HOST, m_cu_device);
        assert(0 == cuCtxSetCurrent(curr_context));

        sycl::context context =
            sycl::make_context<sycl::backend::cuda>(m_cu_context);

        auto interopContext =
            sycl::get_native<sycl::backend::cuda>(context);

        bool found_m_cu_context = false;
        for (const auto &ctx : interopContext) {
            if (ctx == m_cu_context)
                found_m_cu_context = true;
        }

        if (!found_m_cu_context)
          FAIL(log, "context was not constructed correctly");
      }

      /** check make_context (CUcontext, async_handler)
       */
      {
        CUcontext curr_context, m_cu_context;
        assert(0 == cuCtxGetCurrent(&curr_context));
        cuCtxCreate(&m_cu_context, CU_CTX_MAP_HOST, m_cu_device);
        assert(0 == cuCtxSetCurrent(curr_context));

        cts_async_handler asyncHandler;
        sycl::context context =
            sycl::make_context<sycl::backend::cuda>(m_cu_context,
                                                               asyncHandler);

        auto interopContext =
            sycl::get_native<sycl::backend::cuda>(context);

        bool found_m_cu_context = false;
        for (const auto &ctx : interopContext) {
            if (ctx == m_cu_context)
                found_m_cu_context = true;
        }

        if (!found_m_cu_context)
          FAIL(log, "context was not constructed correctly");
      }

      /** check make_queue (CUstream, const context&)
       */
      {
        cuStreamCreate(&m_cu_stream, CU_STREAM_DEFAULT);

        sycl::queue queue = sycl::make_queue<sycl::backend::cuda>(
            m_cu_stream, ctsContext);

        auto interopQueue =
            sycl::get_native<sycl::backend::cuda>(queue);
        if (interopQueue != m_cu_stream) {
          FAIL(log, "queue was not constructed correctly");
        }

        sycl::queue queueCopy(queue);
        auto interopQueueCopy =
            sycl::get_native<sycl::backend::cuda>(queueCopy);
        if (interopQueue != interopQueueCopy) {
          FAIL(log, "queue destination was not copy constructed correctly");
        }
      }

      /** check make_queue (CUstream, const context&, async_handler)
       */
      {
        cuStreamCreate(&m_cu_stream, CU_STREAM_DEFAULT);
        cts_async_handler asyncHandler;
        sycl::queue queue = sycl::make_queue<sycl::backend::cuda>(
            m_cu_stream, ctsContext, asyncHandler);

        auto interopQueue =
            sycl::get_native<sycl::backend::cuda>(queue);
        if (interopQueue != m_cu_stream) {
          FAIL(log, "queue was not constructed correctly");
        }
      }

      /** check make_event (CUevent, const context&)
       */
      {
        CUevent cuEvent;
        auto res = cuEventCreate(&cuEvent, CU_EVENT_DEFAULT);

        sycl::event event = sycl::make_event<sycl::backend::cuda>(
            cuEvent, ctsContext);

        CUevent interopEvent =
            sycl::get_native<sycl::backend::cuda>(event);

        if (interopEvent != cuEvent) {
          FAIL(log, "event was not constructed correctly");
        }
      }
    }
#else
    log.note("The test is skipped because CUDA back-end is not supported");
#endif  // SYCL_BACKEND_CUDA
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace cuda_interop_constructors__ */
