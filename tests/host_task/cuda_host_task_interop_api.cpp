/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification for sycl::interop_handle::get_native functions
//  this test check interop API with CUDA back-end only.
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_BACKEND_CUDA

#include <cuda.h>
#include <sycl/ext/oneapi/experimental/backend/cuda.hpp>

#endif  // SYCL_BACKEND_CUDA

#define TEST_NAME cuda_host_task_interop_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class TEST_NAME : public sycl_cts::util::test_base {
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

#ifdef SYCL_BACKEND_CUDA
  CUresult call_cuda(CUcontext ctx, CUdeviceptr mem, size_t size,
                     uint32_t pattern) {
    cuCtxSetCurrent(ctx);
    CUresult ret = cuMemsetD32(mem, pattern, size);
    if (ret != CUDA_SUCCESS) FAIL(log, "CUDA invocation returned error code.");

    return ret;
  }
#endif  // SYCL_BACKEND_CUDA

  /** execute this test
   */
  void run(util::logger& log) override {
#ifdef SYCL_BACKEND_CUDA
    {
      sycl::queue q{util::get_cts_object::queue()};
      if (q.get_backend() != sycl::backend::cuda) {
        log.note("Interop part is not supported on CUDA backend type");
        return;
      }

      // check get_native_queue
      {
        CUstream cu_native_queue;
        q.submit([&](sycl::handler& cgh) {
          cgh.host_task([=, &cu_native_queue](sycl::interop_handle ih) {
            cu_native_queue = ih.get_native_queue<sycl::backend::cuda>();
          });
        });
        q.wait_and_throw();

        if (cu_native_queue != sycl::get_native<sycl::backend::cuda>(q))
          FAIL(log, "get_native_queue query has failed.");
      }

      // check get_native_device
      {
        CUdevice cu_native_device;
        q.submit([&](sycl::handler& cgh) {
          cgh.host_task([=, &cu_native_device](sycl::interop_handle ih) {
            cu_native_device = ih.get_native_device<sycl::backend::cuda>();
          });
        });
        q.wait_and_throw();

        if (cu_native_device !=
            sycl::get_native<sycl::backend::cuda>(q.get_device()))
          FAIL(log, "get_native_device query has failed.");
      }

      // check get_native_context
      {
        std::vector<CUcontext> cu_native_context{nullptr};
        q.submit([&](sycl::handler& cgh) {
          cgh.host_task([=, &cu_native_context](sycl::interop_handle ih) {
            cu_native_context = ih.get_native_context<sycl::backend::cuda>();
          });
        });
        q.wait_and_throw();

        if (cu_native_context !=
            sycl::get_native<sycl::backend::cuda>(q.get_context()))
          FAIL(log, "get_native_context query has failed.");
      }

      // execute CUDA function
      {
        const size_t size{16};
        const uint32_t pattern{13};
        sycl::buffer<uint32_t, 1> buf(sycl::range<1>{size});
        q.submit([&](sycl::handler& cgh) {
           auto buf_acc_dev{buf.get_access<sycl::access_mode::read_write>(cgh)};
           cgh.host_task([=](sycl::interop_handle ih) {
             auto* native_mem =
                 ih.get_native_mem<sycl::backend::cuda>(buf_acc_dev);

             std::vector<CUcontext> native_ctx =
                 ih.get_native_context<sycl::backend::cuda>();

             // must have at least one context
             assert(native_ctx.size() > 0);
             call_cuda(native_ctx[0], CUdeviceptr(native_mem), size, pattern);
           });
         }).wait();

        {
          auto buf_acc_host{buf.get_access<sycl::access_mode::read>()};
          for (int i = 0; i < size; ++i) {
            if (buf_acc_host[i] != pattern)
              FAIL(log, "CUDA invocation has failed.");
          }
          std::cout << std::endl;
        }
      }
    }
#else
    log.note("The test is skipped because CUDA back-end is not supported");
#endif  // SYCL_BACKEND_CUDA
  }
};

util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
