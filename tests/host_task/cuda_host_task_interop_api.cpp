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

#endif  // SYCL_BACKEND_CUDA

using namespace sycl_cts::util;

#ifdef SYCL_BACKEND_CUDA
CUresult call_cuda(CUcontext ctx, CUdeviceptr mem, size_t size,
                   uint32_t pattern) {
  cuCtxSetCurrent(ctx);
  CUresult ret = cuMemsetD32(mem, pattern, size);
  INFO("Checking if CUDA invocation returned error code.");
  CHECK(ret == CUDA_SUCCESS);

  return ret;
}
#endif  // SYCL_BACKEND_CUDA

TEST_CASE("CUDA host task interop test") {
#ifdef SYCL_BACKEND_CUDA
  sycl::queue queue{get_cts_object::queue()};

  INFO("Checking queue is using CUDA backend");
  REQUIRE(queue.get_backend() == sycl::backend::cuda);

  // check get_native_queue
  {
    CUstream cu_native_queue;
    queue.submit([&](sycl::handler& cgh) {
      cgh.host_task([=, &cu_native_queue](sycl::interop_handle ih) {
        cu_native_queue = ih.get_native_queue<sycl::backend::cuda>();
      });
    });
    queue.wait_and_throw();
  }

  // check get_native_device
  {
    CUdevice cu_native_device;
    queue.submit([&](sycl::handler& cgh) {
      cgh.host_task([=, &cu_native_device](sycl::interop_handle ih) {
        cu_native_device = ih.get_native_device<sycl::backend::cuda>();
      });
    });
    queue.wait_and_throw();

    INFO("Checking get_native_device query was successful");
    CHECK(cu_native_device ==
          sycl::get_native<sycl::backend::cuda>(queue.get_device()));
  }

  // check get_native_context
  {
    std::vector<CUcontext> cu_native_context{nullptr};
    queue.submit([&](sycl::handler& cgh) {
      cgh.host_task([=, &cu_native_context](sycl::interop_handle ih) {
        cu_native_context = ih.get_native_context<sycl::backend::cuda>();
      });
    });
    queue.wait_and_throw();

    INFO("Checking get_native_context query was successful");
    CHECK(cu_native_context ==
          sycl::get_native<sycl::backend::cuda>(queue.get_context()));
  }

  // execute CUDA function
  {
    const size_t size{16};
    const uint32_t pattern{13};
    sycl::buffer<uint32_t, 1> buf(sycl::range<1>{size});
    queue.submit([&](sycl::handler& cgh) {
      auto buf_acc_dev{buf.get_access<sycl::access_mode::read_write>(cgh)};
      cgh.host_task([=](sycl::interop_handle ih) {
        auto* native_mem = ih.get_native_mem<sycl::backend::cuda>(buf_acc_dev);

        std::vector<CUcontext> native_ctx =
            ih.get_native_context<sycl::backend::cuda>();

        // must have at least one context
        assert(native_ctx.size() > 0);
        call_cuda(native_ctx[0], CUdeviceptr(native_mem), size, pattern);
      });
    });
    queue.wait_and_throw();

    {
      INFO("Checking correct pattern has been set");
      auto buf_acc_host{buf.get_access<sycl::access_mode::read>()};
      for (int i = 0; i < size; ++i) {
        CHECK(buf_acc_host[i] == pattern);
      }
    }
  }
#else
  SKIP("CUDA backend is not supported");
#endif  // SYCL_BACKEND_CUDA
}
