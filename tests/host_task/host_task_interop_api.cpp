/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification for sycl::interop_handle::get_native functions
//  this test check interop API with OpenCL back-end only.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME host_task_interop_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class TEST_NAME : public sycl_cts::util::test_base {
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

#ifdef SYCL_BACKEND_OPENCL
  cl_int call_opencl(cl_command_queue q, cl_mem mem, size_t size,
                     size_t pattern) {
    cl_event e{};
    cl_int ret{clEnqueueFillBuffer(q, mem, &pattern, sizeof(size_t),
                                   /*offset*/ 0, size * sizeof(size_t), 0,
                                   nullptr, &e)};
    if (ret == CL_SUCCESS) {
      ret = clWaitForEvents(1, &e);
    }
    return ret;
  }
#endif  // SYCL_BACKEND_OPENCL

  /** execute this test
   */
  void run(util::logger& log) override {
#ifdef SYCL_BACKEND_OPENCL
    for_type_and_vectors<check_buffer_ctors_for_type, sycl::cl_half>(
        log, "sycl::cl_half");
    try {
      sycl::queue q{util::get_cts_object::queue()};
      if (q.get_backend() != sycl::backend::opencl) {
        log.note("Interop part is not supported on this backend type");
        return;
      }

      // check get_native_queue
      {
        cl_command_queue cl_native_queue{nullptr};
        q.submit([&](sycl::handler& cgh) {
          cgh.host_task([=, &cl_native_queue](sycl::interop_handle ih) {
            cl_native_queue = ih.get_native_queue();
          });
        });
        q.wait_and_throw();

        if (cl_native_queue != q.get())
          FAIL(log, "get_native_queue query has failed.");
      }

      // check get_native_device
      {
        cl_device_id cl_native_device_id{nullptr};
        q.submit([&](sycl::handler& cgh) {
          cgh.host_task([=, &cl_native_device_id](sycl::interop_handle ih) {
            cl_native_device_id = ih.get_native_device();
          });
        });
        q.wait_and_throw();

        if (cl_native_device_id != q.get_device().get())
          FAIL(log, "get_native_device query has failed.");
      }

      // check get_native_context
      {
        cl_context cl_native_context{nullptr};
        q.submit([&](sycl::handler& cgh) {
          cgh.host_task([=, &cl_native_context](sycl::interop_handle ih) {
            cl_native_context = ih.get_native_context();
          });
        });
        q.wait_and_throw();

        if (cl_native_context != q.get_context().get())
          FAIL(log, "get_native_context query has failed.");
      }

      // execute OpenCL function
      {
        const size_t size{16};
        const size_t pattern{13};
        sycl::buffer<size_t, 1> buf(sycl::range<1>{size});
        q.submit([&](sycl::handler& cgh) {
          auto buf_acc_dev{buf.get_access<sycl::access::mode::read_write>(cgh)};
          cgh.host_task([=](sycl::interop_handle ih) {
            cl_command_queue native_queue = ih.get_native_queue();
            cl_mem native_mem = ih.get_native_mem(buf_acc_dev);
            call_opencl(native_queue, native_mem, size, pattern);
          });
        });

        {
          auto buf_acc_host{buf.get_access<sycl::access::mode::read>()};
          for (int i = 0; i < size; ++i) {
            if (buf_acc_host[i] != pattern)
              FAIL(log, "OpenCL invocation has failed.");
          }
        }
      }
    } catch (const sycl::exception& e) {
      log_exception(log, e);
      FAIL(log, "An unexpected SYCL exception was caught");
    }
#else
    log.note("The test is skipped because OpenCL back-end is not supported");
#endif  // SYCL_BACKEND_OPENCL
  }
};

util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
