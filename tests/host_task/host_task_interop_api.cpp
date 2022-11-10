/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Provide verification for sycl::interop_handle::get_native functions
//  this test check interop API with OpenCL back-end only.
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_BACKEND_OPENCL
#include <sycl/backend/opencl.hpp>
#endif  // SYCL_BACKEND_OPENCL

#define TEST_NAME host_task_interop_api

namespace TEST_NAMESPACE {
using namespace sycl_cts;

class TEST_NAME : public sycl_cts::util::test_base {
  /** return information about this test
   */
  void get_info(test_base::info& out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

#if defined(SYCL_BACKEND_OPENCL) && SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS
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
#endif  // defined(SYCL_BACKEND_OPENCL) && SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS

  /** execute this test
   */
  void run(util::logger& log) override {
#if defined(SYCL_BACKEND_OPENCL) && SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS
    {
      sycl::queue q{util::get_cts_object::queue()};
      if (q.get_backend() != sycl::backend::opencl) {
        log.note("Interop part is not supported on OpenCL backend type");
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

        if (cl_native_queue != sycl::get_native<sycl::backend::opencl>(q))
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

        if (cl_native_device_id !=
            sycl::get_native<sycl::backend::opencl>(q.get_device()))
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

        if (cl_native_context !=
            sycl::get_native<sycl::backend::opencl>(q.get_context()))
          FAIL(log, "get_native_context query has failed.");
      }

      // execute OpenCL function
      {
        const size_t size{16};
        const size_t pattern{13};
        sycl::buffer<size_t, 1> buf(sycl::range<1>{size});
        q.submit([&](sycl::handler& cgh) {
          auto buf_acc_dev{buf.get_access<sycl::access_mode::read_write>(cgh)};
          cgh.host_task([=](sycl::interop_handle ih) {
            cl_command_queue native_queue = ih.get_native_queue();
            std::vector<cl_mem> native_mems = ih.get_native_mem(buf_acc_dev);
            for (auto native_mem : native_mems)
              call_opencl(native_queue, native_mem, size, pattern);
          });
        });

        {
          auto buf_acc_host{buf.get_access<sycl::access_mode::read>()};
          for (int i = 0; i < size; ++i) {
            if (buf_acc_host[i] != pattern)
              FAIL(log, "OpenCL invocation has failed.");
          }
        }
      }
    }
#else
    log.note(
        "The test is skipped because interop testing is disabled or OpenCL "
        "back-end is not supported");
#endif  // defined(SYCL_BACKEND_OPENCL) && SYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS
  }
};

util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
