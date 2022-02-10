/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
#include "../../util/test_base_cuda.h"

#endif

#define TEST_NAME cuda_interop_get

namespace cuda_interop_get__ {
using namespace sycl_cts;

/** tests the get_native() methods for CUDA inter-op
 */
class TEST_NAME :
#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    public sycl_cts::util::test_base_cuda
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
#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    {
      auto queue = util::get_cts_object::queue();
      if (queue.get_backend() != sycl::backend::ext_oneapi_cuda) {
        WARN(
            "CUDA interoperability part is not supported on non-CUDA "
            "backend types");
        return;
      }
      cts_selector ctsSelector;

      /** check get_native() for platform
       */
      {
        auto platform = util::get_cts_object::platform(ctsSelector);
        auto interopPlatform =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(platform);
        check_return_type<std::vector<CUdevice>>(log, interopPlatform,
                                                 "get_native(platform)");

        if (interopPlatform.size() == 0) {
          FAIL(log,
               "get_native(platform) did not return a valid "
               "std::vector<CUdevice>");
        }
      }

      /** check get_native() for device
       */
      {
        auto device = util::get_cts_object::device(ctsSelector);
        auto interopDevice =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
        check_return_type<CUdevice>(log, interopDevice, "get_native(device)");
        int n_devices;
        cuDeviceGetCount(&n_devices);

        if (interopDevice < 0 || interopDevice >= n_devices) {
          FAIL(log, "get_native(device) did not return a valid CUdevice");
        }
      }

      /** check get_native() for context
       */
      {
        auto context = util::get_cts_object::context(ctsSelector);
        auto interopContext =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(context);
        check_return_type<std::vector<CUcontext>>(log, interopContext,
                                                  "get_native(context)");

        if (interopContext.size() == 0) {
          FAIL(log,
               "get_native(Context) did not return a valid "
               "std::vector<CUcontext>");
        }
      }

      /** check get_native() for queue
       */
      {
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto interopQueue =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(ctsQueue);
        check_return_type<CUstream>(log, interopQueue, "get_native(queue)");
      }

      /** check get_native() for event
       */
      {
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);

        sycl::event event = ctsQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task<class event_kernel>([](){});
        });

        auto interopEvent =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(event);
        check_return_type<CUevent>(log, interopEvent, "get_native(event)");
      }
    }
#else
    log.note("The test is skipped because CUDA back-end is not supported");
#endif  // SYCL_EXT_ONEAPI_BACKEND_CUDA
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace cuda_interop_get__ */
