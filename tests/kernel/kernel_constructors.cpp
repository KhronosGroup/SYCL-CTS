/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_constructors

template <int a>
class test_kernel {
 public:
  void operator()() const {}
};

class kernel0;
class kernel1;

namespace kernel_constructors__ {
using namespace sycl_cts;

/** test sycl::kernel
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      /* Test copy constructor
       */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto deviceList = ctsQueue.get_context().get_devices();
        auto ctx = ctsQueue.get_context();

        using k_name = test_kernel<0>;
        auto kb = sycl::get_kernel_bundle<
                    k_name, sycl::bundle_state::executable>(ctx);
        auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());

        sycl::kernel kernelB(kernelA);

        ctsQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task(k_name());
        });

#ifdef SYCL_BACKEND_OPENCL
        if (ctsQueue.get_backend() == sycl::backend::opencl) {
          auto iopKernelA =
              sycl::get_native<sycl::backend::opencl>(kernelA);
          auto iopKernelB =
              sycl::get_native<sycl::backend::opencl>(kernelB);
          if (!ctsSelector.is_host() && (iopKernelA != iopKernelB)) {
            FAIL(log,
                 "kernel was not constructed correctly. (contains different "
                 "OpenCL kernel object)");
          }
        }
#endif

        ctsQueue.wait_and_throw();
      }

      /* Test assignment operator
       */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto deviceList = ctsQueue.get_context().get_devices();
        auto ctx = ctsQueue.get_context();

        using k_name = test_kernel<1>;
        auto kb = sycl::get_kernel_bundle<
                    k_name, sycl::bundle_state::executable>(ctx);
        auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());

        ctsQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task(k_name());
        });

        sycl::kernel kernelB = kernelA;

#ifdef SYCL_BACKEND_OPENCL
        if (ctsQueue.get_backend() == sycl::backend::opencl) {
          auto iopKernelA =
              sycl::get_native<sycl::backend::opencl>(kernelA);
          auto iopKernelB =
              sycl::get_native<sycl::backend::opencl>(kernelB);
          if (!ctsSelector.is_host() && (iopKernelA != iopKernelB)) {
            FAIL(log,
                 "kernel was not constructed correctly. (contains different "
                 "OpenCL kernel object)");
          }
        }
#endif

        ctsQueue.wait_and_throw();
      }

      /* Test move constructor
       */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto deviceList = ctsQueue.get_context().get_devices();
        auto ctx = ctsQueue.get_context();

        using k_name = test_kernel<2>;
        auto kb = sycl::get_kernel_bundle<
                    k_name, sycl::bundle_state::executable>(ctx);
        auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());

        ctsQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task(k_name());
        });

        sycl::kernel kernelB(std::move(kernelA));

        ctsQueue.wait_and_throw();
      }

      /* Test move assignment operator
       */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto deviceList = ctsQueue.get_context().get_devices();
        auto ctx = ctsQueue.get_context();

        using k_name = test_kernel<3>;
        auto kb = sycl::get_kernel_bundle<
                    k_name, sycl::bundle_state::executable>(ctx);
        auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());
        ctsQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task(k_name());
        });

        sycl::kernel kernelB = std::move(kernelA);

        ctsQueue.wait_and_throw();
      }

      /* Test equality operator
       */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto deviceList = ctsQueue.get_context().get_devices();
        auto ctx = ctsQueue.get_context();

        using k_name4 = test_kernel<4>;
        auto kbA = sycl::get_kernel_bundle<
                      k_name4, sycl::bundle_state::executable>(ctx);
        auto kernelA = kbA.get_kernel(sycl::get_kernel_id<k_name4>());
        ctsQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task(k_name4());
        });
        sycl::kernel kernelB(kernelA);

        using k_name5 = test_kernel<5>;
        auto kbC = sycl::get_kernel_bundle<
                      k_name5, sycl::bundle_state::executable>(ctx);
        auto kernelC = kbC.get_kernel(sycl::get_kernel_id<k_name5>());
        ctsQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task(k_name5());
        });
        kernelC = (kernelA);

        using k_name6 = test_kernel<6>;
        auto kbD = sycl::get_kernel_bundle<
                      k_name6, sycl::bundle_state::executable>(ctx);
        auto kernelD = kbD.get_kernel(sycl::get_kernel_id<k_name6>());

        ctsQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task(k_name6());
        });

        if (!ctsSelector.is_host()) {
#ifdef SYCL_BACKEND_OPENCL
          if (ctsQueue.get_backend() == sycl::backend::opencl) {
            auto iopKernelA =
                sycl::get_native<sycl::backend::opencl>(kernelA);
            auto iopKernelB =
                sycl::get_native<sycl::backend::opencl>(kernelB);
            auto iopKernelC =
                sycl::get_native<sycl::backend::opencl>(kernelC);
            auto iopCtxA = sycl::get_native<sycl::backend::opencl>(
                              kernelA.get_context());
            auto iopCtxB = sycl::get_native<sycl::backend::opencl>(
                              kernelB.get_context());
            auto iopCtxC = sycl::get_native<sycl::backend::opencl>(
                              kernelC.get_context());
            auto iopProgA = sycl::get_native<sycl::backend::opencl>(
                              kernelA.get_kernel_bundle());
            auto iopProgB = sycl::get_native<sycl::backend::opencl>(
                              kernelB.get_kernel_bundle());
            auto iopProgC = sycl::get_native<sycl::backend::opencl>(
                              kernelC.get_kernel_bundle());

            if (kernelA == kernelB && (iopKernelA != iopKernelB ||
                iopCtxA != iopCtxB || iopProgA != iopProgB)) {
              FAIL(log, "kernel equality does not work correctly (copy "
                        "constructed)");
            }
            if (kernelA == kernelC &&
                (iopKernelA != iopKernelC || iopCtxA != iopCtxC ||
                iopProgA != iopProgC)) {
              FAIL(log,
                   "kernel equality does not work correctly (copy assigned)");
            }
          }
#endif
          if (kernelA != kernelB) {
            FAIL(log,
                 "kernel non-equality does not work correctly"
                 "(copy constructed)");
          }
          if (kernelA != kernelC) {
            FAIL(log,
                 "kernel non-equality does not work correctly"
                 "(copy assigned)");
          }
          if (kernelC == kernelD) {
            FAIL(log,
                 "kernel equality does not work correctly"
                 "(comparing same)");
          }
          if (!(kernelC != kernelD)) {
            FAIL(log,
                 "kernel non-equality does not work correctly"
                 "(comparing same)");
          }
        }

        ctsQueue.wait_and_throw();
      }

      /* Test hashing
       */
      {
        auto ctsQueue = util::get_cts_object::queue();
        auto deviceList = ctsQueue.get_context().get_devices();
        auto ctx = ctsQueue.get_context();

        using k_name = test_kernel<7>;
        auto kb = sycl::get_kernel_bundle<
                    k_name, sycl::bundle_state::executable>(ctx);
        auto kernelA = kb.get_kernel(sycl::get_kernel_id<k_name>());
        ctsQueue.submit([&](sycl::handler &cgh) {
          cgh.single_task(k_name());
        });

        sycl::kernel kernelB = kernelA;

        std::hash<sycl::kernel> hasher;

        if (hasher(kernelA) != hasher(kernelB)) {
          FAIL(log,
               "kernel hashing does not work correctly (hashing of equal "
               "failed)");
        }

        ctsQueue.wait_and_throw();
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      std::string errorMsg =
          "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_constructors__ */
