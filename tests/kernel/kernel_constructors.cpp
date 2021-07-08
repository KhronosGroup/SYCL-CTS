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

        if (!is_compiler_available(deviceList)) {
          log.note(
              "online compiler is not available -- skipping test of copy "
              "constructor");
        } else {
          sycl::program prog(ctsQueue.get_context());

          prog.build_with_kernel_type<test_kernel<0>>();
          auto kernelA = prog.get_kernel<test_kernel<0>>();

          sycl::kernel kernelB(kernelA);

          ctsQueue.submit([&](sycl::handler &cgh) {
            cgh.single_task(test_kernel<0>());
          });

#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
          if (!ctsSelector.is_host() && (kernelA.get() != kernelB.get())) {
            FAIL(log,
                 "kernel was not constructed correctly. (contains different "
                 "OpenCL kernel object)");
          }
#endif

          ctsQueue.wait_and_throw();
        }
      }

      /* Test assignment operator
       */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto deviceList = ctsQueue.get_context().get_devices();

        if (!is_compiler_available(deviceList)) {
          log.note(
              "online compiler is not available -- skipping test of assignment "
              "operator");
        } else {
          sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<1>>();
          auto kernelA = prog.get_kernel<test_kernel<1>>();

          ctsQueue.submit([&](sycl::handler &cgh) {
            cgh.single_task(test_kernel<1>());
          });

          sycl::kernel kernelB = kernelA;

#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
          if (!ctsSelector.is_host() && (kernelA.get() != kernelB.get())) {
            FAIL(log,
                 "kernel was not constructed correctly. (contains different "
                 "OpenCL kernel object)");
          }
#endif

          ctsQueue.wait_and_throw();
        }
      }

      /* Test move constructor
       */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto deviceList = ctsQueue.get_context().get_devices();

        if (!is_compiler_available(deviceList)) {
          log.note(
              "online compiler is not available -- skipping test of move "
              "constructor");
        } else {
          sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<2>>();
          auto kernelA = prog.get_kernel<test_kernel<2>>();

          ctsQueue.submit([&](sycl::handler &cgh) {
            cgh.single_task(test_kernel<2>());
          });

          sycl::kernel kernelB(std::move(kernelA));

          ctsQueue.wait_and_throw();
        }
      }

      /* Test move assignment operator
       */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto deviceList = ctsQueue.get_context().get_devices();

        if (!is_compiler_available(deviceList)) {
          log.note(
              "online compiler is not available -- skipping test of move "
              "assignment operator");
        } else {
          sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<3>>();
          auto kernelA = prog.get_kernel<test_kernel<3>>();
          ctsQueue.submit([&](sycl::handler &cgh) {
            cgh.single_task(test_kernel<3>());
          });

          sycl::kernel kernelB = std::move(kernelA);

          ctsQueue.wait_and_throw();
        }
      }

      /* Test equality operator
       */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);
        auto deviceList = ctsQueue.get_context().get_devices();

        if (!is_compiler_available(deviceList)) {
          log.note(
              "online compiler is not available -- skipping test of equality "
              "operator");
        } else {
          sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<4>>();
          auto kernelA = prog.get_kernel<test_kernel<4>>();
          ctsQueue.submit([&](sycl::handler &cgh) {
            cgh.single_task(test_kernel<4>());
          });

          sycl::kernel kernelB(kernelA);

          sycl::program progC(ctsQueue.get_context());
          progC.build_with_kernel_type<test_kernel<5>>();
          auto kernelC = progC.get_kernel<test_kernel<5>>();
          ctsQueue.submit([&](sycl::handler &cgh) {
            cgh.single_task(test_kernel<5>());
          });
          kernelC = (kernelA);

          sycl::program progD(ctsQueue.get_context());
          progD.build_with_kernel_type<test_kernel<6>>();
          auto kernelD = progD.get_kernel<test_kernel<6>>();
          ctsQueue.submit([&](sycl::handler &cgh) {
            cgh.single_task(test_kernel<6>());
          });

          if (!ctsSelector.is_host()) {
#ifdef SYCL_CTS_TEST_OPENCL_INTEROP
            if (kernelA == kernelB &&
                (kernelA.get() != kernelB.get() ||
                 kernelA.get_context().get() != kernelB.get_context().get() ||
                 kernelA.get_program().get() != kernelB.get_program().get())) {
              FAIL(
                  log,
                  "kernel equality does not work correctly (copy constructed)");
            }
            if (kernelA == kernelC &&
                (kernelA.get() != kernelC.get() ||
                 kernelA.get_context().get() != kernelC.get_context().get() ||
                 kernelA.get_program().get() != kernelC.get_program().get())) {
              FAIL(log,
                   "kernel equality does not work correctly (copy assigned)");
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
      }

      /* Test hashing
       */
      {
        auto ctsQueue = util::get_cts_object::queue();
        auto deviceList = ctsQueue.get_context().get_devices();

        if (!is_compiler_available(deviceList)) {
          log.note(
              "online compiler is not available -- skipping test of hashing");
        } else {
          sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<7>>();
          auto kernelA = prog.get_kernel<test_kernel<7>>();
          ctsQueue.submit([&](sycl::handler &cgh) {
            cgh.single_task(test_kernel<7>());
          });

          sycl::kernel kernelB = kernelA;

          sycl::hash_class<sycl::kernel> hasher;

          if (hasher(kernelA) != hasher(kernelB)) {
            FAIL(log,
                 "kernel hashing does not work correctly (hashing of equal "
                 "failed)");
          }

          ctsQueue.wait_and_throw();
        }
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      sycl::string_class errorMsg =
          "a SYCL exception was caught: " + sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_constructors__ */
