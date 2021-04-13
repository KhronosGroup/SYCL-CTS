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

/** test cl::sycl::kernel
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
          cl::sycl::program prog(ctsQueue.get_context());

          prog.build_with_kernel_type<test_kernel<0>>();
          auto kernelA = prog.get_kernel<test_kernel<0>>();

          cl::sycl::kernel kernelB(kernelA);

          ctsQueue.submit([&](cl::sycl::handler &cgh) {
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
          cl::sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<1>>();
          auto kernelA = prog.get_kernel<test_kernel<1>>();

          ctsQueue.submit([&](cl::sycl::handler &cgh) {
            cgh.single_task(test_kernel<1>());
          });

          cl::sycl::kernel kernelB = kernelA;

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
          cl::sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<2>>();
          auto kernelA = prog.get_kernel<test_kernel<2>>();

          ctsQueue.submit([&](cl::sycl::handler &cgh) {
            cgh.single_task(test_kernel<2>());
          });

          cl::sycl::kernel kernelB(std::move(kernelA));

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
          cl::sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<3>>();
          auto kernelA = prog.get_kernel<test_kernel<3>>();
          ctsQueue.submit([&](cl::sycl::handler &cgh) {
            cgh.single_task(test_kernel<3>());
          });

          cl::sycl::kernel kernelB = std::move(kernelA);

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
          cl::sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<4>>();
          auto kernelA = prog.get_kernel<test_kernel<4>>();
          ctsQueue.submit([&](cl::sycl::handler &cgh) {
            cgh.single_task(test_kernel<4>());
          });

          cl::sycl::kernel kernelB(kernelA);

          cl::sycl::program progC(ctsQueue.get_context());
          progC.build_with_kernel_type<test_kernel<5>>();
          auto kernelC = progC.get_kernel<test_kernel<5>>();
          ctsQueue.submit([&](cl::sycl::handler &cgh) {
            cgh.single_task(test_kernel<5>());
          });
          kernelC = (kernelA);

          cl::sycl::program progD(ctsQueue.get_context());
          progD.build_with_kernel_type<test_kernel<6>>();
          auto kernelD = progD.get_kernel<test_kernel<6>>();
          ctsQueue.submit([&](cl::sycl::handler &cgh) {
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
          cl::sycl::program prog(ctsQueue.get_context());
          prog.build_with_kernel_type<test_kernel<7>>();
          auto kernelA = prog.get_kernel<test_kernel<7>>();
          ctsQueue.submit([&](cl::sycl::handler &cgh) {
            cgh.single_task(test_kernel<7>());
          });

          cl::sycl::kernel kernelB = kernelA;

          cl::sycl::hash_class<cl::sycl::kernel> hasher;

          if (hasher(kernelA) != hasher(kernelB)) {
            FAIL(log,
                 "kernel hashing does not work correctly (hashing of equal "
                 "failed)");
          }

          ctsQueue.wait_and_throw();
        }
      }
    } catch (const cl::sycl::exception &e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace kernel_constructors__ */
