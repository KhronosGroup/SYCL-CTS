/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME kernel_constructors

template <int a>
class test_kernel {
 public:
  void operator()() {}
};

class kernel0;
class kernel1;

namespace kernel_constructors__ {
using namespace sycl_cts;

/** test cl::sycl::kernel
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl {
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

        cl::sycl::program prog(ctsQueue.get_context());
        prog.build_from_kernel_name<test_kernel<0>>();
        auto kernelA = prog.get_kernel<test_kernel<0>>();

        cl::sycl::kernel kernelB(kernelA);

        ctsQueue.submit(
            [&](cl::sycl::handler &cgh) { cgh.single_task(test_kernel<0>()); });

        if (!ctsSelector.is_host() && (kernelA.get() != kernelB.get())) {
          FAIL(log,
               "kernel was not constructed correctly. (contains different "
               "OpenCL kernel object)");
        }
      }

      /* Test assignment operator
      */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);

        cl::sycl::program prog(ctsQueue.get_context());
        prog.build_from_kernel_name<test_kernel<1>>();
        auto kernelA = prog.get_kernel<test_kernel<1>>();

        ctsQueue.submit(
            [&](cl::sycl::handler &cgh) { cgh.single_task(test_kernel<1>()); });

        cl::sycl::kernel kernelB = kernelA;

        if (!ctsSelector.is_host() && (kernelA.get() != kernelB.get())) {
          FAIL(log,
               "kernel was not constructed correctly. (contains different "
               "OpenCL kernel object)");
        }
      }

      /* Test move constructor
      */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);

        cl::sycl::program prog(ctsQueue.get_context());
        prog.build_from_kernel_name<test_kernel<2>>();
        auto kernelA = prog.get_kernel<test_kernel<2>>();

        ctsQueue.submit(
            [&](cl::sycl::handler &cgh) { cgh.single_task(test_kernel<2>()); });

        cl::sycl::kernel kernelB(std::move(kernelA));

        if (!ctsSelector.is_host() && kernelB.get() == nullptr) {
          FAIL(log,
               "kernel was not move constructed correctly. (contains different "
               "OpenCL kernel object)");
        }
      }

      /* Test move assignment operator
      */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);

        cl::sycl::program prog(ctsQueue.get_context());
        prog.build_from_kernel_name<test_kernel<3>>();
        auto kernelA = prog.get_kernel<test_kernel<3>>();
        ctsQueue.submit(
            [&](cl::sycl::handler &cgh) { cgh.single_task(test_kernel<3>()); });

        cl::sycl::kernel kernelB = std::move(kernelA);

        if (!ctsSelector.is_host() && kernelB.get() == nullptr) {
          FAIL(log,
               "kernel was not move assigned correctly. (contains different "
               "OpenCL kernel object)");
        }
      }

      /* Test equality operator
      */
      {
        cts_selector ctsSelector;
        auto ctsQueue = util::get_cts_object::queue(ctsSelector);

        cl::sycl::program prog(ctsQueue.get_context());
        prog.build_from_kernel_name<test_kernel<4>>();
        auto kernelA = prog.get_kernel<test_kernel<4>>();
        ctsQueue.submit(
            [&](cl::sycl::handler &cgh) { cgh.single_task(test_kernel<4>()); });

        cl::sycl::kernel kernelB(kernelA);
        cl::sycl::kernel kernelC = kernelA;
        if (!ctsSelector.is_host()) {
          if (kernelA == kernelB &&
              (kernelA.get() != kernelB.get() ||
               kernelA.get_context().get() != kernelB.get_context().get() ||
               kernelA.get_program().get() != kernelB.get_program().get())) {
            FAIL(log,
                 "kernel equality does not work correctly (copy constructed)");
          }
          if (kernelA == kernelC &&
              (kernelA.get() != kernelC.get() ||
               kernelA.get_context().get() != kernelC.get_context().get() ||
               kernelA.get_program().get() != kernelC.get_program().get())) {
            FAIL(log,
                 "kernel equality does not work correctly (copy assigned)");
          }
        }
      }

      /* Test hashing
      */
      {
        auto ctsQueue = util::get_cts_object::queue();

        cl::sycl::program prog(ctsQueue.get_context());
        prog.build_from_kernel_name<test_kernel<5>>();
        auto kernelA = prog.get_kernel<test_kernel<5>>();
        ctsQueue.submit(
            [&](cl::sycl::handler &cgh) { cgh.single_task(test_kernel<5>()); });

        cl::sycl::kernel kernelB = kernelA;

        cl::sycl::hash_class<cl::sycl::kernel> hasher;

        if (hasher(kernelA) != hasher(kernelB)) {
          FAIL(log,
               "kernel hashing does not work correctly (hashing of equal "
               "failed)");
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
