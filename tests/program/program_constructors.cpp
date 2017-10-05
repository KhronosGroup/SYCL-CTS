/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME program_constructors

// Forward declaration of the kernel
template <int N>
struct program_ctrs_kernel {
  void operator()() const {}
};

class test_functor_1 {
  cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer>
      m_acc;

 public:
  test_functor_1(
      cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>
          acc)
      : m_acc(acc) {}

  void operator()() { m_acc[0] *= 2.0f; };
};

namespace program_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::platform
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
      log.note("check (context) constructor");
      {
        auto context = util::get_cts_object::context();
        cl::sycl::program program(context);

        if (!context.is_host() && (program.get() != nullptr)) {
          FAIL(log, "program was not constructed correctly");
        }
      }

      log.note("check (context, vector_class<device>) constructor");
      {
        auto context = util::get_cts_object::context();
        cl::sycl::vector_class<cl::sycl::device> deviceList =
            context.get_devices();
        cl::sycl::program program(context, deviceList);

        if (!context.is_host() && (program.get() != nullptr)) {
          FAIL(log, "program was not constructed correctly");
        }
      }

      log.note(
          "check (vector_class<program>, string_class = \"\") constructor");
      {
        auto context = util::get_cts_object::context();
        auto programA = util::get_cts_object::program::compiled<
            struct program_ctrs_kernel<0>>(context);
        auto programB = util::get_cts_object::program::compiled<
            struct program_ctrs_kernel<1>>(context);

        cl::sycl::vector_class<cl::sycl::program> programList;
        programList.push_back(programA);
        programList.push_back(programB);
        cl::sycl::program programC(programList);

        if (!context.is_host() && (programC.get() == nullptr)) {
          FAIL(log, "program was not constructed correctly");
        }
      }

      log.note("check (vector_class<program>, string_class) constructor");
      {
        auto context = util::get_cts_object::context();
        auto programA = util::get_cts_object::program::compiled<
            struct program_ctrs_kernel<2>>(context);
        auto programB = util::get_cts_object::program::compiled<
            struct program_ctrs_kernel<3>>(context);

        cl::sycl::vector_class<cl::sycl::program> programList;
        programList.push_back(programA);
        programList.push_back(programB);
        cl::sycl::program programC(programList, "-cl-fast-relaxed-math");

        if (!context.is_host() && (programC.get() == nullptr)) {
          FAIL(log, "program was not constructed correctly");
        }
      }

      log.note("check copy constructor");
      {
        auto context = util::get_cts_object::context();
        auto programA = util::get_cts_object::program::compiled<
            struct program_ctrs_kernel<4>>(context);

        cl::sycl::program programB(programA);

        if (!context.is_host() && (programB.get() == nullptr)) {
          FAIL(log, "program was not copy constructed correctly. (get)");
        }
        if (programA.is_linked() != programB.is_linked()) {
          FAIL(log, "program was not copy constructed correctly. (is_linked)");
        }
      }

      log.note("check assignment operator");
      {
        auto context = util::get_cts_object::context();
        auto programA = util::get_cts_object::program::compiled<
            struct program_ctrs_kernel<5>>(context);

        cl::sycl::program programB = programA;

        if (!context.is_host() && (programB.get() == nullptr)) {
          FAIL(log, "program was not copy assigned correctly. (get)");
        }
        if (programA.is_linked() != programB.is_linked()) {
          FAIL(log, "program was not copy assigned correctly. (is_linked)");
        }
      }

      log.note("check move constructor");
      {
        auto context = util::get_cts_object::context();
        auto programA = util::get_cts_object::program::compiled<
            struct program_ctrs_kernel<6>>(context);

        cl::sycl::program programB(std::move(programA));

        if (!context.is_host() && (programB.get() == nullptr)) {
          FAIL(log, "program was not move constructed correctly. (get)");
        }
        if (programB.is_linked()) {
          FAIL(log, "program was not move constructed correctly. (is_linked)");
        }
      }

      log.note("check move assignment operator");
      {
        auto context = util::get_cts_object::context();
        auto programA = util::get_cts_object::program::compiled<
            struct program_ctrs_kernel<7>>(context);

        cl::sycl::program programB = std::move(programA);

        if (!context.is_host() && (programB.get() == nullptr)) {
          FAIL(log, "program was not move assigned correctly. (get)");
        }
        if (programB.is_linked()) {
          FAIL(log, "program was not move assigned correctly. (is_linked)");
        }
      }

      log.note("check equality operator");
      {
        cts_selector selector;
        auto context = util::get_cts_object::context(selector);

        cl::sycl::program programA(context);
        cl::sycl::program programB(programA);
        cl::sycl::program programC = programA;

        if (!(programA == programB) &&
            (context.is_host() && (programA.get() != programB.get()))) {
          FAIL(log,
               "program equality does not work correctly. (copy constructed)");
        }
        if (!(programA == programC) &&
            (context.is_host() && (programA.get() == programC.get()))) {
          FAIL(log,
               "program equality does not work correctly. (copy assigned)");
        }
      }

      log.note("check hashing");
      {
        cts_selector selector;
        auto context = util::get_cts_object::context(selector);

        cl::sycl::program programA(context);
        cl::sycl::program programB(programA);

        cl::sycl::hash_class<cl::sycl::program> hasher;

        if (hasher(programA) != hasher(programB)) {
          FAIL(log,
               "program hashing does not work correctly (hashing of equal "
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

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace program_constructors__ */
