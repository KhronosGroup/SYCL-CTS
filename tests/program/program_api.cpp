/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#define TEST_NAME program_api

// Forward declaration of the kernel
template <int N>
struct program_kernel {
  void operator()() const {}
};

// Forward declaration of the kernel
struct program_api_kernel {
  void operator()() const {}
};

namespace program_api__ {
using namespace sycl_cts;
using namespace cl::sycl;

/** simple OpenCL test kernel
 */
cl::sycl::string_class kernel_source = R"(
__kernel void sample(__global float * input)
{
    input[get_global_id(0)] = get_global_id(0);
}
)";

/** test cl::sycl::program
 */
class TEST_NAME : public sycl_cts::util::test_base_opencl {
 public:
  /** return information about this test
   */
  virtual void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  virtual void run(util::logger &log) override {
    try {
      using namespace cl::sycl;

      auto context = util::get_cts_object::context();

      {
        log.note("check program class methods");

        cts_selector selector;
        auto context = util::get_cts_object::context(selector);
        program prog(context);
        prog.build_from_kernel_name<program_api_kernel>();

        // Check get_devices()
        if (prog.get_devices().size() < 1) {
          FAIL(log, "Wrong value for program.get_devices()");
        }

        // Check get_binaries()
        vector_class<vector_class<char>> binaries = prog.get_binaries();

        // Check get_binary_sizes()
        vector_class<::size_t> binarySizes = prog.get_binary_sizes();

        // Check get_build_options()
        string_class buildOptions = prog.get_build_options();

        // Check get()
        if (!context.is_host()) {
          cl_program clProgram = prog.get();
        }

        // Check get_kernel()
        {
          auto q = queue(selector);
          q.submit([](cl::sycl::handler &cgh) {
            cgh.single_task(program_api_kernel());
          });
          q.wait_and_throw();

          kernel k = prog.get_kernel<program_api_kernel>();
        }

        // Check is_linked()
        bool isLinked = prog.is_linked();
        if (!isLinked) {
          FAIL(log, "Program was not built properly (is_linked())");
        }
      }

      {
        log.note("build program without build options");

        auto myQueue = util::get_cts_object::queue();
        program prog(myQueue.get_context());

        if (prog.is_linked()) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        prog.build_from_kernel_name<program_kernel<0>>();

        if (!prog.is_linked()) {
          FAIL(log, "Program was not built properly (is_linked())");
        }

        // check for get_binaries()
        if (prog.get_binaries().size() < 1) {
          FAIL(log, "Wrong value for program.get_binaries()");
        }

        myQueue.submit(
            [&](handler &cgh) { cgh.single_task(program_kernel<0>()); });
      }

      {
        log.note("build program with build options");

        auto myQueue = util::get_cts_object::queue();

        program prog(myQueue.get_context());

        if (prog.is_linked()) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        prog.build_from_kernel_name<program_kernel<1>>();

        if (!prog.is_linked()) {
          FAIL(log, "Program was not built properly (is_linked())");
        }

        // check for get_binaries()
        if (prog.get_binaries().size() < 1) {
          FAIL(log, "Wrong value for program.get_binaries()");
        }

        myQueue.submit(
            [&](handler &cgh) { cgh.single_task(program_kernel<1>()); });
      }

      {
        log.note(
            "compile and link program without without compile and link "
            "options");

        auto myQueue = util::get_cts_object::queue();
        program prog(myQueue.get_context());

        if (prog.is_linked()) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        prog.compile_from_kernel_name<program_kernel<2>>();

        if (prog.is_linked()) {
          FAIL(log, "Program should not be linked after compilation");
        }

        // Check link()
        prog.link();

        if (!prog.is_linked()) {
          FAIL(log, "Program was not built properly (is_linked())");
        }

        // check for get_binaries()
        if (prog.get_binaries().size() < 1) {
          FAIL(log, "Wrong value for program.get_binaries()");
        }

        // check for get_build_options()
        if (prog.get_build_options().length() == 0) {
          FAIL(log, "program.get_build_options() shouldn't be empty");
        }

        myQueue.submit(
            [&](handler &cgh) { cgh.single_task(program_kernel<2>()); });
      }

      if (!context.is_host()) {
        log.note(
            "link an OpenCL and a SYCL program without compile and link "
            "options");

        auto myQueue = util::get_cts_object::queue();

        // obtain an existing OpenCL C program object
        cl_program myClProgram = nullptr;
        if (!create_compiled_program(kernel_source, myClProgram, log)) {
          FAIL(log, "Didn't create the cl_program");
        }

        // Create a SYCL program object from a cl_program object
        cl::sycl::program myExternProgram(myQueue.get_context(), myClProgram);

        if (myExternProgram.is_linked()) {
          FAIL(log, "Compiled interop program should not be linked yet");
        }

        // Add in the SYCL program object for our kernel
        cl::sycl::program mySyclProgram(myQueue.get_context());
        mySyclProgram.compile_from_kernel_name<program_kernel<3>>();

        if (mySyclProgram.is_linked()) {
          FAIL(log, "Compiled SYCL program should not be linked yet");
        }

        // Link myClProgram with the SYCL program object
        cl::sycl::program myLinkedProgram({myExternProgram, mySyclProgram});

        if (!myLinkedProgram.is_linked()) {
          FAIL(log, "Program was not linked");
        }

        myQueue.submit(
            [&](handler &cgh) { cgh.single_task(program_kernel<3>()); });
      }

      if (!context.is_host()) {
        log.note(
            "link an OpenCL and a SYCL program with compile and link options");

        auto myQueue = util::get_cts_object::queue();

        // obtain an existing OpenCL C program object
        cl_program myClProgram = nullptr;
        if (!create_compiled_program(kernel_source, myClProgram, log)) {
          FAIL(log, "Didn't create the cl_program");
        }

        // Create a SYCL program object from a cl_program object
        cl::sycl::program myExternProgram(myQueue.get_context(), myClProgram);

        if (myExternProgram.is_linked()) {
          FAIL(log, "Compiled interop program should not be linked yet");
        }

        // Add in the SYCL program object for our kernel
        cl::sycl::program mySyclProgram(myQueue.get_context());
        mySyclProgram.compile_from_kernel_name<program_kernel<4>>(
            "-cl-opt-disable");

        if (mySyclProgram.is_linked()) {
          FAIL(log, "Compiled SYCL program should not be linked yet");
        }

        // Link myClProgram with the SYCL program object
        cl::sycl::program myLinkedProgram({myExternProgram, mySyclProgram},
                                          "-cl-fast-relaxed-math");

        if (!myLinkedProgram.is_linked()) {
          FAIL(log, "Program was not linked");
        }

        myQueue.submit(
            [&](handler &cgh) { cgh.single_task(program_kernel<4>()); });
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      cl::sycl::string_class errorMsg =
          "a SYCL exception was caught: " + cl::sycl::string_class(e.what());
      FAIL(log, errorMsg.c_str());
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

} /* namespace program_api__ */
