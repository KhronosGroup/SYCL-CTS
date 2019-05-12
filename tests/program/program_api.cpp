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

/** simple OpenCL test kernel
 */
const cl::sycl::string_class kernelName = "sample";
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
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      auto selector = cts_selector();
      auto context = util::get_cts_object::context(selector);
      const cl::sycl::string_class compileOptions = "-cl-opt-disable";
      const cl::sycl::string_class linkOptions = "-cl-fast-relaxed-math";

      {
        log.note("check program class methods");

        cl::sycl::program prog(context);

        // Check build_with_kernel_type()
        prog.build_with_kernel_type<program_api_kernel>();

        // Check get_devices()
        if (prog.get_devices().size() < 1) {
          FAIL(log, "Wrong value for program.get_devices()");
        }

        // Check is_host()
        bool isHost = prog.is_host();

        // Check get_binaries()
        cl::sycl::vector_class<cl::sycl::vector_class<char>> binaries =
            prog.get_binaries();

        // Check get_context()
        cl::sycl::context progCtx = prog.get_context();

        // Check get_compile_options()
        cl::sycl::string_class progCompileOptions = prog.get_compile_options();

        // Check get_link_options()
        cl::sycl::string_class progLinkOptions = prog.get_link_options();

        // Check get_build_options()
        cl::sycl::string_class progBuildOptions = prog.get_build_options();

        // Check get()
        if (!context.is_host()) {
          cl_program clProgram = prog.get();
        }

        {
          auto q = cl::sycl::queue(context, selector);
          q.submit([](cl::sycl::handler &cgh) {
            cgh.single_task(program_api_kernel());
          });
          q.wait_and_throw();

          // Check has_kernel<>()
          bool hasKernel = prog.has_kernel<program_api_kernel>();
          if (!hasKernel) {
            FAIL(log, "Program was not built properly (has_kernel())");
          }

          // Check get_kernel<>()
          cl::sycl::kernel k = prog.get_kernel<program_api_kernel>();
        }

        // Check get_state()
        cl::sycl::program_state state = prog.get_state();
        if (state != cl::sycl::program_state::linked) {
          FAIL(log, "Program was not built properly (get_state())");
        }
      }

      if (!context.is_host()) {
        log.note("check compiling and building from source");

        {  // Check compile_with_source(source)
          cl::sycl::program prog(context);
          prog.compile_with_source(kernel_source);
        }
        {  // Check compile_with_source(source, options)
          cl::sycl::program prog(context);
          prog.compile_with_source(kernel_source, compileOptions);
        }
        {  // Check build_with_source(source)
          cl::sycl::program prog(context);
          prog.build_with_source(kernel_source);
        }
        {  // Check build_with_source(source, options)
          cl::sycl::program prog(context);
          prog.build_with_source(kernel_source, linkOptions);
        }

        {  // Check retrieveing kernel
          cl::sycl::program prog(context);
          prog.build_with_source(kernel_source);

          // Check has_kernel(string_class)
          bool hasKernel = prog.has_kernel(kernelName);
          if (!hasKernel) {
            FAIL(log,
                 "Program was not built properly (has_kernel(string_class))");
          }

          // Check get_kernel(string_class)
          cl::sycl::kernel k = prog.get_kernel(kernelName);
        }
      }

      {
        log.note("build program without build options");

        auto myQueue = cl::sycl::queue(context, selector);
        cl::sycl::program prog(myQueue.get_context());

        if (prog.get_state() != cl::sycl::program_state::none) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        prog.build_with_kernel_type<program_kernel<0>>();

        if (prog.get_state() != cl::sycl::program_state::linked) {
          FAIL(log, "Program was not built properly (get_state())");
        }

        // check for get_binaries()
        if (prog.get_binaries().size() < 1) {
          FAIL(log, "Wrong value for program.get_binaries()");
        }

        myQueue.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task(program_kernel<0>());
        });
        myQueue.wait_and_throw();
      }

      {
        log.note("build program with build options");

        auto myQueue = cl::sycl::queue(context, selector);

        cl::sycl::program prog(myQueue.get_context());

        if (prog.get_state() != cl::sycl::program_state::none) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        prog.build_with_kernel_type<program_kernel<1>>(linkOptions);

        if (prog.get_state() != cl::sycl::program_state::linked) {
          FAIL(log, "Program was not built properly (get_state())");
        }

        if (prog.get_build_options().find(linkOptions) ==
            cl::sycl::string_class::npos) {
          FAIL(log, "Built program did not store the build options");
        }

        // check for get_binaries()
        if (prog.get_binaries().size() < 1) {
          FAIL(log, "Wrong value for program.get_binaries()");
        }

        myQueue.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task(program_kernel<1>());
        });
        myQueue.wait_and_throw();
      }

      {
        log.note(
            "compile and link program without without compile and link "
            "options");

        auto myQueue = cl::sycl::queue(context, selector);
        cl::sycl::program prog(myQueue.get_context());

        if (prog.get_state() != cl::sycl::program_state::none) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        // Check compile_with_kernel_type()
        prog.compile_with_kernel_type<program_kernel<2>>();

        if (prog.get_state() != cl::sycl::program_state::compiled) {
          FAIL(log, "Program should be in compiled state after compilation");
        }

        // Check link()
        prog.link();

        if (prog.get_state() != cl::sycl::program_state::linked) {
          FAIL(log, "Program was not built properly (get_state())");
        }

        // check for get_binaries()
        if (prog.get_binaries().size() < 1) {
          FAIL(log, "Wrong value for program.get_binaries()");
        }

        // check for get_build_options()
        if (prog.get_build_options().length() != 0) {
          FAIL(log, "program.get_build_options() should be empty");
        }

        myQueue.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task(program_kernel<2>());
        });
        myQueue.wait_and_throw();
      }

      {
        log.note(
            "compile and link program without with compile and link "
            "options");

        auto myQueue = cl::sycl::queue(context, selector);
        cl::sycl::program prog(myQueue.get_context());

        if (prog.get_state() != cl::sycl::program_state::none) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        // Check compile_with_kernel_type(options)
        prog.compile_with_kernel_type<program_kernel<2>>(compileOptions);

        if (prog.get_state() != cl::sycl::program_state::compiled) {
          FAIL(log, "Program should be in compiled state after compilation");
        }

        // Check link(options)
        prog.link(linkOptions);

        if (prog.get_state() != cl::sycl::program_state::linked) {
          FAIL(log, "Program was not built properly (get_state())");
        }

        // check for get_binaries()
        if (prog.get_binaries().size() < 1) {
          FAIL(log, "Wrong value for program.get_binaries()");
        }

        // check for get_build_options()
        if (prog.get_build_options().length() == 0) {
          FAIL(log, "program.get_build_options() shouldn't be empty");
        }

        // check compile options
        if (prog.get_compile_options() != compileOptions) {
          FAIL(log, "Linked program did not store the compile options");
        }

        // check link options
        if (prog.get_link_options() != linkOptions) {
          FAIL(log, "Linked program did not store the link options");
        }

        myQueue.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task(program_kernel<3>());
        });
        myQueue.wait_and_throw();
      }

      if (!context.is_host()) {
        log.note(
            "link an OpenCL and a SYCL program without compile and link "
            "options");

        auto myQueue = cl::sycl::queue(context, selector);

        // obtain an existing OpenCL C program object
        cl_program myClProgram = nullptr;
        if (!create_compiled_program(kernel_source, context.get(),
                                     myQueue.get_device().get(), myClProgram,
                                     log)) {
          FAIL(log, "Didn't create the cl_program");
        }

        // Create a SYCL program object from a cl_program object
        cl::sycl::program myExternProgram(myQueue.get_context(), myClProgram);

        if (myExternProgram.get_state() != cl::sycl::program_state::compiled) {
          FAIL(log, "Compiled interop program should be in compiled state");
        }

        // Add in the SYCL program object for our kernel
        cl::sycl::program mySyclProgram(myQueue.get_context());
        mySyclProgram.compile_with_kernel_type<program_kernel<3>>();

        if (mySyclProgram.get_state() != cl::sycl::program_state::compiled) {
          FAIL(log, "Compiled SYCL program should be in compiled state");
        }

        // Link myClProgram with the SYCL program object
        cl::sycl::program myLinkedProgram({myExternProgram, mySyclProgram});

        if (myLinkedProgram.get_state() != cl::sycl::program_state::linked) {
          FAIL(log, "Program was not linked");
        }

        myQueue.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task(program_kernel<4>());
        });
        myQueue.wait_and_throw();
      }

      if (!context.is_host()) {
        log.note(
            "link an OpenCL and a SYCL program with compile and link options");

        auto myQueue = cl::sycl::queue(context, selector);

        // obtain an existing OpenCL C program object
        cl_program myClProgram = nullptr;
        if (!create_compiled_program(kernel_source, context.get(),
                                     myQueue.get_device().get(), myClProgram,
                                     log)) {
          FAIL(log, "Didn't create the cl_program");
        }

        // Create a SYCL program object from a cl_program object
        cl::sycl::program myExternProgram(myQueue.get_context(), myClProgram);

        if (myExternProgram.get_state() != cl::sycl::program_state::compiled) {
          FAIL(log, "Compiled interop program should be in compiled state");
        }

        // Add in the SYCL program object for our kernel
        cl::sycl::program mySyclProgram(myQueue.get_context());
        mySyclProgram.compile_with_kernel_type<program_kernel<4>>(
            compileOptions);

        if (mySyclProgram.get_state() != cl::sycl::program_state::compiled) {
          FAIL(log, "Compiled SYCL program should be in compiled state");
        }

        if (mySyclProgram.get_compile_options() != compileOptions) {
          FAIL(log, "Compiled SYCL program did not store the compile options");
        }

        // Link myClProgram with the SYCL program object
        cl::sycl::program myLinkedProgram({myExternProgram, mySyclProgram},
                                          linkOptions);

        if (myLinkedProgram.get_state() != cl::sycl::program_state::linked) {
          FAIL(log, "Program was not linked");
        }

        if (myLinkedProgram.get_link_options() != linkOptions) {
          FAIL(log, "Linked program did not store the link options");
        }

        myQueue.submit([&](cl::sycl::handler &cgh) {
          cgh.single_task(program_kernel<5>());
        });
        myQueue.wait_and_throw();
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

} /* namespace program_api__ */
