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

/** test cl::sycl::program
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
      auto selector = cts_selector();
      auto context = util::get_cts_object::context(selector);
      auto deviceList = context.get_devices();

      // Do ALL devices support online compiler / linker?
      bool compiler_available = is_compiler_available(deviceList);
      bool linker_available = is_linker_available(deviceList);

      const cl::sycl::string_class compileOptions = "-cl-opt-disable";
      const cl::sycl::string_class linkOptions = "-cl-fast-relaxed-math";

      {
        log.note("check program class methods");

        cl::sycl::program prog(context);

        // Check build_with_kernel_type()
        try {
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
          cl::sycl::string_class progCompileOptions =
              prog.get_compile_options();

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

        } catch (const cl::sycl::feature_not_supported &fnse_build) {
          if (!compiler_available || !linker_available) {
            log.note(
                "online compiler or linker not available -- skipping check");
          } else {
            throw;
          }
        }
      }

      {
        log.note("build program without build options");

        auto myQueue = cl::sycl::queue(context, selector);
        cl::sycl::program prog(myQueue.get_context());

        if (prog.get_state() != cl::sycl::program_state::none) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        try {
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

        } catch (const cl::sycl::feature_not_supported &fnse_build) {
          if (!compiler_available || !linker_available) {
            log.note(
                "online compiler or linker not available -- skipping check");
          } else {
            throw;
          }
        }
      }

      {
        log.note("build program with build options");

        auto myQueue = cl::sycl::queue(context, selector);

        cl::sycl::program prog(myQueue.get_context());

        if (prog.get_state() != cl::sycl::program_state::none) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        try {
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

        } catch (const cl::sycl::feature_not_supported &fnse_build) {
          if (!compiler_available || !linker_available) {
            log.note(
                "online compiler or linker not available -- skipping check");
          } else {
            throw;
          }
        }
      }

      {
        log.note("compile and link program without compile and link options");

        auto myQueue = cl::sycl::queue(context, selector);
        cl::sycl::program prog(myQueue.get_context());

        if (prog.get_state() != cl::sycl::program_state::none) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        // Check compile_with_kernel_type()
        try {
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

        } catch (const cl::sycl::feature_not_supported &fnse_compile) {
          if (!compiler_available) {
            log.note("online compiler not available -- skipping check");
          } else {
            throw;
          }
        }
      }

      {
        log.note("compile and link program with compile and link options");

        auto myQueue = cl::sycl::queue(context, selector);
        cl::sycl::program prog(myQueue.get_context());

        if (prog.get_state() != cl::sycl::program_state::none) {
          FAIL(log, "Newly created program should not be linked yet");
        }

        // Check compile_with_kernel_type(options)
        try {
          prog.compile_with_kernel_type<program_kernel<3>>(compileOptions);

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

        } catch (const cl::sycl::feature_not_supported &fnse_compile) {
          if (!compiler_available) {
            log.note("online compiler not available -- skipping check");
          } else {
            throw;
          }
        }
      }

      {
        log.note(
            "link SYCL programs without compile and link "
            "options");

        if (!compiler_available) {
          log.note("online compiler not available -- skipping check");
        } else {
          auto myQueue = cl::sycl::queue(context, selector);

          // Create SYCL programs to link
          cl::sycl::program mySyclProgram1(myQueue.get_context());
          mySyclProgram1.compile_with_kernel_type<program_kernel<4>>();

          if (mySyclProgram1.get_state() != cl::sycl::program_state::compiled) {
            FAIL(log, "Compiled SYCL program should be in compiled state");
          }

          cl::sycl::program mySyclProgram2(myQueue.get_context());
          mySyclProgram2.compile_with_kernel_type<program_kernel<5>>();

          if (mySyclProgram2.get_state() != cl::sycl::program_state::compiled) {
            FAIL(log, "Compiled SYCL program should be in compiled state");
          }

          // Link SYCL programs
          try {
            cl::sycl::program myLinkedProgram({mySyclProgram1, mySyclProgram2});

            if (myLinkedProgram.get_state() !=
                cl::sycl::program_state::linked) {
              FAIL(log, "Program was not linked");
            }

            myQueue.submit([&](cl::sycl::handler &cgh) {
              cgh.single_task(program_kernel<4>());
            });
            myQueue.wait_and_throw();

            myQueue.submit([&](cl::sycl::handler &cgh) {
              cgh.single_task(program_kernel<5>());
            });
            myQueue.wait_and_throw();

          } catch (const cl::sycl::feature_not_supported &fnse_link) {
            if (!linker_available) {
              log.note("online linker not available -- skipping check");
            } else {
              throw;
            }
          }
        }
      }

      {
        log.note("link SYCL programs with compile and link options");

        if (!compiler_available) {
          log.note("online compiler not available -- skipping check");
        } else {
          auto myQueue = cl::sycl::queue(context, selector);

          // Create SYCL program objects with specified compile options
          cl::sycl::program mySyclProgram1(myQueue.get_context());
          mySyclProgram1.compile_with_kernel_type<program_kernel<6>>(
              compileOptions);

          if (mySyclProgram1.get_state() != cl::sycl::program_state::compiled) {
            FAIL(log, "Compiled SYCL program should be in compiled state");
          }

          if (mySyclProgram1.get_compile_options() != compileOptions) {
            FAIL(log,
                 "Compiled SYCL program did not store the compile options");
          }

          // Create SYCL program objects with specified compile options
          cl::sycl::program mySyclProgram2(myQueue.get_context());
          mySyclProgram2.compile_with_kernel_type<program_kernel<7>>(
              compileOptions);

          if (mySyclProgram2.get_state() != cl::sycl::program_state::compiled) {
            FAIL(log, "Compiled SYCL program should be in compiled state");
          }

          if (mySyclProgram2.get_compile_options() != compileOptions) {
            FAIL(log,
                 "Compiled SYCL program did not store the compile options");
          }

          // Link created SYCL programs using provided link options
          try {
            cl::sycl::program myLinkedProgram({mySyclProgram1, mySyclProgram2},
                                              linkOptions);

            if (myLinkedProgram.get_state() !=
                cl::sycl::program_state::linked) {
              FAIL(log, "Program was not linked");
            }

            if (myLinkedProgram.get_link_options() != linkOptions) {
              FAIL(log, "Linked program did not store the link options");
            }

            myQueue.submit([&](cl::sycl::handler &cgh) {
              cgh.single_task(program_kernel<6>());
            });
            myQueue.wait_and_throw();

            myQueue.submit([&](cl::sycl::handler &cgh) {
              cgh.single_task(program_kernel<7>());
            });
            myQueue.wait_and_throw();

          } catch (const cl::sycl::feature_not_supported &fnse_link) {
            if (!linker_available) {
              log.note("online linker not available -- skipping check");
            } else {
              throw;
            }
          }
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

} /* namespace program_api__ */
