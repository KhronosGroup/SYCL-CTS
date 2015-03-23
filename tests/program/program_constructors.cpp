/*************************************************************************
//
//  SYCL Conformance Test Suite
//
//  Copyright:	(c) 2015 by Codeplay Software LTD. All Rights Reserved.
//
**************************************************************************/

#include "../common/common.h"

#define TEST_NAME program_constructors

namespace program_constructors__ {
using namespace sycl_cts;

/** tests the constructors for cl::sycl::platform
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
      /** check (context) constructor
      */
      {
        cts_selector selector;
        cl::sycl::context context;
        cl::sycl::program program(context);

        if (program.get() != 0) {
          FAIL(log, "program was not constructed correctly");
        }
      }

      /** check (context, vector_class<device>) constructor
      */
      {
        cts_selector selector;
        cl::sycl::context context;
        cl::sycl::vector_class<cl::sycl::device> deviceList =
            context.get_devices();
        cl::sycl::program program(context, deviceList);

        if (program.get() != 0) {
          FAIL(log, "program was not constructed correctly");
        }
      }

      /** check (vector_class<program>, string_class = "") constructor
      */
      {
        cts_selector selector;
        cl::sycl::context context;
        cl::sycl::program programA(context);
        cl::sycl::program programB(context);
        cl::sycl::vector_class<cl::sycl::program> programList;
        programList.push_back(programA);
        programList.push_back(programB);
        cl::sycl::program programC(programList);

        if (programC.get() != 0) {
          FAIL(log, "program was not constructed correctly");
        }
      }

      /** check (vector_class<program>, string_class) constructor
      */
      {
        cts_selector selector;
        cl::sycl::context context;
        cl::sycl::program programA(context);
        cl::sycl::program programB(context);
        cl::sycl::vector_class<cl::sycl::program> programList;
        programList.push_back(programA);
        programList.push_back(programB);
        cl::sycl::program programC(programList, "-cl-fast-relaxed-mat");

        if (programC.get() != 0) {
          FAIL(log, "program was not constructed correctly");
        }
      }

      /** check copy constructor
      */
      {
        cts_selector selector;
        cl::sycl::context context;
        cl::sycl::program programA(context);
        cl::sycl::program programB(programA);

        if (programA.get() != programB.get()) {
          FAIL(log, "program was not copied correctly.");
        }
      }

      /** check assignment operator
      */
      {
        cts_selector selector;
        cl::sycl::context context;
        cl::sycl::program programA(context);
        cl::sycl::program programB = programA;

        if (programA.get() != programB.get()) {
          FAIL(log, "program was not copied correctly.");
        }
      }
    } catch (cl::sycl::exception e) {
      log_exception(log, e);
      FAIL(log, "sycl exception caught");
    }
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace program_constructors__ */
